from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from transformers import AutoTokenizer

from ttt.config import Config, register_configs
from ttt.infra.checkpoint import Checkpointer, unify_dict_with_eqx_module
from ttt.model.data import Batch
from ttt.model.sharding import ModelSharding
from ttt.model.transformer import MetaModel
from ttt.utils.jax_utils import eval_shape_and_sharding, initialize_distibuted, set_random_seed

register_configs()

RULER_REPO = "https://github.com/NVIDIA/RULER.git"
DEFAULT_SYNTHETIC_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _ensure_ruler(ruler_dir: Path) -> None:
    if ruler_dir.exists():
        return
    ruler_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", RULER_REPO, str(ruler_dir)], cwd=ruler_dir.parent, env=os.environ.copy())


def _ensure_eval_compat(compat_dir: Path) -> Path:
    """Provide just the NeMo manifest helpers official RULER eval imports."""
    manifest_path = compat_dir / "nemo" / "collections" / "asr" / "parts" / "utils" / "manifest_utils.py"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for parent in manifest_path.parents:
        if parent == compat_dir.parent:
            break
        init_file = parent / "__init__.py"
        init_file.touch(exist_ok=True)
    manifest_path.write_text(
        "\n".join(
            [
                "import json",
                "",
                "def read_manifest(path):",
                "    with open(path, encoding='utf-8') as f:",
                "        return [json.loads(line) for line in f if line.strip()]",
                "",
                "def write_manifest(path, target_manifest, ensure_ascii=True):",
                "    with open(path, 'w', encoding='utf-8') as f:",
                "        for row in target_manifest:",
                "            json.dump(row, f, ensure_ascii=ensure_ascii)",
                "            f.write('\\n')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return compat_dir


def _download_ruler_aux_data(ruler_dir: Path, env: dict[str, str]) -> None:
    json_dir = ruler_dir / "scripts" / "data" / "synthetic" / "json"
    if not (json_dir / "PaulGrahamEssays.json").exists():
        _run([sys.executable, "download_paulgraham_essay.py"], cwd=json_dir, env=env)
    if not (json_dir / "squad.json").exists() or not (json_dir / "hotpotqa.json").exists():
        _run(["bash", "download_qa_dataset.sh"], cwd=json_dir, env=env)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _compose_config(args: argparse.Namespace) -> Config:
    overrides = [
        "+deploy=interactive",
        f"+experiment={args.experiment}",
        f"training.checkpoint_path={args.checkpoint_path}",
        f"backend.num_devices={args.num_devices}",
        f"backend.compilation_cache_dir={args.jax_cache_dir}",
        "training.log_wandb=false",
        "training.load_part=params",
    ]
    if args.wandb_entity:
        overrides.append(f"training.wandb_entity={args.wandb_entity}")
    if args.wandb_project:
        overrides.append(f"training.wandb_project={args.wandb_project}")
    if args.wandb_key:
        overrides.append(f"training.wandb_key={args.wandb_key}")

    with hydra.initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    with open_dict(cfg):
        cfg.training.resume_exp_name = args.checkpoint_exp_name or cfg.training.exp_name
        cfg.training.n_data_parallel = args.num_devices
        cfg.training.n_state_parallel = 1
        cfg.model.seq_len = args.max_seq_length
        cfg.training.seq_length = args.max_seq_length

    OmegaConf.resolve(cfg)
    return cfg


def _load_model(cfg: Config) -> tuple[MetaModel, eqx.nn.State, jax.sharding.Mesh]:
    initialize_distibuted(cfg.backend)
    key = set_random_seed(cfg.training.model_seed)
    model_sharding = ModelSharding(cfg)
    mesh = model_sharding.mesh

    @eqx.filter_jit
    def create_sharded_model_and_state() -> tuple[MetaModel, eqx.nn.State]:
        model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
        state = jax.device_put(state, jax.NamedSharding(mesh, jax.sharding.PartitionSpec()))
        model = model_sharding.shard_params(model)
        return model, state

    with mesh:
        abstract_model_weights = eval_shape_and_sharding(lambda: create_sharded_model_and_state()[0].weights())
        checkpointer = Checkpointer(config=cfg, for_saving=False)
        out_state = checkpointer.load_checkpoint(
            step=cfg.training.resume_step,
            targets={"model_weights": abstract_model_weights},
            restore=cfg.training.load_part,
        )
        model, state = create_sharded_model_and_state()
        model = unify_dict_with_eqx_module(out_state["model_weights"], model)[0]
        checkpointer.close()
    return model, state, mesh


@eqx.filter_jit
def _next_token(model: MetaModel, state: eqx.nn.State, input_ids: jax.Array, token_index: jax.Array) -> jax.Array:
    target_tokens = jnp.concatenate([input_ids[1:], input_ids[-1:]], axis=0)
    seq = Batch(
        input_ids=input_ids,
        target_tokens=target_tokens,
        loss_masks=jnp.ones(input_ids.shape, dtype=bool),
    )
    outputs = model.language_model.model(state, seq)
    hidden = outputs.last_hidden_state[token_index]
    logits = model.language_model.wte_disembed_call(hidden)
    return jnp.argmax(logits).astype(jnp.int32)


def _generate_greedy(
    model: MetaModel,
    state: eqx.nn.State,
    tokenizer,
    prompt: str,
    *,
    max_seq_length: int,
    tokens_to_generate: int,
    eos_token_id: int | None,
) -> str:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    max_prompt_len = max_seq_length - tokens_to_generate
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    token_buffer = jnp.zeros((max_seq_length,), dtype=jnp.int32)
    token_buffer = token_buffer.at[: len(prompt_ids)].set(jnp.asarray(prompt_ids, dtype=jnp.int32))
    generated: list[int] = []
    cur_len = len(prompt_ids)

    for _ in range(tokens_to_generate):
        token = int(jax.device_get(_next_token(model, state, token_buffer, jnp.asarray(cur_len - 1, dtype=jnp.int32))))
        if eos_token_id is not None and token == eos_token_id:
            break
        generated.append(token)
        if cur_len >= max_seq_length:
            break
        token_buffer = token_buffer.at[cur_len].set(token)
        cur_len += 1

    return tokenizer.decode(generated, skip_special_tokens=True)


def _ruler_task_tokens_to_generate(ruler_dir: Path, task: str) -> int:
    import yaml

    constants: dict[str, dict] = {}
    exec((ruler_dir / "scripts" / "data" / "synthetic" / "constants.py").read_text(encoding="utf-8"), constants)
    with (ruler_dir / "scripts" / "synthetic.yaml").open(encoding="utf-8") as f:
        task_cfg = yaml.safe_load(f)[task]
    return int(constants["TASKS"][task_cfg["task"]]["tokens_to_generate"])


def _prepare_task(ruler_dir: Path, task: str, args: argparse.Namespace, env: dict[str, str], data_dir: Path) -> None:
    task_file = data_dir / task / "validation.jsonl"
    _run(
        [
            sys.executable,
            "data/prepare.py",
            "--save_dir",
            str(data_dir),
            "--benchmark",
            "synthetic",
            "--task",
            task,
            "--tokenizer_path",
            args.tokenizer_name,
            "--tokenizer_type",
            "hf",
            "--max_seq_length",
            str(args.max_seq_length),
            "--model_template_type",
            "base",
            "--num_samples",
            str(args.num_samples),
        ],
        cwd=ruler_dir / "scripts",
        env=env,
    )
    if not task_file.exists():
        raise FileNotFoundError(f"Official RULER did not create {task_file}")


def _predict_task(
    model: MetaModel,
    state: eqx.nn.State,
    tokenizer,
    ruler_dir: Path,
    task: str,
    args: argparse.Namespace,
    data_dir: Path,
    pred_dir: Path,
) -> None:
    task_file = data_dir / task / "validation.jsonl"
    pred_file = pred_dir / f"{task}.jsonl"
    done = {row["index"] for row in _read_jsonl(pred_file)} if pred_file.exists() else set()
    rows = [row for row in _read_jsonl(task_file) if row["index"] not in done]
    tokens_to_generate = _ruler_task_tokens_to_generate(ruler_dir, task)
    if args.tokens_to_generate_limit is not None:
        tokens_to_generate = min(tokens_to_generate, args.tokens_to_generate_limit)

    outputs = []
    for row in tqdm(rows, desc=f"Predicting {task}"):
        pred = _generate_greedy(
            model,
            state,
            tokenizer,
            row["input"],
            max_seq_length=args.max_seq_length,
            tokens_to_generate=tokens_to_generate,
            eos_token_id=tokenizer.eos_token_id,
        )
        outputs.append(
            {
                "index": row["index"],
                "pred": pred,
                "input": row["input"],
                "outputs": row["outputs"],
                "others": row.get("others", {}),
                "truncation": row.get("truncation", -1),
                "length": row.get("length", -1),
            }
        )
        _append_jsonl(pred_file, outputs)
        outputs = []


def _evaluate(ruler_dir: Path, pred_dir: Path, env: dict[str, str]) -> None:
    _run(
        [sys.executable, "eval/evaluate.py", "--data_dir", str(pred_dir), "--benchmark", "synthetic"],
        cwd=ruler_dir / "scripts",
        env=env,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint-exp-name", default="")
    parser.add_argument("--checkpoint-path", default="./checkpoints")
    parser.add_argument("--output-root", type=Path, default=Path("./ruler_results"))
    parser.add_argument("--ruler-dir", type=Path, default=Path("./NVIDIA_RULER"))
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--max-seq-length", type=int, default=131072)
    parser.add_argument("--tokenizer-name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--tokens-to-generate-limit", type=int)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--jax-cache-dir", default="/tmp/jax_cache")
    parser.add_argument("--download-aux-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-entity", default="miki-aisle")
    parser.add_argument("--wandb-project", default="thesis-125m")
    parser.add_argument("--wandb-key", default=os.environ.get("WANDB_API_KEY", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

    ruler_dir = args.ruler_dir.resolve()
    _ensure_ruler(ruler_dir)
    compat_dir = _ensure_eval_compat(args.output_root.resolve() / "_compat")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{compat_dir}:{env.get('PYTHONPATH', '')}"
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    if args.download_aux_data:
        _download_ruler_aux_data(ruler_dir, env)

    tasks = DEFAULT_SYNTHETIC_TASKS if args.tasks == "all" else [task.strip() for task in args.tasks.split(",") if task.strip()]
    run_name = args.checkpoint_exp_name or Path(args.experiment).name
    result_dir = args.output_root / run_name / "synthetic" / str(args.max_seq_length)
    data_dir = result_dir / "data"
    pred_dir = result_dir / "pred"
    data_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        _prepare_task(ruler_dir, task, args, env, data_dir)

    cfg = _compose_config(args)
    model, state, mesh = _load_model(cfg)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    with mesh:
        for task in tasks:
            _predict_task(model, state, tokenizer, ruler_dir, task, args, data_dir, pred_dir)

    _evaluate(ruler_dir, pred_dir, env)


if __name__ == "__main__":
    main()
