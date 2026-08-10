from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ttt.config import Config, register_configs
from ttt.infra.checkpoint import Checkpointer, unify_dict_with_eqx_module
from ttt.model.data import Batch
from ttt.model.sharding import ModelSharding
from ttt.model.transformer import BlockCollectionSplit, MetaModel
from ttt.utils.filter_utils import get_filter_spec
from ttt.utils.jax_utils import clone_pytree, eval_shape_and_sharding, initialize_distibuted, scan_remat_chunk, set_random_seed, tree_rearrange

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


def _make_truncated_tokenizer(original, target_vocab_size: int):
    """Mirror the training-time truncated Llama-3 BPE tokenizer."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE

    tok_json = json.loads(original.backend_tokenizer.to_str())
    original_merges = tok_json["model"]["merges"]
    original_vocab = tok_json["model"]["vocab"]

    def parts(merge):
        return tuple(merge.split(" ")) if isinstance(merge, str) else tuple(merge)

    all_merge_results = {"".join(parts(merge)) for merge in original_merges}
    added_strs = {token["content"] for token in tok_json.get("added_tokens", [])}
    base_tokens = sorted(
        [
            (token, token_id)
            for token, token_id in original_vocab.items()
            if token not in all_merge_results and token not in added_strs
        ],
        key=lambda item: item[1],
    )

    num_merges = target_vocab_size - len(base_tokens)
    if num_merges <= 0:
        raise ValueError(f"target_vocab_size={target_vocab_size} is too small for {len(base_tokens)} base tokens")

    valid_tokens = {token for token, _ in base_tokens}
    kept_merges: list[tuple[str, str]] = []
    for merge in original_merges:
        if len(kept_merges) >= num_merges:
            break
        left, right = parts(merge)
        result = left + right
        if left in valid_tokens and right in valid_tokens and result not in valid_tokens:
            valid_tokens.add(result)
            kept_merges.append((left, right))

    new_vocab = {token: i for i, (token, _) in enumerate(base_tokens)}
    for left, right in kept_merges:
        new_vocab[left + right] = len(new_vocab)

    tokenizer = Tokenizer(BPE(vocab=new_vocab, merges=kept_merges))
    tokenizer.pre_tokenizer = original.backend_tokenizer.pre_tokenizer
    tokenizer.decoder = original.backend_tokenizer.decoder
    return tokenizer


def _load_tokenizer(tokenizer_name: str, cfg: Config):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    target_vocab_size = int(cfg.model.vocab_size)
    if target_vocab_size >= tokenizer.vocab_size:
        return tokenizer, tokenizer_name

    truncated = _make_truncated_tokenizer(tokenizer, target_vocab_size)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=truncated,
        clean_up_tokenization_spaces=False,
    )

    tokenizer_dir = Path(tempfile.mkdtemp(prefix="ruler_tokenizer_"))
    fast_tokenizer.save_pretrained(tokenizer_dir)
    print(f"Using truncated tokenizer {tokenizer.vocab_size} -> {target_vocab_size} at {tokenizer_dir}")
    return fast_tokenizer, str(tokenizer_dir)


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
        state = state.set(model.step_index, jnp.array(jnp.iinfo(jnp.int32).max - 100, dtype=jnp.int32))
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


@eqx.filter_jit
def _adapt_e2e_model(model: MetaModel, state: eqx.nn.State, input_ids: jax.Array, loss_token_count: jax.Array) -> MetaModel:
    cfg = model.config
    tokens_per_chunk = cfg.model.mini_batch_size

    target_tokens = jnp.concatenate([input_ids[1:], input_ids[-1:]], axis=0)
    loss_masks = jnp.arange(input_ids.shape[0]) < jnp.maximum(loss_token_count, 0)
    seq = Batch(input_ids=input_ids, target_tokens=target_tokens, loss_masks=loss_masks)

    block_collection = model.language_model.model.h.blocks
    prime_storage = model.language_model.model.h.prime_storage
    new_collection = BlockCollectionSplit(
        cfg.model,
        block_collection=block_collection,
        prime_storage=prime_storage,
        key=jax.random.PRNGKey(0),
    )

    state_prefix_suffix = state.substate(model.language_model.model.h.blocks)
    state_prefix, state_suffix = BlockCollectionSplit.split_state(state_prefix_suffix, cfg.model.suffix_len)
    state_all = clone_pytree(state)

    model = eqx.tree_at(lambda m: m.language_model.model.h, model, new_collection)
    model = jax.tree.map(lambda p: p.astype(model.state_dtype), model)
    inner_opt_state = model.inner_optimizer(state_all).init(model.inner_parameters())

    xt_embed = model.language_model.wte_call(seq.input_ids)
    prefix_output = model.language_model.prefix_call(model.language_model.model.h.prefix_blocks, xt_embed, state_prefix, seq).last_hidden_state

    def process_suffix_chunk(model__opt_state__state, inputs: tuple[Batch, jnp.ndarray]):
        model_inner, inner_opt_state, state_tuple = model__opt_state__state
        suffix_chunk, prefix_chunk = inputs

        spec_inner = get_filter_spec(model_inner, cfg.training.spec_inner, "inner parameters")
        inner_params, _ = eqx.partition(model_inner, spec_inner)
        _, outer_params = eqx.partition(model, spec_inner)
        model_inner = eqx.combine(inner_params, outer_params)

        new_model, inner_opt_state, state_tuple, _metrics = MetaModel.inner_loop_step(
            model_inner, inner_opt_state, state_tuple, suffix_chunk, prefix_chunk
        )
        return (new_model, inner_opt_state, state_tuple), None

    seq = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)
    prefix_output = tree_rearrange(prefix_output, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)
    (adapted_model, _inner_opt_state, _state_tuple), _ = scan_remat_chunk(
        process_suffix_chunk,
        (model, inner_opt_state, (state_all, state_suffix)),
        (seq, prefix_output),
        remat_n_loops=cfg.training.inner_remat_freq,
        unroll=cfg.model.unroll_inner_scan,
    )
    return adapted_model


@eqx.filter_jit
def _next_token_e2e_logits(model: MetaModel, state: eqx.nn.State, input_ids: jax.Array, token_index: jax.Array) -> jax.Array:
    cfg = model.config
    tokens_per_chunk = cfg.model.mini_batch_size

    target_tokens = jnp.concatenate([input_ids[1:], input_ids[-1:]], axis=0)
    seq = Batch(
        input_ids=input_ids,
        target_tokens=target_tokens,
        loss_masks=jnp.ones(input_ids.shape, dtype=bool),
    )

    block_collection = model.language_model.model.h.blocks
    prime_storage = model.language_model.model.h.prime_storage
    new_collection = BlockCollectionSplit(
        cfg.model,
        block_collection=block_collection,
        prime_storage=prime_storage,
        key=jax.random.PRNGKey(0),
    )

    state_prefix_suffix = state.substate(model.language_model.model.h.blocks)
    state_prefix, state_suffix = BlockCollectionSplit.split_state(state_prefix_suffix, cfg.model.suffix_len)
    state_all = clone_pytree(state)

    model = eqx.tree_at(lambda m: m.language_model.model.h, model, new_collection)
    model = jax.tree.map(lambda p: p.astype(model.state_dtype), model)
    inner_opt_state = model.inner_optimizer(state_all).init(model.inner_parameters())

    xt_embed = model.language_model.wte_call(seq.input_ids)
    prefix_output = model.language_model.prefix_call(model.language_model.model.h.prefix_blocks, xt_embed, state_prefix, seq).last_hidden_state

    token_chunk = token_index // tokens_per_chunk
    token_offset = token_index % tokens_per_chunk

    seq = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)
    prefix_output = tree_rearrange(prefix_output, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)
    chunk_ids = jnp.arange(seq.input_ids.shape[0], dtype=jnp.int32)

    def process_suffix_chunk(model__opt_state__state, inputs: tuple[Batch, jnp.ndarray, jax.Array]):
        model_inner, inner_opt_state, state_tuple = model__opt_state__state
        suffix_chunk, prefix_chunk, chunk_id = inputs

        spec_inner = get_filter_spec(model_inner, cfg.training.spec_inner, "inner parameters")
        inner_params, _ = eqx.partition(model_inner, spec_inner)
        _, outer_params = eqx.partition(model, spec_inner)
        model_inner = eqx.combine(inner_params, outer_params)

        state_all, suffix_state = state_tuple
        lm_outputs = model_inner.language_model.suffix_call(prefix_chunk, suffix_state, suffix_chunk)
        chunk_logits = lm_outputs.logits[token_offset]
        selected_logits = jnp.where(chunk_id == token_chunk, chunk_logits, jnp.zeros_like(chunk_logits))

        def update_after_chunk():
            new_model, new_inner_opt_state, new_state_tuple, _metrics = MetaModel.inner_loop_step(
                model_inner, inner_opt_state, state_tuple, suffix_chunk, prefix_chunk
            )
            return new_model, new_inner_opt_state, new_state_tuple

        def carry_forward_without_update():
            return model_inner, inner_opt_state, (state_all, lm_outputs.new_state)

        new_carry = jax.lax.cond(chunk_id < token_chunk, update_after_chunk, carry_forward_without_update)
        return new_carry, selected_logits

    _carry, selected_logits = scan_remat_chunk(
        process_suffix_chunk,
        (model, inner_opt_state, (state_all, state_suffix)),
        (seq, prefix_output, chunk_ids),
        remat_n_loops=cfg.training.inner_remat_freq,
        unroll=cfg.model.unroll_inner_scan,
    )
    logits = jnp.sum(selected_logits, axis=0)
    return logits


@eqx.filter_jit
def _next_token_e2e_no_ttt_logits(model: MetaModel, state: eqx.nn.State, input_ids: jax.Array, token_index: jax.Array) -> jax.Array:
    cfg = model.config
    tokens_per_chunk = cfg.model.mini_batch_size

    target_tokens = jnp.concatenate([input_ids[1:], input_ids[-1:]], axis=0)
    seq = Batch(
        input_ids=input_ids,
        target_tokens=target_tokens,
        loss_masks=jnp.ones(input_ids.shape, dtype=bool),
    )

    block_collection = model.language_model.model.h.blocks
    prime_storage = model.language_model.model.h.prime_storage
    new_collection = BlockCollectionSplit(
        cfg.model,
        block_collection=block_collection,
        prime_storage=prime_storage,
        key=jax.random.PRNGKey(0),
    )

    state_prefix_suffix = state.substate(model.language_model.model.h.blocks)
    state_prefix, state_suffix = BlockCollectionSplit.split_state(state_prefix_suffix, cfg.model.suffix_len)
    model = eqx.tree_at(lambda m: m.language_model.model.h, model, new_collection)

    xt_embed = model.language_model.wte_call(seq.input_ids)
    prefix_output = model.language_model.prefix_call(model.language_model.model.h.prefix_blocks, xt_embed, state_prefix, seq).last_hidden_state

    token_chunk = token_index // tokens_per_chunk
    token_offset = token_index % tokens_per_chunk

    seq = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)
    prefix_output = tree_rearrange(prefix_output, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)
    chunk_ids = jnp.arange(seq.input_ids.shape[0], dtype=jnp.int32)

    def process_suffix_chunk(state_suffix, inputs: tuple[Batch, jnp.ndarray, jax.Array]):
        suffix_chunk, prefix_chunk, chunk_id = inputs
        lm_outputs = model.language_model.suffix_call(prefix_chunk, state_suffix, suffix_chunk)
        chunk_logits = lm_outputs.logits[token_offset]
        selected_logits = jnp.where(chunk_id == token_chunk, chunk_logits, jnp.zeros_like(chunk_logits))
        return lm_outputs.new_state, selected_logits

    _state_suffix, selected_logits = scan_remat_chunk(
        process_suffix_chunk,
        state_suffix,
        (seq, prefix_output, chunk_ids),
        remat_n_loops=cfg.training.inner_remat_freq,
        unroll=cfg.model.unroll_inner_scan,
    )
    logits = jnp.sum(selected_logits, axis=0)
    return logits


def _next_token_e2e(model: MetaModel, state: eqx.nn.State, input_ids: jax.Array, token_index: jax.Array) -> jax.Array:
    logits = _next_token_e2e_logits(model, state, input_ids, token_index)
    return jnp.argmax(logits).astype(jnp.int32)


def _next_token_e2e_no_ttt(model: MetaModel, state: eqx.nn.State, input_ids: jax.Array, token_index: jax.Array) -> jax.Array:
    logits = _next_token_e2e_no_ttt_logits(model, state, input_ids, token_index)
    return jnp.argmax(logits).astype(jnp.int32)


def _validate_e2e_ttt(model: MetaModel, state: eqx.nn.State, input_ids: jax.Array) -> None:
    chunk_size = model.config.model.mini_batch_size
    before_update = jnp.asarray(chunk_size - 1, dtype=jnp.int32)
    after_update = jnp.asarray(2 * chunk_size - 1, dtype=jnp.int32)

    active_before = _next_token_e2e_logits(model, state, input_ids, before_update)
    off_before = _next_token_e2e_no_ttt_logits(model, state, input_ids, before_update)
    np.testing.assert_allclose(np.asarray(active_before), np.asarray(off_before), rtol=0, atol=0)

    active_after = _next_token_e2e_logits(model, state, input_ids, after_update)
    off_after = _next_token_e2e_no_ttt_logits(model, state, input_ids, after_update)
    update_delta = float(jnp.max(jnp.abs(active_after - off_after)))
    if update_delta <= 1e-6:
        raise AssertionError(f"E2E TTT did not change logits after a completed chunk: max delta={update_delta}")

    future_ids = input_ids.at[2 * chunk_size :].set(jnp.flip(input_ids[2 * chunk_size :]))
    active_with_changed_future = _next_token_e2e_logits(model, state, future_ids, after_update)
    np.testing.assert_allclose(np.asarray(active_after), np.asarray(active_with_changed_future), rtol=0, atol=0)

    selected = _next_token_e2e(model, state, input_ids, after_update)
    expected = jnp.argmax(active_after).astype(jnp.int32)
    if int(selected) != int(expected):
        raise AssertionError("RULER generation did not select the active E2E logits")

    print(
        "E2E RULER validation passed: "
        f"pre-update max delta=0, post-update max delta={update_delta:.6g}, "
        "future-token invariance passed, generation dispatch uses active logits.",
        flush=True,
    )


def _generate_greedy(
    model: MetaModel,
    state: eqx.nn.State,
    tokenizer,
    prompt: str,
    *,
    max_seq_length: int,
    tokens_to_generate: int,
    eos_token_id: int | None,
    e2e_ttt_off: bool,
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
        if model.config.training.train_mode == "meta":
            next_token_fn = _next_token_e2e_no_ttt if e2e_ttt_off else _next_token_e2e
        else:
            next_token_fn = _next_token
        token = int(jax.device_get(next_token_fn(model, state, token_buffer, jnp.asarray(cur_len - 1, dtype=jnp.int32))))
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
    done = {row["input"] for row in _read_jsonl(pred_file)} if pred_file.exists() else set()
    rows = [row for row in _read_jsonl(task_file) if row["input"] not in done]
    tokens_to_generate = _ruler_task_tokens_to_generate(ruler_dir, task)
    if args.tokens_to_generate_limit is not None:
        tokens_to_generate = min(tokens_to_generate, args.tokens_to_generate_limit)

    outputs = []
    for row in tqdm(rows, desc=f"Predicting {task}"):
        prompt = row["input"] + row.get("answer_prefix", "")
        pred = _generate_greedy(
            model,
            state,
            tokenizer,
            prompt,
            max_seq_length=args.max_seq_length,
            tokens_to_generate=tokens_to_generate,
            eos_token_id=tokenizer.eos_token_id,
            e2e_ttt_off=args.e2e_ttt_off,
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
                "answer_prefix": row.get("answer_prefix", ""),
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
    parser.add_argument("--e2e-ttt-off", action="store_true")
    parser.add_argument("--validate-e2e-only", action="store_true")
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

    cfg = _compose_config(args)
    if args.validate_e2e_only:
        if cfg.training.train_mode != "meta":
            raise ValueError("--validate-e2e-only requires a meta-trained E2E experiment")
        model, state, mesh = _load_model(cfg)
        input_ids = jax.random.randint(
            jax.random.PRNGKey(0),
            (args.max_seq_length,),
            minval=0,
            maxval=cfg.model.vocab_size,
            dtype=jnp.int32,
        )
        with mesh:
            _validate_e2e_ttt(model, state, input_ids)
        return

    ruler_dir = args.ruler_dir.resolve()
    _ensure_ruler(ruler_dir)
    compat_dir = _ensure_eval_compat(args.output_root.resolve() / "_compat")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{compat_dir}:{env.get('PYTHONPATH', '')}"
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    if args.download_aux_data:
        _download_ruler_aux_data(ruler_dir, env)

    tokenizer, ruler_tokenizer_name = _load_tokenizer(args.tokenizer_name, cfg)
    args.tokenizer_name = ruler_tokenizer_name

    tasks = DEFAULT_SYNTHETIC_TASKS if args.tasks == "all" else [task.strip() for task in args.tasks.split(",") if task.strip()]
    run_name = args.checkpoint_exp_name or Path(args.experiment).name
    result_dir = args.output_root / run_name / "synthetic" / str(args.max_seq_length)
    data_dir = result_dir / "data"
    pred_dir = result_dir / "pred"
    data_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        _prepare_task(ruler_dir, task, args, env, data_dir)

    model, state, mesh = _load_model(cfg)

    with mesh:
        for task in tasks:
            _predict_task(model, state, tokenizer, ruler_dir, task, args, data_dir, pred_dir)

    _evaluate(ruler_dir, pred_dir, env)


if __name__ == "__main__":
    main()
