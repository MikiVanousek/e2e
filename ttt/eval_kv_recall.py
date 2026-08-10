import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from ttt.config import register_configs
from ttt.dataloader.lm_dataset import SyntheticKVDataset
from ttt.infra.checkpoint import Checkpointer, unify_dict_with_eqx_module
from ttt.model.data import Batch
from ttt.model.sharding import ModelSharding
from ttt.model.transformer import BlockCollectionSplit, MetaModel
from ttt.utils.filter_utils import get_filter_spec
from ttt.utils.jax_utils import (
    clone_pytree,
    eval_shape_and_sharding,
    initialize_distibuted,
    scan_remat_chunk,
    set_random_seed,
    tree_rearrange,
)

register_configs()


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    experiment: str
    resume_exp_name: str
    seq_len: int = 8192
    attention: str = "swa"
    exp_folder: str = "kv14m-wide-kv-pretrain"
    resume_step: int = 9999
    resume_checkpoint_dir: str = ""
    extra_overrides: tuple[str, ...] = ()


CHECKPOINTS = {
    "fa": CheckpointSpec(
        label="FA",
        experiment="kv14m_wide/kv_pretrain/pretrain-8K-kv14m-wide-fa-kv-10Ksteps",
        resume_exp_name="pretrain-8K-kv14m-wide-fa-kv-10Ksteps",
        attention="fa",
    ),
    "swa": CheckpointSpec(
        label="Stateful SWA-1K",
        experiment="kv14m_wide/kv_pretrain/pretrain-8K-kv14m-wide-swa1k-kv-10Ksteps",
        resume_exp_name="pretrain-8K-kv14m-wide-stateful-swa1k-kv-10Ksteps",
        attention="swa",
    ),
    "e2e": CheckpointSpec(
        label="TTT-E2E SWA-1K",
        experiment="kv14m_wide/kv_pretrain/pretrain-8K-kv14m-wide-e2e-swa1k-ilr1e-4-kv-10Ksteps",
        resume_exp_name="pretrain-8K-kv14m-wide-e2e-swa1k-ilr1e-4-kv-10Ksteps",
        attention="e2e",
    ),
    "ipttt": CheckpointSpec(
        label="IP-TTT conv mb2048",
        experiment="kv14m_wide/kv_pretrain/pretrain-8K-kv14m-wide-ip-ttt-swa1k-kv-10Ksteps",
        resume_exp_name="pretrain-8K-kv14m-wide-ip-ttt-conv-embedding-mb2048-swa1k-kv-10Ksteps",
        attention="ip-ttt",
        extra_overrides=("model.mini_batch_size=2048",),
    ),
    "ipttt_i2048": CheckpointSpec(
        label="IP-TTT conv mb2048 i2048",
        experiment="kv14m_wide/kv_pretrain/pretrain-8K-kv14m-wide-ip-ttt-swa1k-kv-10Ksteps",
        resume_exp_name="pretrain-8K-kv14m-wide-ip-ttt-conv-embedding-mb2048-intermediate2048-swa1k-kv-10Ksteps",
        attention="ip-ttt",
        extra_overrides=("model.mini_batch_size=2048", "model.intermediate_size=2048"),
    ),
    "ipttt_all": CheckpointSpec(
        label="IP-TTT all layers",
        experiment="kv14m_wide/kv_pretrain/pretrain-8K-kv14m-wide-ip-ttt-swa1k-kv-10Ksteps",
        resume_exp_name="pretrain-8K-kv14m-wide-ip-ttt-all-layers-conv-embedding-mb2048-swa1k-kv-10Ksteps",
        attention="ip-ttt",
        extra_overrides=("model.mini_batch_size=2048", "model.ip_ttt_layers=[0,1,2,3]"),
    ),
    "e2e_all": CheckpointSpec(
        label="TTT-E2E all layers",
        experiment="kv14m_wide/kv_pretrain/pretrain-8K-kv14m-wide-e2e-all-layers-fastweights-matched-swa1k-kv-10Ksteps",
        resume_exp_name="pretrain-8K-kv14m-wide-e2e-all-layers-fastweights-matched-swa1k-kv-10Ksteps",
        attention="e2e",
    ),
}

BUCKET_LABELS = {
    "not_previously_present": "First occurrence",
    "present_outside_window": "Outside 1K",
    "present_within_window": "Inside 1K",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate top-1 synthetic KV recall accuracy from trained checkpoints.")
    parser.add_argument("--labels", default="fa,swa", help=f"Comma-separated checkpoint labels: {','.join(CHECKPOINTS)}.")
    parser.add_argument("--checkpoint-root", default="/checkpoints")
    parser.add_argument("--output-dir", default="/tmp/kv_recall")
    parser.add_argument("--max-docs", type=int, default=4, help="Number of eval documents per checkpoint. Use 0 for full eval split.")
    parser.add_argument("--logit-chunk-size", type=int, default=32)
    parser.add_argument("--sliding-window-size", type=int, default=1024)
    parser.add_argument("--eval-split", choices=("train", "eval", "all"), default="eval")
    parser.add_argument("--jax-cache-dir", default="/jax_cache")
    parser.add_argument("--validate-e2e", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def selected_specs(labels: str) -> list[CheckpointSpec]:
    specs = []
    for label in labels.split(","):
        label = label.strip()
        if not label:
            continue
        if label not in CHECKPOINTS:
            raise ValueError(f"Unknown checkpoint label {label!r}; available: {sorted(CHECKPOINTS)}")
        specs.append(CHECKPOINTS[label])
    if not specs:
        raise ValueError("No checkpoint labels selected.")
    return specs


def compose_config(spec: CheckpointSpec, args: argparse.Namespace):
    overrides = [
        "+deploy=interactive",
        f"+experiment={spec.experiment}",
        f"training.checkpoint_path={args.checkpoint_root}",
        "training.log_wandb=false",
        "training.load_part=params",
        f"training.resume_step={spec.resume_step}",
        "backend.backend=gpu",
        "backend.distributed=false",
        "backend.num_devices=1",
        f"backend.compilation_cache_dir={args.jax_cache_dir}",
        "training.n_data_parallel=1",
        "training.n_state_parallel=1",
        "training.eval_batch_size=1",
        "training.shuffle_train=false",
        "model.force_flash=false",
    ]
    overrides.extend(spec.extra_overrides)
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    with open_dict(cfg):
        cfg.training.resume_exp_name = spec.resume_exp_name
        cfg.training.seq_length = spec.seq_len
        cfg.model.seq_len = spec.seq_len
        cfg.backend.backend = "gpu"
        cfg.backend.distributed = False
        cfg.backend.num_devices = 1
        cfg.model.force_flash = False
        if spec.resume_checkpoint_dir:
            cfg.checkpoint.resume_checkpoint_dir = spec.resume_checkpoint_dir
        else:
            cfg.checkpoint.resume_checkpoint_dir = str(Path(args.checkpoint_root) / spec.exp_folder / spec.resume_exp_name)

    OmegaConf.resolve(cfg)
    return cfg


def load_checkpoint_model(spec: CheckpointSpec, args: argparse.Namespace):
    cfg = compose_config(spec, args)
    initialize_distibuted(cfg.backend)
    key = set_random_seed(cfg.training.model_seed)
    model_sharding = ModelSharding(cfg)
    mesh = model_sharding.mesh

    @eqx.filter_jit
    def create_sharded_model_and_state():
        model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
        state = jax.device_put(state, jax.NamedSharding(mesh, jax.sharding.PartitionSpec()))
        model = model_sharding.shard_params(model)
        return model, state

    with mesh:
        model, state = create_sharded_model_and_state()
        abstract_model_weights = eval_shape_and_sharding(lambda: create_sharded_model_and_state()[0].weights())
        checkpointer = Checkpointer(config=cfg, for_saving=False)
        out_state = checkpointer.load_checkpoint(
            step=cfg.training.resume_step,
            targets={"model_weights": abstract_model_weights},
            restore=cfg.training.load_part,
        )
        model = unify_dict_with_eqx_module(out_state["model_weights"], model)[0]
        checkpointer.close()
        state = state.set(model.step_index, jnp.array(jnp.iinfo(jnp.int32).max - 100, dtype=jnp.int32))

    return cfg, model, state, mesh


@eqx.filter_jit
def hidden_states_for_tokens(model: MetaModel, state, input_ids, target_tokens, loss_masks):
    seq = Batch(input_ids=input_ids, target_tokens=target_tokens, loss_masks=loss_masks)
    outputs = model.language_model.model(state, seq)
    return outputs.last_hidden_state, outputs.state


@eqx.filter_jit
def e2e_hidden_states_for_tokens(model: MetaModel, state, input_ids, target_tokens, loss_masks):
    cfg = model.config
    seq = Batch(input_ids=input_ids, target_tokens=target_tokens, loss_masks=loss_masks)
    block_collection = model.language_model.model.h.blocks
    new_collection = BlockCollectionSplit(
        cfg.model,
        block_collection=block_collection,
        prime_storage=model.language_model.model.h.prime_storage,
        key=jax.random.PRNGKey(0),
    )
    state_prefix_suffix = state.substate(block_collection)
    state_prefix, state_suffix = BlockCollectionSplit.split_state(state_prefix_suffix, cfg.model.suffix_len)
    state_all = clone_pytree(state)
    model = eqx.tree_at(lambda m: m.language_model.model.h, model, new_collection)
    model = jax.tree.map(lambda p: p.astype(model.state_dtype), model)
    inner_opt_state = model.inner_optimizer(state_all).init(model.inner_parameters())

    input_embeds = model.language_model.wte_call(seq.input_ids)
    prefix_output = model.language_model.prefix_call(model.language_model.model.h.prefix_blocks, input_embeds, state_prefix, seq).last_hidden_state
    chunk_size = cfg.model.mini_batch_size
    seq = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=chunk_size)
    prefix_output = tree_rearrange(prefix_output, "(chunk token) ... -> chunk token ...", token=chunk_size)

    def process_chunk(model__opt_state__state, inputs):
        model_inner, opt_state, state_tuple = model__opt_state__state
        seq_chunk, prefix_chunk = inputs
        spec_inner = get_filter_spec(model_inner, cfg.training.spec_inner, "inner parameters")
        inner_params, _ = eqx.partition(model_inner, spec_inner)
        _, outer_params = eqx.partition(model, spec_inner)
        model_inner = eqx.combine(inner_params, outer_params)

        outputs = model_inner.language_model.suffix_call(prefix_chunk, state_tuple[1], seq_chunk)
        new_model, opt_state, state_tuple, _ = MetaModel.inner_loop_step(
            model_inner,
            opt_state,
            state_tuple,
            seq_chunk,
            prefix_chunk,
        )
        return (new_model, opt_state, state_tuple), outputs.last_hidden_states

    _, hidden_chunks = scan_remat_chunk(
        process_chunk,
        (model, inner_opt_state, (state_all, state_suffix)),
        (seq, prefix_output),
        remat_n_loops=cfg.training.inner_remat_freq,
        unroll=cfg.model.unroll_inner_scan,
    )
    return tree_rearrange(hidden_chunks, "chunk token ... -> (chunk token) ...")


@eqx.filter_jit
def predict_value_tokens(model: MetaModel, hidden_chunk):
    logits = model.language_model.wte_disembed_call(hidden_chunk)
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


def value_target_table(tokens: np.ndarray, loss_masks: np.ndarray, *, sliding_window_size: int) -> pd.DataFrame:
    rows = []
    last_query_pos: dict[int, int] = {}

    for pos in np.flatnonzero(loss_masks):
        query_token = int(tokens[pos])
        value_token = int(tokens[pos + 1])
        previous_query_pos = last_query_pos.get(query_token)

        if previous_query_pos is None:
            previous_value_distance = -1
            bucket = "not_previously_present"
        else:
            previous_value_distance = int(pos - (previous_query_pos + 1))
            bucket = "present_within_window" if previous_value_distance < sliding_window_size else "present_outside_window"

        rows.append(
            {
                "position": int(pos),
                "query_token": query_token,
                "value_token": value_token,
                "previous_value_distance": previous_value_distance,
                "bucket": bucket,
            }
        )
        last_query_pos[query_token] = int(pos)

    return pd.DataFrame(rows)


def evaluate_document(
    model: MetaModel,
    state,
    mesh,
    tokens: np.ndarray,
    loss_masks: np.ndarray,
    args: argparse.Namespace,
    *,
    validate_e2e: bool = False,
) -> pd.DataFrame:
    target_df = value_target_table(tokens, loss_masks, sliding_window_size=args.sliding_window_size)
    input_ids = jnp.asarray(tokens[:-1], dtype=jnp.int32)
    target_tokens = jnp.asarray(tokens[1:], dtype=jnp.int32)
    loss_masks_jax = jnp.asarray(loss_masks, dtype=bool)

    with mesh:
        if model.config.training.train_mode == "meta":
            hidden = e2e_hidden_states_for_tokens(model, state, input_ids, target_tokens, loss_masks_jax)
        elif model.config.model.seq_modeling_block == "SWA":
            chunk_size = model.config.model.mini_batch_size
            hidden_chunks = []
            chunk_state = state
            for start in range(0, int(input_ids.shape[0]), chunk_size):
                end = start + chunk_size
                hidden_chunk, chunk_state = hidden_states_for_tokens(
                    model,
                    chunk_state,
                    input_ids[start:end],
                    target_tokens[start:end],
                    loss_masks_jax[start:end],
                )
                hidden_chunks.append(hidden_chunk)
            hidden = jnp.concatenate(hidden_chunks, axis=0)
        else:
            hidden, _ = hidden_states_for_tokens(model, state, input_ids, target_tokens, loss_masks_jax)

        predictions = []
        positions = target_df["position"].to_numpy(dtype=np.int32)
        for start in range(0, len(positions), args.logit_chunk_size):
            pos_chunk = jnp.asarray(positions[start : start + args.logit_chunk_size])
            pred_chunk = predict_value_tokens(model, hidden[pos_chunk])
            predictions.append(np.asarray(jax.device_get(pred_chunk)))

        if validate_e2e:
            from ttt.ruler import _next_token_e2e

            test_positions = np.array([0, model.config.model.mini_batch_size, len(input_ids) - 1], dtype=np.int32)
            batched = np.asarray(jax.device_get(predict_value_tokens(model, hidden[jnp.asarray(test_positions)])))
            ruler = np.array(
                [
                    int(jax.device_get(_next_token_e2e(model, state, input_ids, jnp.asarray(position, dtype=jnp.int32))))
                    for position in test_positions
                ],
                dtype=np.int32,
            )
            np.testing.assert_array_equal(batched, ruler)
            print(f"Validated batched E2E predictions against RULER at positions {test_positions.tolist()}", flush=True)

    target_df["predicted_value_token"] = np.concatenate(predictions) if predictions else np.array([], dtype=np.int32)
    target_df["correct"] = target_df["predicted_value_token"].to_numpy() == target_df["value_token"].to_numpy()
    return target_df


def make_eval_dataset(cfg, spec: CheckpointSpec, args: argparse.Namespace) -> SyntheticKVDataset:
    return SyntheticKVDataset(
        seq_len=spec.seq_len,
        vocab_size=cfg.model.vocab_size,
        bos_token_id=cfg.model.bos_token_id,
        eos_token_id=cfg.model.eos_token_id,
        num_pairs=cfg.dataset.synthetic_num_pairs,
        num_docs=cfg.dataset.synthetic_num_docs,
        seed=cfg.dataset.synthetic_seed,
        data_partition=args.eval_split,
        eval_fraction=cfg.training.eval_fraction,
    )


def wilson_ci(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return center - half, center + half


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary = rows.groupby(["checkpoint", "attention", "bucket"], as_index=False).agg(correct=("correct", "sum"), total=("correct", "size"))
    summary["accuracy"] = summary["correct"] / summary["total"]
    intervals = [wilson_ci(int(row.correct), int(row.total)) for row in summary.itertuples()]
    summary["ci95_low"] = [low for low, _ in intervals]
    summary["ci95_high"] = [high for _, high in intervals]
    summary["bucket_label"] = summary["bucket"].map(BUCKET_LABELS)
    return summary.sort_values(["checkpoint", "bucket"])


def evaluate_checkpoint(spec: CheckpointSpec, args: argparse.Namespace) -> pd.DataFrame:
    print(f"Loading {spec.label} from {spec.resume_exp_name}:{spec.resume_step}", flush=True)
    cfg, model, state, mesh = load_checkpoint_model(spec, args)
    dataset = make_eval_dataset(cfg, spec, args)
    n_docs = len(dataset) if args.max_docs == 0 else min(args.max_docs, len(dataset))
    doc_results = []

    for doc_idx in tqdm(range(n_docs), desc=f"eval {spec.label}"):
        tokens, loss_masks = dataset[doc_idx]
        doc_df = evaluate_document(
            model,
            state,
            mesh,
            np.asarray(tokens),
            np.asarray(loss_masks, dtype=bool),
            args,
            validate_e2e=args.validate_e2e and spec.attention == "e2e" and doc_idx == 0,
        )
        doc_df.insert(0, "doc_idx", doc_idx)
        doc_results.append(doc_df)

    result = pd.concat(doc_results, ignore_index=True)
    result.insert(0, "resume_step", spec.resume_step)
    result.insert(0, "checkpoint", spec.label)
    result.insert(0, "attention", spec.attention)
    result.insert(0, "seq_len", spec.seq_len)

    del model, state
    jax.clear_caches()
    return result


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    specs = selected_specs(args.labels)
    raw = pd.concat([evaluate_checkpoint(spec, args) for spec in specs], ignore_index=True)
    summary = summarize(raw)

    raw_path = Path(args.output_dir) / "kv_recall_raw.csv"
    summary_path = Path(args.output_dir) / "kv_recall_summary.csv"
    metadata_path = Path(args.output_dir) / "kv_recall_metadata.json"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "checkpoints": [asdict(spec) for spec in specs],
                "num_rows": int(len(raw)),
            },
            indent=2,
        )
    )

    print(summary[["checkpoint", "bucket_label", "correct", "total", "accuracy", "ci95_low", "ci95_high"]].to_string(index=False), flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {raw_path}", flush=True)


if __name__ == "__main__":
    main()
