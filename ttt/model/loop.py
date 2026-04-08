import logging
from pathlib import Path

import equinox as eqx
import grain.python as grain
import jax
import jax.ad_checkpoint
import jax.numpy as jnp
import numpy as np
from equinox import nn
from optax import OptState
from tqdm import tqdm

from ttt.config import Config
from ttt.dataloader.lm_dataset import dummy_dataset, lm_dataset
from ttt.infra.wandb_utils import WandbLogger
from ttt.model.data import Batch
from ttt.model.transformer import MetaModel
from ttt.optimizers import make_optimizer
from ttt.utils.filter_utils import filter_apply_updates
from ttt.utils.jax_utils import global_norm_safe, master_log, tree_rearrange, vmap_mean, welfords_online_mean

logger = logging.getLogger(__name__)

M = MetaModel.MetricType


@eqx.filter_jit
@eqx.filter_vmap(axis_name="data_parallel", in_axes=(None, 0, None), out_axes=None)
def eval_step_fn(
    meta_model: MetaModel,
    seq: Batch,
    state: eqx.nn.State,
):
    """
    Get loss and token-wise NLL metrics for a single batch of data, reduced across all devices.

    Args:
        meta_model: The model to evaluate.
        batch: The batch to evaluate on.
    """

    loss, metrics = meta_model.loss_for_sequence(seq, state)  # Reduce over per-device batch

    _avg_loss, avg_metrics = jax.lax.pmean((loss, metrics), axis_name="data_parallel")  # Reduce over devices

    return avg_metrics


@eqx.filter_jit
@eqx.filter_vmap(axis_name="data_parallel", in_axes=(None, 0, None), out_axes=None)
def eval_step_fn_no_ttt(
    meta_model: MetaModel,
    seq: Batch,
    state: eqx.nn.State,
):
    """Evaluate without TTT inner loop (pretrain-mode forward pass)."""
    loss, metrics = meta_model.loss_for_sequence(seq, state, train_mode="pretrain")
    _avg_loss, avg_metrics = jax.lax.pmean((loss, metrics), axis_name="data_parallel")
    return avg_metrics


class Evaluator:
    """
    Contains data loading and evaluation logic + state.
    """

    def __init__(
        self,
        global_batch_size: int,
        data_sharding: jax.sharding.NamedSharding,
        config: Config,
        wandb_logger: WandbLogger,
        log_dir: Path,
    ):
        self.train_holdout_loader = (
            lm_dataset(
                hf_dataset=config.dataset.hf_dataset,
                hf_subset=config.dataset.hf_subset,
                hf_text_column=config.dataset.hf_text_column,
                seq_len=config.training.seq_length,
                split=config.training.eval_split,
                global_batch_size=global_batch_size,
                seed=0,
                repeat=False,
                shuffle=False,
                bos_token_id=config.model.bos_token_id,
                eos_token_id=config.model.eos_token_id,
                tokenizer_name=config.training.tokenizer_name,
                vocab_size=config.model.vocab_size,
                cache_dir=config.dataset.hf_cache_dir,
            )
            if not config.training.dummy_dataset
            else dummy_dataset(
                seq_len=config.training.seq_length,
                global_batch_size=global_batch_size,
                bos_token_id=config.model.bos_token_id,
                eos_token_id=config.model.eos_token_id,
                num_tokens=2**25,
            )
        )
        self.data_sharding = data_sharding
        self.config = config
        self.global_batch_size = global_batch_size
        self.wandb_logger = wandb_logger
        self.log_dir = log_dir
        self.bytes_per_token = self._compute_bytes_per_token()

    def _compute_bytes_per_token(self) -> float:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.training.tokenizer_name)
        total_bytes = 0
        total_tokens = 0
        for i in range(min(2, len(self.train_holdout_loader))):
            batch = self.train_holdout_loader[i]
            for row in np.asarray(batch.input_ids):
                text = tokenizer.decode(row.tolist())
                total_bytes += len(text.encode('utf-8'))
                total_tokens += len(row)
        bpt = total_bytes / total_tokens
        master_log(logger, f"Computed bytes_per_token={bpt:.2f} (from {total_tokens:,} tokens)")
        return bpt

    def eval_fn(self, model: MetaModel, state: eqx.nn.State, step: int, max_batches: int | None = None):
        pid = jax.process_index()
        is_meta = self.config.training.train_mode == "meta"

        loader_dict = {"train_holdout": self.train_holdout_loader}

        def load_to_sharded_array(arr):
            return jax.make_array_from_process_local_data(sharding=self.data_sharding, local_data=arr, global_shape=(self.global_batch_size, *arr.shape[1:]))

        eval_metrics = {}
        eval_stds = {}
        eval_counts = {}

        def _aggregate(results):
            n = len(results)
            means = jax.tree.map(
                lambda *x: np.asarray(jnp.mean(jnp.stack(x), axis=0, dtype=jnp.float32)), *results
            )
            if n > 1:
                stds = jax.tree.map(
                    lambda *x: np.std(np.stack([np.asarray(xi, dtype=np.float32) for xi in x]), axis=0, ddof=1),
                    *results,
                )
            else:
                stds = jax.tree.map(np.zeros_like, means)
            return means, stds, n

        for eval_name, ds in loader_dict.items():
            n_batches = min(max_batches, len(ds)) if max_batches is not None else len(ds)
            batch_loader = ds.to_iter_dataset().map(lambda batch: jax.tree.map(load_to_sharded_array, batch))

            results = []
            results_no_ttt = []
            for i, batch in enumerate(tqdm(batch_loader, desc=f"Evaluating {eval_name}", total=n_batches, disable=pid != 0)):
                if i >= n_batches:
                    break
                result = eval_step_fn(model, batch, state)
                results.append(result)

                if is_meta:
                    result_no_ttt = eval_step_fn_no_ttt(model, batch, state)
                    results_no_ttt.append(result_no_ttt)

            eval_metrics[eval_name], eval_stds[eval_name], eval_counts[eval_name] = _aggregate(results)

            if is_meta and results_no_ttt:
                k = f"{eval_name}_no_ttt"
                eval_metrics[k], eval_stds[k], eval_counts[k] = _aggregate(results_no_ttt)

        self.log_eval_results(eval_metrics, eval_stds, eval_counts, step)

    @staticmethod
    def _ci95_width_pct(mean_val: float, std_val: float, n: int) -> float:
        """95% CI full width as a percentage of |mean|."""
        if n <= 1 or mean_val == 0:
            return 0.0
        return float(2 * 1.96 * std_val / (np.sqrt(n) * abs(mean_val)) * 100)

    def log_eval_results(self, eval_metrics, eval_stds, eval_counts, step):
        ln2 = np.log(2)

        for eval_name, v in eval_metrics.items():
            stds = eval_stds.get(eval_name, {})
            n = eval_counts.get(eval_name, 1)

            for metric_name, metric in v.items():
                if metric_name == M.loss:
                    loss_val = float(metric.mean())
                    bpb = loss_val / (self.bytes_per_token * ln2)
                    master_log(logger, f"Eval -- {eval_name}/{metric_name}: {loss_val:.4f} (BPB: {bpb:.4f})")

                    log_data = {f"{eval_name}/{metric_name}": loss_val, f"{eval_name}/bpb": bpb}
                    if eval_name == "train_holdout":
                        log_data["val/loss"] = loss_val
                        log_data["val/bpb"] = bpb
                    self.wandb_logger.log(log_data, step)

                    if M.loss in stds:
                        loss_std = float(stds[M.loss].mean())
                        bpb_std = loss_std / (self.bytes_per_token * ln2)
                        ci_data = {
                            f"{eval_name}_ci95_pct/loss": self._ci95_width_pct(loss_val, loss_std, n),
                            f"{eval_name}_ci95_pct/bpb": self._ci95_width_pct(bpb, bpb_std, n),
                        }
                        if eval_name == "train_holdout":
                            ci_data["val_ci95_pct/loss"] = ci_data[f"{eval_name}_ci95_pct/loss"]
                            ci_data["val_ci95_pct/bpb"] = ci_data[f"{eval_name}_ci95_pct/bpb"]
                        self.wandb_logger.log(ci_data, step)

                else:
                    save_dir = self.log_dir / f"{eval_name}_{metric_name}.npy"
                    if metric_name == M.token_nll_loss:
                        self.wandb_logger.log_token_nll_loss(metric, step, eval_name)
                    np.save(save_dir, metric)
                    self.wandb_logger.save(save_dir, self.log_dir)

        for eval_name in list(eval_metrics.keys()):
            no_ttt_name = f"{eval_name}_no_ttt"
            if no_ttt_name in eval_metrics and M.loss in eval_metrics[eval_name] and M.loss in eval_metrics[no_ttt_name]:
                ttt_loss = float(eval_metrics[eval_name][M.loss].mean())
                no_ttt_loss = float(eval_metrics[no_ttt_name][M.loss].mean())
                ttt_gain = no_ttt_loss - ttt_loss
                ttt_gain_bpb = ttt_gain / (self.bytes_per_token * ln2)
                master_log(logger, f"Eval -- {eval_name}/ttt_gain: {ttt_gain:.4f} (BPB: {ttt_gain_bpb:.4f})")
                self.wandb_logger.log({
                    f"{eval_name}/ttt_gain": ttt_gain,
                    f"{eval_name}/ttt_gain_bpb": ttt_gain_bpb,
                }, step)

                n_with = eval_counts.get(eval_name, 1)
                n_without = eval_counts.get(no_ttt_name, 1)
                if n_with > 1 and n_without > 1:
                    with_std = float(eval_stds[eval_name].get(M.loss, np.array(0.0)).mean())
                    without_std = float(eval_stds[no_ttt_name].get(M.loss, np.array(0.0)).mean())
                    se_gain = np.sqrt(with_std**2 / n_with + without_std**2 / n_without)
                    ci_width = 2 * 1.96 * se_gain
                    gain_ci = float(ci_width / abs(ttt_gain) * 100) if abs(ttt_gain) > 0 else 0.0
                    gain_bpb_ci = float(ci_width / (self.bytes_per_token * ln2) / abs(ttt_gain_bpb) * 100) if abs(ttt_gain_bpb) > 0 else 0.0
                    self.wandb_logger.log({
                        f"{eval_name}_ci95_pct/ttt_gain": gain_ci,
                        f"{eval_name}_ci95_pct/ttt_gain_bpb": gain_bpb_ci,
                    }, step)


@eqx.filter_jit(donate="all-except-first")
@eqx.filter_vmap(in_axes=(None, None, None, 0, None), out_axes=None, axis_name="data_parallel")  # Functions as pmap at call site
def train_on_sequence(
    state: nn.State, meta_model: MetaModel, opt_state: OptState, batch: Batch, cfg: Config
) -> tuple[MetaModel, OptState, jnp.ndarray, dict[MetaModel.MetricType, jnp.ndarray]]:
    """
    Train the model for a single step on a sequence.

    Args:
        outer_optimizer_mut: The optimizer to update with. The model weights and optimizer state are updated in-place.
        batch: The batch to train on. Should be of shape [per_device_batch_size * accum_steps, T]
        cfg: Full configuration.

    Returns:
        loss: Current step computed loss.
        metrics: Dict of metrics (MetricType: aggregated token-wise negative log likelihood loss and gradient norms)
    """
    M = MetaModel.MetricType
    seqlen = cfg.training.seq_length
    mini_batch_size = cfg.model.mini_batch_size
    assert seqlen % mini_batch_size == 0, "Right now only supports seqlen as a multiple of mini batch size"
    assert batch.shape[0] % cfg.training.accum_steps == 0, (
        f"Gradient accumulation steps should divide the per-device batch size, got {batch.shape[0]=} !% {cfg.training.accum_steps=}"
    )

    batch = tree_rearrange(batch, "(accum batch) ... -> accum batch ...", accum=cfg.training.accum_steps)

    # MetaModel.loss_for_sequence computes the loss for a single sequence -- need to vmap and then mean to compute the loss for a batch
    loss_fn = lambda model, b: vmap_mean(lambda seq: MetaModel.loss_for_sequence(model, seq, state), b, axis_name="batch")  # Outputs a single loss value

    meta_grad_fn = lambda batch: eqx.filter_value_and_grad(loss_fn, has_aux=True)(meta_model, batch)

    meta_grad_fn = lambda batch, fun=meta_grad_fn: welfords_online_mean(fun, batch)  # Accumulate statistics and gradients
    (loss, metrics), grads_meta = meta_grad_fn(batch)

    # Aggregate across data parallel dim
    avg_loss, avg_metrics, avg_grads_meta = jax.lax.pmean((loss, metrics, grads_meta), axis_name="data_parallel")

    avg_grads_meta = avg_grads_meta.trainable_parameters()  # Handle frozen parameter spec
    avg_outer_gnorm = global_norm_safe(avg_grads_meta)
    avg_metrics[M.outer_grad_norm] = avg_outer_gnorm

    outer_tx, _ = make_optimizer(cfg.training.optimizer_outer)
    updates, opt_state = outer_tx.update(avg_grads_meta, opt_state, meta_model.trainable_parameters())

    meta_model = filter_apply_updates(meta_model, updates)

    # Do not return new state
    return (meta_model, opt_state, avg_loss, avg_metrics)
