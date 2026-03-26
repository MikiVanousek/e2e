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
        chunks_dir: str | None = None,
    ):
        self.train_holdout_loader = (
            lm_dataset(
                path=config.training.dataset_path,
                seq_len=config.training.seq_length,
                split=config.training.eval_split,
                global_batch_size=global_batch_size,
                seed=0,
                repeat=False,
                shuffle=False,
                bos_token_id=config.model.bos_token_id,
                eos_token_id=config.model.eos_token_id,
                vocab_size=config.model.vocab_size,
                tokenizer_name=config.training.tokenizer_name,
                chunks_dir=chunks_dir,
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

    def eval_fn(self, model: MetaModel, state: eqx.nn.State, step: int):
        pid = jax.process_index()  # 0 -- (n_host - 1)
        max_batches = self.config.training.max_eval_batches

        loader_dict = {"train_holdout": self.train_holdout_loader}

        def load_to_sharded_array(arr):
            return jax.make_array_from_process_local_data(sharding=self.data_sharding, local_data=arr, global_shape=(self.global_batch_size, *arr.shape[1:]))

        eval_metrics = {}
        eval_loss_ci = {}

        for name, ds in loader_dict.items():
            batch_loader = ds.to_iter_dataset(
                grain.ReadOptions(num_threads=self.config.training.loader_workers, prefetch_buffer_size=500),
            ).map(lambda batch: jax.tree.map(load_to_sharded_array, batch))
            total = min(len(ds), max_batches) if max_batches > 0 else len(ds)

            results = []
            for batch in tqdm(batch_loader, desc=f"Evaluating on sequence {name}", total=total, disable=pid != 0):
                results.append(eval_step_fn(model, batch, state))
                if max_batches > 0 and len(results) >= max_batches:
                    break

            eval_metrics[name] = jax.tree.map(lambda *x: np.asarray(jnp.mean(jnp.stack(x), axis=0, dtype=jnp.float32)), *results)

            loss_per_batch = np.array([np.asarray(r[M.loss]).mean() for r in results])
            n = len(loss_per_batch)
            eval_loss_ci[name] = 1.96 * loss_per_batch.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0

        self.log_eval_results(eval_metrics, eval_loss_ci, step)

    def log_eval_results(self, eval_metrics, eval_loss_ci, step):
        for eval_name, v in eval_metrics.items():
            ci = eval_loss_ci.get(eval_name, 0.0)
            for metric_name, metric in v.items():
                if metric_name == M.loss:
                    mean_loss = float(metric.mean())
                    master_log(logger, f"Eval -- {eval_name}/{metric_name}: {mean_loss:.4f} ± {ci:.4f}")
                    self.wandb_logger.log({
                        f"{eval_name}/{metric_name}": mean_loss,
                        f"{eval_name}/{metric_name}_ci_lower": mean_loss - ci,
                        f"{eval_name}/{metric_name}_ci_upper": mean_loss + ci,
                    }, step)

                else:
                    save_dir = self.log_dir / f"{eval_name}_{metric_name}.npy"
                    if metric_name == M.token_nll_loss:
                        self.wandb_logger.log_token_nll_loss(metric, step, eval_name)
                    np.save(save_dir, metric)
                    self.wandb_logger.save(save_dir, self.log_dir)


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
