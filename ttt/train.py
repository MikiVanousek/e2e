import logging
import math
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pformat

import equinox as eqx
import grain.python as grain
import hydra
import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from omegaconf import OmegaConf
from optax import OptState
from tqdm import tqdm as _tqdm

from ttt.config import Config, register_configs
from ttt.dataloader.lm_dataset import CHUNK_SIZE, chunks_volume_key, dummy_dataset, lm_dataset, save_chunk
from ttt.infra.checkpoint import Checkpointer, unify_dict_with_eqx_module
from ttt.infra.wandb_utils import WandbLogger
from ttt.model.loop import Evaluator, train_on_sequence
from ttt.model.sharding import ModelSharding
from ttt.model.transformer import MetaModel
from ttt.optimizers import make_optimizer
from ttt.utils.jax_utils import eval_shape_and_sharding, get_custom_tqdm, initialize_distibuted, master_log, set_random_seed, tree_rearrange
from ttt.utils.memory_utils import log_memory_breakdown, log_memory_gauge

register_configs()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class _BatchPrefetcher:
    """Prefetches the next batch in a background thread so data loading overlaps with GPU compute."""

    def __init__(self, data_iter, transform_fn):
        self._iter = data_iter
        self._transform = transform_fn
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = self._executor.submit(self._load_next)

    def _load_next(self):
        return self._transform(next(self._iter))

    def next(self):
        batch = self._future.result()
        self._future = self._executor.submit(self._load_next)
        return batch

    @property
    def underlying_iter(self):
        return self._iter

    def close(self):
        self._future.cancel()
        self._executor.shutdown(wait=False)


def _prepare_data_parallelism(cfg: Config, global_dev_num: int) -> int:
    if cfg.training.n_data_parallel is None:
        assert global_dev_num % cfg.training.n_state_parallel == 0, "Number of devices must be divisible by state parallelism"
        cfg.training.n_data_parallel = global_dev_num // cfg.training.n_state_parallel
    assert cfg.training.n_data_parallel * cfg.training.n_state_parallel == global_dev_num, (
        f"Data parallelism ({cfg.training.n_data_parallel}) and state parallelism ({cfg.training.n_state_parallel}) must match the number of devices ({global_dev_num})"
    )
    return cfg.training.n_data_parallel


def _resolve_chunks_dir(cfg: Config, split: str) -> str | None:
    """If chunks_dir is configured and chunk files exist, copy them to /tmp and return the local path."""
    if not cfg.training.chunks_dir:
        return None
    key = chunks_volume_key(cfg.model.vocab_size, cfg.training.dataset_name, cfg.training.seq_length, split)
    vol_dir = Path(cfg.training.chunks_dir) / key
    if not vol_dir.exists() or not any(vol_dir.glob("chunk_*.npy")):
        return None
    local_dir = Path("/tmp/chunks") / key
    if not local_dir.exists():
        local_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(vol_dir.glob("chunk_*.npy")):
            print(f"Copying {f} -> {local_dir / f.name}")
            shutil.copy2(f, local_dir / f.name)
    return str(local_dir)


def _make_train_iterator(cfg: Config, model_cfg, data_sharding: jax.sharding.Sharding, n_data_parallel: int, chunks_dir: str | None = None):
    train_ds = (
        lm_dataset(
            path=cfg.training.dataset_path,
            split=cfg.training.data_split,
            seq_len=cfg.training.seq_length,
            seed=cfg.training.data_seed,
            global_batch_size=cfg.training.global_batch_size,
            repeat=True,
            bos_token_id=model_cfg.bos_token_id,
            eos_token_id=model_cfg.eos_token_id,
            vocab_size=model_cfg.vocab_size,
            tokenizer_name=cfg.training.tokenizer_name,
            chunks_dir=chunks_dir,
        )
        if not cfg.training.dummy_dataset
        else dummy_dataset(
            seq_len=cfg.training.seq_length,
            global_batch_size=cfg.training.global_batch_size,
            bos_token_id=model_cfg.bos_token_id,
            eos_token_id=model_cfg.eos_token_id,
            repeat=True,
        )
    )

    def load_to_sharded_array(arr):
        return jax.make_array_from_process_local_data(sharding=data_sharding, local_data=arr, global_shape=(cfg.training.global_batch_size, *arr.shape[1:]))

    def to_sharded_batch(batch):
        batch = jax.tree.map(lambda x: load_to_sharded_array(x), batch)
        return tree_rearrange(batch, "(data_parallel batch) ... -> data_parallel batch ...", data_parallel=n_data_parallel)

    train_iter_ds = train_ds.to_iter_dataset(
        grain.ReadOptions(num_threads=cfg.training.loader_workers, prefetch_buffer_size=500),
    )
    return iter(train_iter_ds), to_sharded_batch


def _main(cfg: Config) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    logger.info("\n".join([f"{k}={v}" for k, v in os.environ.items()]))
    logger.info(f"Launching with \n {pformat(cfg_dict)}.")

    model_cfg = cfg.model
    backend_cfg = cfg.backend

    initialize_distibuted(backend_cfg)

    key = set_random_seed(cfg.training.model_seed)

    n_host = jax.process_count()

    global_dev_num = jax.device_count()
    local_dev_num = jax.local_device_count()
    master_process = jax.process_index() == 0

    n_data_parallel = _prepare_data_parallelism(cfg, global_dev_num)

    log_dir = Path(cfg.training.exp_dir) / cfg.training.exp_folder / cfg.training.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    with WandbLogger(
        entity=cfg.training.wandb_entity,
        project=cfg.training.wandb_project,
        run_name=cfg.training.run_name,
        log_dir=log_dir,
        wandb_key=cfg.training.wandb_key,
        logging_process=0,
        config=cfg_dict,
        enabled=cfg.training.log_wandb,
    ) as wandb_logger:
        dev_info = f"Process # {n_host}\tLocal dev # {local_dev_num}\tTotal dev # {global_dev_num}"
        master_log(logger, dev_info)

        checkpointer = Checkpointer(config=cfg, for_saving=True)

        optimizer_outer_loop, optimizer_info_outer_loop = make_optimizer(cfg.training.optimizer_outer)

        model_sharding = ModelSharding(cfg)
        mesh = model_sharding.mesh
        data_sharding = jax.NamedSharding(mesh, P("data"))
        cfg.model.seq_len = cfg.training.seq_length

        resolved_train_chunks_dir = _resolve_chunks_dir(cfg, cfg.training.data_split)
        if resolved_train_chunks_dir:
            master_log(logger, f"Using chunked train data from {resolved_train_chunks_dir}")
        resolved_eval_chunks_dir = _resolve_chunks_dir(cfg, cfg.training.eval_split)
        if resolved_eval_chunks_dir:
            master_log(logger, f"Using chunked eval data from {resolved_eval_chunks_dir}")

        train_ds_iter, to_sharded_batch = _make_train_iterator(cfg, model_cfg, data_sharding, n_data_parallel, chunks_dir=resolved_train_chunks_dir)

        @eqx.filter_jit
        def create_sharded_model_and_state() -> tuple[MetaModel, eqx.nn.State]:
            model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
            state = jax.device_put(state, jax.NamedSharding(mesh, P()))  # Replicate initial (empty) state
            model = model_sharding.shard_params(model)
            return model, state

        @eqx.filter_jit
        def create_stepped_opt_state(model: MetaModel) -> OptState:
            """
            Create optimizer state with correct sharding after having a single update step applied.
            """
            trainable_params = model.trainable_parameters()
            opt_state = optimizer_outer_loop.init(trainable_params)
            _, opt_state = optimizer_outer_loop.update(trainable_params, opt_state, model.trainable_parameters())
            # Should be sharded the same way as the model parameters
            return opt_state

        t_setup = time.perf_counter()
        if checkpointer.checkpoint_exists() or cfg.training.load_part != "none":
            if checkpointer.checkpoint_exists():
                load_part = "all"
                load_checkpointer = checkpointer
            else:
                assert cfg.checkpoint.resume_checkpoint_dir is not None
                load_part = cfg.training.load_part
                load_checkpointer = Checkpointer(
                    config=cfg, for_saving=False
                )  # Use the resumption path only if the run is starting from scratch. Otherwise use the current checkpointing path.

            if load_part == "all" and cfg.training.eval_mode:  # prevent uncessary opt and loop state resumption
                load_part = "params"

            t0 = time.perf_counter()
            abstract_model_weights = eval_shape_and_sharding(lambda: create_sharded_model_and_state()[0].weights())
            abstract_opt_state = eval_shape_and_sharding(lambda: create_stepped_opt_state(create_sharded_model_and_state()[0]))
            master_log(logger, f"[timing] eval_shape_and_sharding: {time.perf_counter() - t0:.1f}s")

            t0 = time.perf_counter()
            out_state = load_checkpointer.load_checkpoint(
                step=cfg.training.resume_step,
                targets={"model_weights": abstract_model_weights, "opt_state": abstract_opt_state, "train_ds_iter": train_ds_iter},
                restore=load_part,
            )
            master_log(logger, f"[timing] load_checkpoint: {time.perf_counter() - t0:.1f}s")

            def load_model_weights(model: MetaModel, out_state) -> MetaModel:
                model_loaded, not_found = unify_dict_with_eqx_module(out_state["model_weights"], model)
                if not_found:
                    master_log(logger, f"Parameters initialized from scratch (not in checkpoint): {not_found}")
                return model_loaded

            master_log(logger, "Restoring model weights")
            t0 = time.perf_counter()
            model, state = create_sharded_model_and_state()
            master_log(logger, f"[timing] create_sharded_model_and_state: {time.perf_counter() - t0:.1f}s")

            model = load_model_weights(model, out_state)

            if "opt_state" not in out_state:  # Create new optimizer state
                master_log(logger, "Restored model weights, creating new optimizer state")
                t0 = time.perf_counter()
                opt_state = optimizer_outer_loop.init(model.trainable_parameters())
                master_log(logger, f"[timing] optimizer_init: {time.perf_counter() - t0:.1f}s")
                start_step = 0

            else:  # Restore optimizer state

                def create_opt_state_with_loaded_weights(model: MetaModel, out_state) -> OptState:
                    opt_state = create_stepped_opt_state(model)
                    opt_state = unify_dict_with_eqx_module(out_state["opt_state"], opt_state)[0]
                    return opt_state

                master_log(logger, "Restoring optimizer state")
                t0 = time.perf_counter()
                opt_state = create_opt_state_with_loaded_weights(model, out_state)
                master_log(logger, f"[timing] restore_optimizer_state: {time.perf_counter() - t0:.1f}s")
                start_step = int(jax.device_get(out_state["train_ds_iter"].get_state()["next_index"]))

            del out_state, load_checkpointer

        else:  # Create new model and optimizer state
            t0 = time.perf_counter()
            model, state = create_sharded_model_and_state()
            master_log(logger, f"[timing] create_sharded_model_and_state: {time.perf_counter() - t0:.1f}s")
            t0 = time.perf_counter()
            opt_state = optimizer_outer_loop.init(model.trainable_parameters())  # Sharding taken from model
            master_log(logger, f"[timing] optimizer_init: {time.perf_counter() - t0:.1f}s")
            start_step = 0
        master_log(logger, f"[timing] total setup (model+ckpt+opt): {time.perf_counter() - t_setup:.1f}s")

        ### Include Storage
        num_trainable_params = sum(x.size for x in jax.tree_util.tree_leaves(model.trainable_parameters()))
        num_non_embedding_params = num_trainable_params - model.language_model.model.wte.weight.size
        if model.language_model.lm_head is not None:
            num_non_embedding_params -= model.language_model.lm_head.weight.size
        logger.info(f"#Trainable params: {num_trainable_params}")
        logger.info(f"#Non-embed params: {num_non_embedding_params}")
        log_memory_breakdown(model, opt_state, step=-1, wandb_logger=wandb_logger)

        M = MetaModel.MetricType
        evaluator = Evaluator(
            global_batch_size=max(cfg.training.eval_batch_size, cfg.training.global_batch_size // cfg.training.accum_steps * 4),  # Larger bs to speed up eval
            data_sharding=data_sharding,
            config=cfg,
            wandb_logger=wandb_logger,
            log_dir=log_dir,
            chunks_dir=resolved_eval_chunks_dir,
        )

        total_steps = cfg.training.total_steps
        assert total_steps >= 1, "Total step must >=1, otherwise, lower global batch size"
        eval_steps = {round((total_steps - 1) * i / cfg.training.num_evals) for i in range(1, cfg.training.num_evals + 1)}
        master_log(logger, f"Total steps: {total_steps}, eval at steps: {sorted(eval_steps)}")

        with mesh:
            if cfg.training.eval_mode or start_step == total_steps:
                state = state.set(model.step_index, jnp.array(jnp.iinfo(jnp.int32).max - 100, dtype=jnp.int32))
                t_eval = time.perf_counter()
                evaluator.eval_fn(model, state, start_step)
                master_log(logger, f"[timing] eval (step {start_step}): {time.perf_counter() - t_eval:.1f}s")
                jax.experimental.multihost_utils.sync_global_devices("eval finished")
                return

            tqdm = get_custom_tqdm() if master_process else _tqdm
            prefetcher = _BatchPrefetcher(train_ds_iter, to_sharded_batch)
            for step in tqdm(range(start_step, total_steps), initial=start_step, total=total_steps, desc="Outer Loop Training", disable=not master_process):
                if 0 < cfg.training.break_step < step:
                    jax.experimental.multihost_utils.sync_global_devices("reached break step")
                    break

                t_batch = time.perf_counter()
                batch = prefetcher.next()
                t_batch = time.perf_counter() - t_batch

                state = state.set(model.step_index, jnp.array(step, dtype=jnp.int32))

                profile_this_step = step == start_step + 1
                trace_dir = tempfile.mkdtemp(prefix="jax_trace_") if profile_this_step else None
                try:
                    is_first_step = step == start_step
                    if is_first_step:
                        master_log(logger, f"Step {step}: dispatching train_on_sequence (first call triggers XLA compilation)...")
                    t_train = time.perf_counter()

                    if profile_this_step:
                        with jax.profiler.trace(trace_dir):
                            model, opt_state, loss, metrics = train_on_sequence(state, model, opt_state, batch, cfg)
                    else:
                        model, opt_state, loss, metrics = train_on_sequence(state, model, opt_state, batch, cfg)

                    if is_first_step:
                        t_dispatch = time.perf_counter() - t_train
                        master_log(logger, f"Step {step}: dispatch returned in {t_dispatch:.1f}s, waiting for result (block_until_ready)...")
                        jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else None, (loss, metrics))
                        t_total = time.perf_counter() - t_train
                        master_log(logger, f"Step {step}: train_on_sequence complete — dispatch={t_dispatch:.1f}s  total={t_total:.1f}s  data_load={t_batch:.1f}s")
                except Exception:
                    if step == start_step:
                        log_memory_breakdown(model, opt_state, step=step, wandb_logger=wandb_logger)
                    raise

                if step == start_step:
                    log_memory_breakdown(model, opt_state, step=step, wandb_logger=wandb_logger)

                if trace_dir is not None:
                    for f in Path(trace_dir).rglob("*"):
                        if f.is_file():
                            wandb_logger.save(str(f), base_path=trace_dir)
                    master_log(logger, f"Uploaded JAX profiler trace from {trace_dir} to wandb")

                loss_ce = metrics[M.loss].mean()

                update = {
                    "loss": jax.device_get(loss_ce).item(),
                    "gradient_norm": jax.device_get(metrics[M.outer_grad_norm]).item(),
                    "outer_learning_rate": jnp.asarray(optimizer_info_outer_loop["learning_rate_schedule"](int(opt_state[1][2].count) - 1)).item(),
                }

                log_memory_gauge(step, wandb_logger)
                wandb_logger.log(update, step)

                if (cfg.training.save_milestone_freq > 0 and step % cfg.training.save_milestone_freq == 0 and step != 0) or (step == cfg.training.total_steps - 1):
                    master_log(logger, f"Saving checkpoint at step {step}, do not kill...")
                    is_milestone = (cfg.training.save_milestone_freq > 0) and (step % cfg.training.save_milestone_freq == 0)

                    checkpointer.save_checkpoint(
                        step=step,
                        model=model,
                        opt_state=opt_state,
                        train_ds_iter=prefetcher.underlying_iter,
                        is_milestone=is_milestone,
                    )

                    checkpointer.wait_until_finished()

                if step in eval_steps:
                    t_eval = time.perf_counter()
                    evaluator.eval_fn(model, state, step)
                    master_log(logger, f"[timing] eval (step {step}): {time.perf_counter() - t_eval:.1f}s")

            prefetcher.close()
            checkpointer.close()  # Always wait until checkpoints are done saving

            if cfg.backend.distributed:
                jax.experimental.multihost_utils.sync_global_devices("end_of_training")


@hydra.main(version_base=None, config_path=str(Path("configs").absolute().resolve()), config_name="config")
def main(cfg: Config):
    if cfg.backend.compilation_cache_dir is not None:
        import jax

        jax.config.update("jax_compilation_cache_dir", cfg.backend.compilation_cache_dir)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    logging.getLogger("jax.experimental.compilation_cache.compilation_cache").setLevel(logging.DEBUG)
    _main(cfg)


def _preprocess_split(cfg: Config, split: str, total_sequences: int, chunk_idx: int | None = None) -> None:
    model_cfg = cfg.model
    key = chunks_volume_key(model_cfg.vocab_size, cfg.training.dataset_name, cfg.training.seq_length, split)
    output_dir = Path(cfg.training.chunks_dir) / key

    num_chunks = math.ceil(total_sequences / CHUNK_SIZE)

    if chunk_idx is not None:
        if chunk_idx >= num_chunks:
            print(f"chunk_idx {chunk_idx} >= num_chunks {num_chunks}, nothing to do for split={split}")
            return
        indices = [chunk_idx]
    else:
        indices = list(range(num_chunks))

    for idx in indices:
        save_chunk(
            path=cfg.training.dataset_path,
            split=split,
            seq_len=cfg.training.seq_length,
            output_dir=str(output_dir),
            chunk_idx=idx,
            vocab_size=model_cfg.vocab_size,
            tokenizer_name=cfg.training.tokenizer_name,
            bos_token_id=model_cfg.bos_token_id,
            eos_token_id=model_cfg.eos_token_id,
        )


def _total_chunks(cfg: Config) -> int:
    train_sequences = cfg.training.total_steps * cfg.training.global_batch_size
    eval_batch_size = max(cfg.training.eval_batch_size, cfg.training.global_batch_size // cfg.training.accum_steps * 4)
    eval_sequences = cfg.training.max_eval_batches * eval_batch_size
    return max(math.ceil(train_sequences / CHUNK_SIZE), math.ceil(eval_sequences / CHUNK_SIZE))


def _preprocess(cfg: Config, chunk_idx: int | None = None) -> None:
    assert cfg.training.chunks_dir, "training.chunks_dir must be set for preprocessing"

    train_sequences = cfg.training.total_steps * cfg.training.global_batch_size
    _preprocess_split(cfg, cfg.training.data_split, train_sequences, chunk_idx=chunk_idx)

    eval_batch_size = max(cfg.training.eval_batch_size, cfg.training.global_batch_size // cfg.training.accum_steps * 4)
    eval_sequences = cfg.training.max_eval_batches * eval_batch_size
    _preprocess_split(cfg, cfg.training.eval_split, eval_sequences, chunk_idx=chunk_idx)


@hydra.main(version_base=None, config_path=str(Path("configs").absolute().resolve()), config_name="config")
def preprocess_main(cfg: Config):
    if cfg.training.preprocess_chunk_idx == -2:
        print(f"CHUNK_COUNT={_total_chunks(cfg)}")
        return
    chunk_idx = cfg.training.preprocess_chunk_idx if cfg.training.preprocess_chunk_idx >= 0 else None
    _preprocess(cfg, chunk_idx=chunk_idx)


if __name__ == "__main__":
    main()
