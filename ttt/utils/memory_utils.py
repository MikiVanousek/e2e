import logging

import jax

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def pytree_bytes(tree) -> int:
    return sum(x.nbytes for x in jax.tree_util.tree_leaves(tree))


def log_memory_breakdown(model, opt_state, step, wandb_logger):
    if jax.process_index() != 0:
        return

    device = jax.local_devices()[0]
    stats = device.memory_stats()
    if stats is None:
        logger.warning("memory_stats() not supported on this backend")
        return

    model_bytes = pytree_bytes(model)
    opt_bytes = pytree_bytes(opt_state)
    total_in_use = stats["bytes_in_use"]
    peak = stats["peak_bytes_in_use"]
    limit = stats["bytes_limit"]

    other = total_in_use - model_bytes - opt_bytes
    activation_peak = peak - total_in_use

    def gb(b):
        return b / 1e9

    logger.info(
        f"[Memory @ step {step}] "
        f"model={gb(model_bytes):.2f}GB  opt={gb(opt_bytes):.2f}GB  "
        f"other={gb(other):.2f}GB  act_peak={gb(activation_peak):.2f}GB  "
        f"total={gb(total_in_use):.2f}GB  peak={gb(peak):.2f}GB  "
        f"limit={gb(limit):.2f}GB"
    )

    metrics = {
        "memory/model_weights_gb": gb(model_bytes),
        "memory/optimizer_state_gb": gb(opt_bytes),
        "memory/other_gb": gb(other),
        "memory/activation_peak_gb": gb(activation_peak),
        **_device_memory_metrics(stats),
    }
    wandb_logger.log(metrics, step)


def _device_memory_metrics(stats: dict) -> dict:
    def gb(b):
        return b / 1e9

    return {
        "memory/total_in_use_gb": gb(stats["bytes_in_use"]),
        "memory/peak_gb": gb(stats["peak_bytes_in_use"]),
        "memory/device_limit_gb": gb(stats["bytes_limit"]),
        "memory/utilization_pct": stats["bytes_in_use"] / stats["bytes_limit"] * 100,
    }


def log_memory_gauge(step: int, wandb_logger):
    if jax.process_index() != 0:
        return
    device = jax.local_devices()[0]
    stats = device.memory_stats()
    if stats is None:
        return
    wandb_logger.log(_device_memory_metrics(stats), step)
