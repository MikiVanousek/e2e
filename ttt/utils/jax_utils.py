import logging
import os
import random
import typing as tp
from collections.abc import Callable, Hashable
from functools import partial, wraps
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from einops import rearrange
from jax import lax
from jaxtyping import PRNGKeyArray, PyTree
from optax._src import numerics
from tqdm import tqdm as _tqdm

from ttt.config import JaxDistributedConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Dtype = jax.typing.DTypeLike | Any


def master_log(logger, *args, level=logging.INFO, **kwargs):
    if jax.process_index() == 0:
        logger.log(level, *args, **kwargs)


def initialize_distibuted(distributed_config: JaxDistributedConfig):
    if distributed_config.backend:
        os.environ["JAX_PLATFORM_NAME"] = distributed_config.backend

        if distributed_config.backend == "cpu":
            cpu_count = os.cpu_count()
            if distributed_config.num_devices and cpu_count is not None and cpu_count < distributed_config.num_devices:
                raise ValueError(f"Requested {distributed_config.num_devices} CPU devices, but only {os.cpu_count()} are available.")

            core_count = distributed_config.num_devices or os.cpu_count()
            os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={core_count}"

    if distributed_config.distributed:
        try:
            local_device_ids = None
            if distributed_config.local_device_ids:
                local_device_ids = [int(x) for x in distributed_config.local_device_ids.split(",")]

            jax.distributed.initialize(
                coordinator_address=distributed_config.coordinator_address,
                num_processes=distributed_config.num_processes,
                process_id=distributed_config.process_id,
                local_device_ids=local_device_ids,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize JAX distributed: {e}")


def get_float_dtype_by_name(dtype):
    match dtype:
        case "bf16" | "bfloat16":
            return jnp.bfloat16
        case "fp16" | "float16":
            return jnp.float16
        case "fp32" | "float32":
            return jnp.float32
        case "fp64" | "float64":
            return jnp.float64
        case _:
            raise ValueError(f"Unknown dtype: {dtype}")


def get_gradient_checkpoint_policy(
    name: Literal["everything_saveable", "nothing_saveable", "checkpoint_dots", "checkpoint_dots_with_no_batch_dims"] | Callable[..., bool],
):
    if not isinstance(name, str):
        return name
    match name:
        case "everything_saveable":
            return jax.checkpoint_policies.everything_saveable
        case "nothing_saveable":
            return jax.checkpoint_policies.nothing_saveable
        case "checkpoint_dots":
            return jax.checkpoint_policies.checkpoint_dots
        case "checkpoint_dots_with_no_batch_dims":
            return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
        case _:
            raise ValueError(f"Unknown policy: {name}")


def set_random_seed(seed: int) -> PRNGKeyArray:
    np.random.seed(seed)
    random.seed(seed)

    return jrandom.PRNGKey(seed)


def get_custom_tqdm():
    logger = logging.getLogger("Custom TQDM Timing")
    logger.setLevel(logging.INFO)  # Set the logging level to INFO

    class tqdm(_tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.warmup_time_elapsed = 0

        def update(self, n=1):
            super().update(n)
            warmup_steps = 50
            step_passed = self.n - self.initial
            if step_passed == warmup_steps:
                self.warmup_time_elapsed = self.format_dict["elapsed"]
                logger.info(f"Warmup {warmup_steps} Iteration Time: {self.format_interval(self.warmup_time_elapsed)}")
            if (step_passed > warmup_steps and step_passed % 100 == 0) or self.n == self.total:
                # NOTE: the starting up time is also included in the elapsed time
                elapsed = self.format_dict["elapsed"] - self.warmup_time_elapsed
                inv_rate = elapsed / (step_passed - warmup_steps)
                eta = (self.total - self.n) * inv_rate
                logger.info(
                    f"{self.n}/{self.total}: Average Speed: {inv_rate:.2f} s/it, elapsed: {self.format_interval(elapsed)} | remaining: {self.format_interval(eta)} "
                )

    return tqdm


def vmap_mean(fun, batch, *, axis_name: Hashable):
    vmap_dim_size = jax.tree.flatten(batch)[0][0].shape[0]
    if vmap_dim_size == 1:
        single_microbatch = tree_rearrange(batch, "1 ... -> ...")
        return fun(single_microbatch)

    @partial(jax.vmap, in_axes=(0,), out_axes=None, axis_name=axis_name)
    def vmapped_fn(x):
        return jax.lax.pmean(fun(x), axis_name=axis_name)

    return vmapped_fn(batch)


def welfords_online_mean(fun, batch):
    """
    Compute mean without storing intermediary results in memory. This function implements Welford's algorithm for numerical accuracy.
    Mathematically equivalent to `mean([fun(x) for x in batch], axis=0)` for PyTree inputs and outputs to `fun`.

    Short-circuits to `fun` if the number of loops is 1.

    Not sure if this should used when gradients need to be computed based on the meaned results since the computational graph might be quite deep.

    Args:
        fun: Function to evaluate on each element of the batch.
        batch: Batch of data to scan through. The number of loops is equal to the outer dimension of this PyTree.

    Returns:
        Meaned results.
    """
    num_loops = jax.tree.flatten(batch)[0][0].shape[0]
    if num_loops == 1:  # Skip if trivial
        single_microbatch = tree_rearrange(batch, "1 ... -> ...")
        return fun(single_microbatch)

    def update_online_grad_mean(carry, batch_slice):
        """Welford's online mean algorithm for stable numerics"""
        (acc_carry, count) = carry

        acc_delta = fun(batch_slice)

        acc_carry = jax.tree.map(lambda delta, acc: acc + (delta - acc) / count, acc_delta, acc_carry)

        return (acc_carry, count + 1), None

    first_batch_slice = jax.tree.map(lambda x: x[0], batch)

    acc_init = jax.tree.map(lambda x: jnp.zeros_like(x), jax.eval_shape(fun, first_batch_slice))
    count_init = 1

    (acc_result, _count), _ = lax.scan(update_online_grad_mean, (acc_init, count_init), batch)

    return acc_result


def scan_or_loop(
    f,
    init,
    xs,
    use_loop=False,
):
    """
    Version of scan that can be switched to a loop for debugging purposes.
    Using unroll=True still requires the function to be jitted, so will mess up NaN debugging.
    """
    if not use_loop:
        return jax.lax.scan(f, init, xs)

    carry = init
    xs_size = jax.tree.leaves(xs)[0].shape[0]
    ys = []
    for i in range(xs_size):
        x = tree_slice(xs, i)
        carry, y = f(carry, x)
        ys.append(y)

    stack_args = lambda *args: jnp.stack(args) if not all(arg is None for arg in args) else None  # Nones should stack to None

    return carry, jax.tree.map(stack_args, *ys)


def scan_remat_chunk(f, carry, x, *, remat_n_loops: int, unroll: bool):
    """
    Remat every n steps. Allow for a scan or a loop.

    Args:
        f: Function to apply to each chunk.
        remat_n_loops: Number of loops to remat. If 0, no remat is performed.
        carry: Initial carry value.
        x: Input PyTree.
        use_scan: Whether to use a scan or a loop.
    """

    num_loops = jax.tree.leaves(x)[0].shape[0]

    if remat_n_loops == 0:
        carry, y = scan_or_loop(f, carry, x, use_loop=unroll)
        return carry, y

    n_remat_chunks = num_loops // remat_n_loops

    x_grouped = tree_rearrange(x, "(remat_chunk remat_loops) ... -> remat_chunk remat_loops ...", remat_chunk=n_remat_chunks, remat_loops=remat_n_loops)

    @partial(jax.remat, prevent_cse=False, policy=get_gradient_checkpoint_policy("nothing_saveable"))
    def chunk_f(carry, x_chunk):
        return scan_or_loop(f, carry, x_chunk, use_loop=unroll)

    carry, result = scan_or_loop(chunk_f, carry, x_grouped, use_loop=unroll)

    result = tree_rearrange(result, "remat_chunk remat_loops ... -> (remat_chunk remat_loops) ...")
    return carry, result


def tree_slice(tree: PyTree, i: int) -> PyTree:
    return jax.tree.map(lambda x: x[i], tree)


def tree_rearrange(tree: PyTree, pattern: str, **axes_lengths) -> PyTree:
    def rearrange_fn(x):
        return rearrange(x, pattern, **axes_lengths)

    return jax.tree.map(rearrange_fn, tree)


def canonicalize_dtype(*args, dtype: Dtype | None = None, inexact: bool = True) -> Dtype:
    """Copied from linen https://flax.readthedocs.io/en/latest/_modules/flax/nnx/nn/dtypes.html#canonicalize_dtype"""
    if dtype is None:
        args_filtered = [jnp.asarray(x) for x in args if x is not None]
        dtype = jnp.result_type(*args_filtered)
        if inexact and not jnp.issubdtype(dtype, jnp.inexact):
            dtype = jnp.promote_types(jnp.float32, dtype)
    if inexact and not jnp.issubdtype(dtype, jnp.inexact):
        raise ValueError(f"Dtype must be inexact: {dtype}")
    return dtype


def promote_dtype(*args, dtype=None, inexact=True) -> list[Any]:
    """Copied from linen https://flax.readthedocs.io/en/latest/_modules/flax/nnx/nn/dtypes.html#canonicalize_dtype"""
    dtype = canonicalize_dtype(*args, dtype=dtype, inexact=inexact)
    return [jnp.asarray(x, dtype) if x is not None else None for x in args]


def eval_shape_and_sharding(f, *args, **kwargs):
    """
    Like `jax.eval_shape`, but also retains output sharding information by compiling the model.
    """
    f_jit = jax.jit(f)
    shapes = f_jit.eval_shape(*args, **kwargs)
    sharding = f_jit.lower(*args, **kwargs).compile().output_shardings

    def add_sharding(shapes, sharding):
        shapes.sharding = sharding
        return shapes

    return jax.tree.map(add_sharding, shapes, sharding)


_StaticArgs = tp.TypeVar("_StaticArgs")
_SavedArgs = tp.TypeVar("_SavedArgs", bound=PyTree)


def remat_bwd(
    fun: tp.Callable[..., tp.Any],
    *,
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: Callable[..., bool] | None = None,
) -> Callable[..., tp.Any]:
    """
    Like `jax.remat`, but applies only to the backward pass of the function.
    This means you can choose what to save for the backward pass of the backward pass.

    Args:
        compute_fn: The function to apply remat to.
        prevent_cse: Whether to prevent common subexpression elimination. Set to False for remats inside scan loops.
        static_argnums: Which arguments to treat as static for the remat.
        policy: The checkpoint policy to use.
    """

    @wraps(fun)
    @jax.custom_vjp
    def standard_fn(*args):
        return fun(*args)

    @partial(jax.remat, prevent_cse=prevent_cse, policy=policy, static_argnums=static_argnums)
    def fwd_fn(*args):
        output, vjp_compute_fn = jax.vjp(fun, *args)
        residuals = vjp_compute_fn
        return output, residuals

    @partial(jax.remat, prevent_cse=prevent_cse, policy=policy, static_argnums=static_argnums)
    def bwd_fn(residuals: _SavedArgs, g):
        dl_d_output = g
        vjp_compute_fn = residuals
        d_saved_args = vjp_compute_fn(dl_d_output)
        return d_saved_args

    standard_fn.defvjp(fwd_fn, bwd_fn)

    standard_fn = jax.remat(standard_fn, prevent_cse=prevent_cse, policy=policy, static_argnums=static_argnums)

    return standard_fn


def clone_pytree(tree: PyTree):
    """
    'Clone' a pytree, preserving all leaf values but recreating the structure.

    Useful e.g. for avoiding state invalidation issues with Equinox.State.{get, set} when we want to reuse model state.
    For example, for evaluating inner loop loss for a meta model after taking an inner loop gradient step.
    See https://github.com/patrick-kidger/equinox/blob/main/equinox/nn/_stateful.py#L85.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    tree_clone = jax.tree_util.tree_unflatten(treedef, leaves)
    return tree_clone


def maybe_remat(
    fun: Callable,
    *,
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: str,
) -> Callable[..., bool]:
    """Calls `jax.remat` with the arguments and the correct policy only if the policy is non-empty (not \"\")."""
    if policy:
        return jax.remat(fun, prevent_cse=prevent_cse, static_argnums=static_argnums, policy=get_gradient_checkpoint_policy(policy))
    else:
        return fun


@wraps(remat_bwd)
def maybe_remat_bwd(
    fun: tp.Callable[..., tp.Any],
    *,
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: str,
) -> Callable:
    if policy:
        return remat_bwd(fun, prevent_cse=prevent_cse, static_argnums=static_argnums, policy=get_gradient_checkpoint_policy(policy))
    else:
        return fun


def maybe_double_remat(
    fun: Callable,
    *,
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy_remat: str,
    policy_remat_bwd: str,
) -> Callable:
    return maybe_remat_bwd(
        fun=maybe_remat(
            fun,
            prevent_cse=prevent_cse,
            static_argnums=static_argnums,
            policy=policy_remat,
        ),
        prevent_cse=prevent_cse,
        static_argnums=static_argnums,
        policy=policy_remat_bwd,
    )


def safe_sqrt(x, eps=1e-5):
    """Must always be positive"""
    return jnp.sqrt(x + eps)


def global_norm_safe(updates):
    """Compute the global norm across a nested structure of tensors."""
    return safe_sqrt(sum(jnp.sum(numerics.abs_sq(x)) for x in jax.tree.leaves(updates)))
