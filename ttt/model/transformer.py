from __future__ import annotations

from enum import StrEnum, auto

import equinox as eqx
import jax
import jax.ad_checkpoint
import jax.nn
import jax.numpy as jnp
import jax.random as jrandom
from einops import rearrange
from equinox import nn
from jaxtyping import PRNGKeyArray
from optax import OptState

from ttt.config import Config, ModelConfig
from ttt.model.attention import SWA, Attention, AttentionBase, NormalLinear, SWAFull
from ttt.model.data import BaseModelOutput, Batch
from ttt.model.loss import cross_entropy_loss_and_accuracy, token_log_probs
from ttt.optimizers import make_optimizer
from ttt.utils.filter_utils import filter_apply_updates, filter_parameters, get_filter_spec
from ttt.utils.jax_utils import (
    clone_pytree,
    get_float_dtype_by_name,
    maybe_double_remat,
    promote_dtype,
    scan_or_loop,
    scan_remat_chunk,
    tree_rearrange,
)


IP_TTT_TARGET_CONV_CURRENT_LAYER = "conv_current_layer"
IP_TTT_TARGET_NEXT_CURRENT_LAYER = "next_current_layer"
IP_TTT_TARGET_NEXT_EMBEDDING = "next_embedding"
IP_TTT_TARGET_CONV_EMBEDDING = "conv_embedding"


def _uses_embedding_ip_ttt_target(config: ModelConfig) -> bool:
    return config.ip_ttt and config.ip_ttt_target in (IP_TTT_TARGET_NEXT_EMBEDDING, IP_TTT_TARGET_CONV_EMBEDDING)


def _one_token_future(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([x[1:], jnp.zeros_like(x[:1])], axis=0)


class SwiGLUMLP(eqx.Module):
    """Single SwiGLU MLP block"""

    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)
    w1: NormalLinear
    w2: NormalLinear
    w3: NormalLinear
    dropout: nn.Dropout = eqx.field(static=True)

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.config = config
        self.compute_dtype = get_float_dtype_by_name(self.config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(self.config.param_dtype)

        w1_key, w2_key, w3_key = jrandom.split(key, 3)

        self.w1 = NormalLinear(
            self.config, in_features=config.hidden_size, out_features=config.intermediate_size, std=config.initializer_range, key=w1_key, name="w1"
        )

        self.w2 = NormalLinear(
            self.config, in_features=config.intermediate_size, out_features=config.hidden_size, std=config.initializer_range, key=w2_key, name="w2"
        )

        self.w3 = NormalLinear(
            self.config, in_features=config.hidden_size, out_features=config.intermediate_size, std=config.initializer_range, key=w3_key, name="w3"
        )

        self.dropout = nn.Dropout(p=self.config.resid_pdrop)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z1 = self.w1(x)
        z1_act = jax.nn.silu(z1)
        z3 = self.w3(x)
        x2 = z1_act * z3
        z2 = self.w2(x2)
        output = self.dropout(z2)
        return output


class SwiGLUMLPWithIPTTT(eqx.Module):
    """SwiGLU MLP with In-Place TTT fast-weight updates on w2 (down_proj).

    Port of TTTLlamaMLP from the PyTorch reimplementation.
    Reference: Feng et al., "In-Place Test-Time Training", ICLR 2026 Oral.

    The w2 (down projection) weight is treated as fast weights updated
    chunk-wise along the sequence via a prefix-sum of outer-product deltas.
    At init, ttt_conv is zero and ttt_proj is near-identity, so the initial
    behaviour matches the standard SwiGLUMLP.
    """

    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)
    w1: NormalLinear
    w2: NormalLinear
    w3: NormalLinear
    dropout: nn.Dropout = eqx.field(static=True)

    ttt_conv_weight: jax.Array
    ttt_proj_weight: jax.Array | None
    ttt_chunk: int = eqx.field(static=True)
    ttt_lr: float = eqx.field(static=True)
    ttt_disabled: jax.Array

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
        base_mlp: SwiGLUMLP | None = None,
    ):
        self.config = config
        self.compute_dtype = get_float_dtype_by_name(self.config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(self.config.param_dtype)

        if base_mlp is not None:
            self.w1 = base_mlp.w1
            self.w2 = base_mlp.w2
            self.w3 = base_mlp.w3
            self.dropout = base_mlp.dropout
        else:
            w1_key, w2_key, w3_key, key = jrandom.split(key, 4)
            self.w1 = NormalLinear(config, in_features=config.hidden_size, out_features=config.intermediate_size, std=config.initializer_range, key=w1_key, name="w1")
            self.w2 = NormalLinear(config, in_features=config.intermediate_size, out_features=config.hidden_size, std=config.initializer_range, key=w2_key, name="w2")
            self.w3 = NormalLinear(config, in_features=config.hidden_size, out_features=config.intermediate_size, std=config.initializer_range, key=w3_key, name="w3")
            self.dropout = nn.Dropout(p=config.resid_pdrop)

        self.ttt_chunk = config.ip_ttt_chunk
        self.ttt_lr = config.ip_ttt_lr
        self.ttt_disabled = jnp.array(False)

        key_conv, key_proj = jrandom.split(key, 2)

        # Depthwise conv1d: [hidden_size, 1, kernel_size] -- zero-init
        self.ttt_conv_weight = jnp.zeros((config.hidden_size, 1, 5), dtype=self.param_dtype)

        # Projection: diagonal-normal init so initial update is near-zero
        if config.ip_ttt_proj:
            proj = jnp.zeros((config.hidden_size, config.hidden_size), dtype=self.param_dtype)
            diag = jrandom.normal(key_proj, shape=(config.hidden_size,), dtype=self.param_dtype) * config.initializer_range
            proj = proj.at[jnp.arange(config.hidden_size), jnp.arange(config.hidden_size)].set(diag)
            self.ttt_proj_weight = proj
        else:
            self.ttt_proj_weight = None

    def _ttt_forward(self, x: jnp.ndarray, h: jnp.ndarray, ttt_target: jnp.ndarray | None = None) -> jnp.ndarray:
        x, h, ttt_target = promote_dtype(x, h, ttt_target, dtype=self.compute_dtype)
        w2_weight = self.w2.weight.astype(self.compute_dtype)  # [intermediate, hidden]

        seq_len = x.shape[0]
        d = x.shape[-1]
        h_dim = h.shape[-1]
        chunk = self.ttt_chunk

        pad_len = (chunk - seq_len % chunk) % chunk
        h_pad = jnp.pad(h, ((0, pad_len), (0, 0)))

        n_chunks = h_pad.shape[0] // chunk
        h_c = h_pad.reshape(n_chunks, chunk, h_dim)

        if self.config.ip_ttt_target in (IP_TTT_TARGET_CONV_CURRENT_LAYER, IP_TTT_TARGET_CONV_EMBEDDING):
            if self.config.ip_ttt_target == IP_TTT_TARGET_CONV_EMBEDDING:
                if ttt_target is None:
                    raise ValueError("ip_ttt_target='conv_embedding' requires a TTT target")
                target = ttt_target
            else:
                target = x
            target_pad = jnp.pad(target, ((0, pad_len), (0, 0)))
            t = target_pad.reshape(n_chunks, chunk, d)
            # NTP-aligned target via depthwise conv
            # conv_w: [d, 1, 5], input must be [batch, d, chunk] for groups=d
            conv_w = self.ttt_conv_weight.astype(self.compute_dtype)  # [d, 1, 5]
            t_transposed = t.transpose(0, 2, 1)  # [n_chunks, d, chunk]
            t_conv = jax.lax.conv_general_dilated(
                t_transposed,
                conv_w,
                window_strides=(1,),
                padding=((2, 2),),
                feature_group_count=d,
            )  # [n_chunks, d, chunk]
            t = t_conv.transpose(0, 2, 1)  # [n_chunks, chunk, d]
        elif self.config.ip_ttt_target == IP_TTT_TARGET_NEXT_CURRENT_LAYER:
            t = _one_token_future(x)
            t = jnp.pad(t, ((0, pad_len), (0, 0))).reshape(n_chunks, chunk, d)
        elif self.config.ip_ttt_target == IP_TTT_TARGET_NEXT_EMBEDDING:
            if ttt_target is None:
                raise ValueError("ip_ttt_target='next_embedding' requires a TTT target")
            t = jnp.pad(ttt_target, ((0, pad_len), (0, 0))).reshape(n_chunks, chunk, d)
        else:
            raise ValueError(f"Unknown ip_ttt_target: {self.config.ip_ttt_target}")

        # Fast-weight deltas (apply-then-update: use chunks [:-1])
        # w2.weight is [intermediate, hidden] in JAX (transposed vs PyTorch)
        # h_c: [n, chunk, intermediate], t: [n, chunk, hidden]
        if self.ttt_proj_weight is not None:
            proj_w = self.ttt_proj_weight.astype(self.compute_dtype)  # [hidden, hidden]
            tz = jnp.einsum("ncd,de->nce", t[:-1], proj_w)  # [n-1, chunk, hidden]
            delta = jnp.einsum("nci,nch->nih", h_c[:-1], tz)  # [n-1, intermediate, hidden]
        else:
            delta = jnp.einsum("nci,nch->nih", h_c[:-1], t[:-1])  # [n-1, intermediate, hidden]

        # Prepend w2 weight, cumsum
        w0 = w2_weight[jnp.newaxis, :, :]  # [1, intermediate, hidden]
        w = jnp.concatenate([w0, delta * self.ttt_lr], axis=0).cumsum(axis=0)  # [n_chunks, intermediate, hidden]

        # Apply per-chunk weights: out_token = h_token @ w_chunk  (like h @ w2.weight)
        out = jnp.einsum("nci,nih->nch", h_c, w)  # [n_chunks, chunk, hidden]
        return out.reshape(-1, d)[:seq_len]

    def _vanilla_forward(self, h: jnp.ndarray) -> jnp.ndarray:
        return self.w2(h)

    def __call__(self, x: jnp.ndarray, ttt_target: jnp.ndarray | None = None) -> jnp.ndarray:
        z1 = self.w1(x)
        z1_act = jax.nn.silu(z1)
        z3 = self.w3(x)
        h = z1_act * z3  # [T, intermediate_size]

        out = jax.lax.cond(
            self.ttt_disabled,
            lambda: self._vanilla_forward(h),
            lambda: self._ttt_forward(x, h, ttt_target),
        )
        return self.dropout(out)


class PrimeStorage(eqx.Module):
    ffn_prime_norm: nn.RMSNorm
    ffn_prime_post_norm: nn.RMSNorm
    feed_forward_prime: SwiGLUMLP

    def __init__(
        self,
        config: ModelConfig,
        *,
        key,
    ) -> None:
        param_dtype = get_float_dtype_by_name(config.param_dtype)
        suffix_len = config.suffix_len

        suffix_keys = jrandom.split(key, suffix_len)
        if config.feed_forward_prime != "swiglu":
            raise NotImplementedError("Only feed_forward_prime='swiglu' is supported.")

        self.feed_forward_prime = jax.vmap(lambda k: SwiGLUMLP(config, key=k))(suffix_keys)
        if config.prime_zero_init:
            zero_w2 = jnp.zeros_like(self.feed_forward_prime.w2.weight)
            self.feed_forward_prime = eqx.tree_at(lambda m: m.w2.weight, self.feed_forward_prime, zero_w2)
        self.ffn_prime_norm = jax.vmap(lambda _: nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_bias=False, dtype=param_dtype))(suffix_keys)
        self.ffn_prime_post_norm = jax.vmap(lambda _: nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_bias=False, dtype=param_dtype))(suffix_keys)

    def __call__(self):
        pass


class Block(eqx.Module):
    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)

    seq_modeling_block: AttentionBase
    feed_forward: SwiGLUMLP
    seq_norm: nn.RMSNorm
    ffn_norm: nn.RMSNorm
    seq_post_norm: nn.RMSNorm
    ffn_post_norm: nn.RMSNorm
    ffn_prime_norm: nn.RMSNorm | None
    ffn_prime_post_norm: nn.RMSNorm | None
    feed_forward_prime: SwiGLUMLP | None

    def __init__(
        self,
        config: ModelConfig,
        *,
        key,
        feed_forward_prime: SwiGLUMLP | None = None,
        ffn_prime_norm: nn.RMSNorm = None,
        ffn_prime_post_norm: nn.RMSNorm | None = None,
    ) -> None:
        self.config = config
        self.compute_dtype = get_float_dtype_by_name(self.config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(self.config.param_dtype)

        seq_modeling_block_type = self.config.seq_modeling_block

        match seq_modeling_block_type:
            case "self_attention":
                seq_modeling_block = Attention
            case "SWA":
                seq_modeling_block = SWA
            case "SWAFull":
                seq_modeling_block = SWAFull
            case _:
                raise NotImplementedError(f"Sequence Modeling Layer {self.config.seq_modeling_block} Not Implemented.")

        key_seq_modeling_block, key_ffn = jrandom.split(key, 2)
        self.seq_modeling_block = seq_modeling_block(self.config, key=key_seq_modeling_block)
        if self.config.ip_ttt:
            self.feed_forward = SwiGLUMLPWithIPTTT(self.config, key=key_ffn)
        else:
            self.feed_forward = SwiGLUMLP(self.config, key=key_ffn)
        self.seq_norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)
        self.ffn_norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)
        self.seq_post_norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)
        self.ffn_post_norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)

        self.ffn_prime_norm = ffn_prime_norm
        self.ffn_prime_post_norm = ffn_prime_post_norm
        self.feed_forward_prime = feed_forward_prime

    def seq_modeling_forward(
        self, seq_modeling_block_fn, rms_forward_fn, seq_norm, seq_modeling_block, seq_post_norm, hidden_states, state: nn.State, seq: Batch, is_prefix: bool
    ):
        if self.config.pre_norm:
            seq_modeling_input = jax.vmap(lambda x: rms_forward_fn(seq_norm, x))(hidden_states)
        else:
            seq_modeling_input = hidden_states

        seq_modeling_hidden_states, state = seq_modeling_block_fn(seq_modeling_block, seq_modeling_input, seq, state, is_prefix)

        if self.config.post_norm:
            seq_modeling_hidden_states = jax.vmap(lambda x: rms_forward_fn(seq_post_norm, x))(seq_modeling_hidden_states)

        return seq_modeling_hidden_states, state

    def ffn_forward(
        self,
        feed_forward_fn,
        rms_forward_fn,
        ffn_norm,
        feed_forward,
        ffn_post_norm,
        hidden_states,
        ttt_target: jnp.ndarray | None = None,
        use_ip_ttt_target: bool = False,
    ):
        if self.config.pre_norm:
            feed_forward_input = jax.vmap(lambda x: rms_forward_fn(ffn_norm, x))(hidden_states)
        else:
            feed_forward_input = hidden_states

        if use_ip_ttt_target:
            feed_forward_hidden_states = feed_forward_fn(feed_forward, feed_forward_input, ttt_target)
        else:
            feed_forward_hidden_states = feed_forward_fn(feed_forward, feed_forward_input)

        if self.config.post_norm:
            feed_forward_hidden_states = jax.vmap(lambda x: rms_forward_fn(ffn_post_norm, x))(feed_forward_hidden_states)

        return feed_forward_hidden_states

    def __call__(self, hidden_states, state: nn.State, seq: Batch, is_prefix: bool = False, ttt_target: jnp.ndarray | None = None):
        config = self.config

        seq_modeling_block_fn = maybe_double_remat(
            self.seq_modeling_block.__class__.__call__, prevent_cse=True, policy_remat=config.remat_attention, policy_remat_bwd=config.remat_attention_bwd
        )
        feed_forward_fn = maybe_double_remat(
            self.feed_forward.__class__.__call__, prevent_cse=True, policy_remat=config.remat_mlp, policy_remat_bwd=config.remat_mlp_bwd
        )
        rms_forward_fn = maybe_double_remat(nn.RMSNorm.__call__, prevent_cse=True, policy_remat=config.remat_rms, policy_remat_bwd=config.remat_rms_bwd)
        if self.feed_forward_prime is not None:
            feed_forward_prime_fn = maybe_double_remat(
                self.feed_forward_prime.__class__.__call__, prevent_cse=True, policy_remat=config.remat_mlp, policy_remat_bwd=config.remat_mlp_bwd
            )

        seq_modeling_output, state = self.seq_modeling_forward(
            seq_modeling_block_fn, rms_forward_fn, self.seq_norm, self.seq_modeling_block, self.seq_post_norm, hidden_states, state, seq, is_prefix
        )

        hidden_states = hidden_states + seq_modeling_output

        feed_forward_prime_hidden_states = None
        if self.feed_forward_prime is not None:
            feed_forward_prime_hidden_states = self.ffn_forward(
                feed_forward_prime_fn,
                rms_forward_fn,
                self.ffn_prime_norm,
                self.feed_forward_prime,
                self.ffn_prime_post_norm,
                hidden_states,
            )

            feed_forward_prime_hidden_states = hidden_states + feed_forward_prime_hidden_states
            hidden_states = feed_forward_prime_hidden_states

        feed_forward_hidden_states = self.ffn_forward(
            feed_forward_fn,
            rms_forward_fn,
            self.ffn_norm,
            self.feed_forward,
            self.ffn_post_norm,
            hidden_states,
            ttt_target=ttt_target,
            use_ip_ttt_target=self.config.ip_ttt,
        )

        hidden_states = hidden_states + feed_forward_hidden_states

        return hidden_states, state

    def weights(self):
        return eqx.filter(self, eqx.is_inexact_array)

    def inner_parameters(self, config: Config):
        inner_specs = config.training.spec_inner
        inner_specs_rebased = []
        for spec in inner_specs:
            assert "suffix_blocks" in spec, "Inner params must lie in suffix blocks"
            # e.g., **.suffix_blocks.feed_forward_prime.** --> feed_forward_prime.**
            spec_rebased = spec.split("suffix_blocks")[1][1:]
            inner_specs_rebased.append(spec_rebased)
        return filter_parameters(self.weights(), inner_specs_rebased, "inner parameters")


class BlockCollectionSplit(eqx.Module):
    config: ModelConfig = eqx.field(static=True, repr=False)
    prefix_blocks: Block  # vmap-ed init and application
    suffix_blocks: Block  # vmap-ed init and application

    def __init__(
        self,
        config: ModelConfig,
        block_collection: Block,
        prime_storage: PrimeStorage,
        *,
        key: PRNGKeyArray,
    ):
        self.config = config
        suffix_len = self.config.suffix_len
        self.prefix_blocks = jax.tree.map(lambda m: m[:-suffix_len], block_collection) if suffix_len > 0 else block_collection
        self.suffix_blocks = None

        if suffix_len > 0:
            self.suffix_blocks = jax.tree.map(lambda m: m[-suffix_len:], block_collection)
            if prime_storage is not None:
                suffix_keys = jrandom.split(key, suffix_len)

                argdict = {"key": suffix_keys}
                argdict["ffn_prime_norm"] = prime_storage.ffn_prime_norm
                argdict["ffn_prime_post_norm"] = prime_storage.ffn_prime_post_norm
                argdict["feed_forward_prime"] = prime_storage.feed_forward_prime

                # copy in prime params
                suffix_blocks_template = jax.vmap(lambda **kwargs: Block(config=self.config, **kwargs))(**argdict)

                # copy in non-prime params
                self.suffix_blocks = eqx.tree_at(
                    lambda m: (m.seq_norm, m.seq_modeling_block, m.seq_post_norm, m.ffn_norm, m.feed_forward, m.ffn_post_norm),
                    suffix_blocks_template,
                    (
                        self.suffix_blocks.seq_norm,
                        self.suffix_blocks.seq_modeling_block,
                        self.suffix_blocks.seq_post_norm,
                        self.suffix_blocks.ffn_norm,
                        self.suffix_blocks.feed_forward,
                        self.suffix_blocks.ffn_post_norm,
                    ),
                )

    @staticmethod
    def split_state(state: nn.State, suffix_len: int):
        if suffix_len > 0:
            return (
                jax.tree.map(lambda s: s[:-suffix_len], state),
                jax.tree.map(lambda s: s[-suffix_len:] if len(s) >= suffix_len else jnp.zeros((suffix_len, *s.shape[1:]), dtype=s.dtype), state),
            )
        else:
            return (state, None)

    def prefix_call(self, prefix_blocks, hidden_states: jnp.ndarray, state: nn.State, seq: Batch, ttt_target: jnp.ndarray | None = None):
        if prefix_blocks is not None:
            def block_call(block, x, substate, seq, target):
                return block.__class__.__call__(block, x, substate, seq, is_prefix=True, ttt_target=target)

            block_fn = maybe_double_remat(block_call, prevent_cse=True, policy_remat=self.config.remat_prefix_block, policy_remat_bwd="")

            # Note: Prefix has no state
            def apply_block_prefix(x, block):
                x, _ = block_fn(block, x, None, seq, ttt_target)
                return x, None

            hidden_states, _ = jax.lax.scan(
                apply_block_prefix,
                hidden_states,
                prefix_blocks,
                unroll=self.config.unroll_block_scan,
            )

        outputs = BaseModelOutput(last_hidden_state=hidden_states, state=state)
        return outputs

    def suffix_call(self, hidden_states: jnp.ndarray, state: nn.State, seq: Batch, ttt_target: jnp.ndarray | None = None):
        if self.suffix_blocks is not None:
            def block_call(block, x, substate, seq, target):
                return block.__class__.__call__(block, x, substate, seq, is_prefix=False, ttt_target=target)

            block_fn = maybe_double_remat(
                block_call,
                prevent_cse=True,
                policy_remat=self.config.remat_block,
                policy_remat_bwd=self.config.remat_block_bwd,
            )

            def apply_block_suffix(x, block__substate):
                block, substate = block__substate
                x, substate = block_fn(block, x, substate, seq, ttt_target)
                return x, substate

            hidden_states, state = jax.lax.scan(
                apply_block_suffix,
                hidden_states,
                (self.suffix_blocks, state),
                unroll=self.config.unroll_block_scan,
            )

        outputs = BaseModelOutput(last_hidden_state=hidden_states, state=state)
        return outputs

    def __call__(
        self,
        hidden_states,
        state: tuple[nn.State, nn.State],
        seq: Batch,
        ttt_target: jnp.ndarray | None = None,
    ):
        def block_call(block, x, substate, seq, target):
            return block.__class__.__call__(block, x, substate, seq, is_prefix=False, ttt_target=target)

        block_fn = maybe_double_remat(block_call, prevent_cse=True, policy_remat=self.config.remat_block, policy_remat_bwd=self.config.remat_block_bwd)

        def apply_block_prefix(x, block__substate):
            block, substate = block__substate
            x, substate = block_fn(block, x, substate, seq, ttt_target)
            return x, substate

        substate_prefix, substate_suffix = state

        hidden_states, substate_prefix = jax.lax.scan(
            apply_block_prefix,
            hidden_states,
            (self.prefix_blocks, substate_prefix),
            unroll=self.config.unroll_block_scan,
        )

        if self.suffix_blocks is not None:

            def apply_block_suffix(x, block__substate):
                block, substate = block__substate
                x, substate = block_fn(block, x, substate, seq, ttt_target)
                return x, substate

            hidden_states, substate_suffix = jax.lax.scan(
                apply_block_suffix,
                hidden_states,
                (self.suffix_blocks, substate_suffix),
                unroll=self.config.unroll_block_scan,
            )

        outputs = BaseModelOutput(last_hidden_state=hidden_states, state=(substate_prefix, substate_suffix))
        return outputs


class BlockCollection(eqx.Module):
    config: ModelConfig = eqx.field(static=True, repr=False)
    blocks: Block  # vmap-ed init and application
    prime_storage: PrimeStorage

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.config = config

        key, prime_key = jrandom.split(key, 2)

        keys = jrandom.split(key, config.num_hidden_layers)
        self.blocks = jax.vmap(lambda k: Block(config, key=k))(keys)

        if config.ip_ttt:
            disabled = jnp.ones(config.num_hidden_layers, dtype=jnp.bool_)
            for idx in config.ip_ttt_layers:
                disabled = disabled.at[idx].set(False)
            self.blocks = eqx.tree_at(
                lambda m: m.feed_forward.ttt_disabled,
                self.blocks,
                disabled,
            )

        self.prime_storage = None

        if config.prime:
            self.prime_storage = PrimeStorage(config, key=prime_key)

    def __call__(
        self,
        hidden_states,
        state: nn.State,
        seq: Batch,
        ttt_target: jnp.ndarray | None = None,
    ):
        substate = state.substate(self.blocks)
        def block_call(block, x, substate, seq, target):
            return block.__class__.__call__(block, x, substate, seq, is_prefix=False, ttt_target=target)

        block_fn = maybe_double_remat(block_call, prevent_cse=True, policy_remat=self.config.remat_block, policy_remat_bwd=self.config.remat_block_bwd)

        def apply_block(x, block__substate):
            block, substate = block__substate

            x, substate = block_fn(block, x, substate, seq, ttt_target)
            return x, substate

        hidden_states, substate = scan_or_loop(apply_block, hidden_states, (self.blocks, substate), use_loop=self.config.unroll_block_scan)

        state = state.update(substate)

        outputs = BaseModelOutput(last_hidden_state=hidden_states, state=state)
        return outputs


class TransformerModel(eqx.Module):
    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)

    wte: nn.Embedding
    dropout: nn.Dropout = eqx.field(static=True)
    ln_f: nn.RMSNorm
    h: BlockCollection | BlockCollectionSplit

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.config = config
        self.compute_dtype = get_float_dtype_by_name(self.config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(self.config.param_dtype)

        key_embed, key_block = jrandom.split(key, 2)

        vocab_size, embed_dim = config.vocab_size, config.hidden_size

        self.wte = nn.Embedding(
            weight=jax.nn.initializers.normal(stddev=self.config.initializer_range, dtype=self.param_dtype)(key_embed, (vocab_size, embed_dim)),
        )

        self.dropout = nn.Dropout(p=self.config.embd_pdrop)
        self.h = BlockCollection(
            self.config,
            key=key_block,
        )
        self.ln_f = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)

    def wte_call(self, input_ids: jnp.ndarray):
        input_embeds = jax.vmap(self.wte)(input_ids.astype(jnp.int32))
        input_embeds = input_embeds.astype(self.compute_dtype)
        hidden_states = self.dropout(input_embeds)
        return hidden_states

    def ip_ttt_target_call(self, seq: Batch, input_embeds: jnp.ndarray | None = None):
        if not _uses_embedding_ip_ttt_target(self.config):
            return None
        if self.config.ip_ttt_target == IP_TTT_TARGET_NEXT_EMBEDDING:
            return self.wte_call(seq.target_tokens)
        if input_embeds is not None:
            return input_embeds
        return self.wte_call(seq.input_ids)

    def prefix_call(
        self,
        prefix: Block,
        hidden_states: jnp.ndarray,
        state: nn.State,
        seq: Batch,
    ):
        ttt_target = self.ip_ttt_target_call(seq, input_embeds=hidden_states)
        outputs: BaseModelOutput = self.h.prefix_call(prefix, hidden_states, state=state, seq=seq, ttt_target=ttt_target)
        return outputs

    def suffix_call(
        self,
        prefix_outputs: jnp.ndarray,
        state: nn.State,
        seq: Batch,
    ):
        ttt_target = self.ip_ttt_target_call(seq)
        outputs: BaseModelOutput = self.h.suffix_call(prefix_outputs, state=state, seq=seq, ttt_target=ttt_target)
        hidden_states = outputs.last_hidden_state
        hidden_states = jax.vmap(self.ln_f)(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states, state=outputs.state)

    def __call__(
        self,
        state: nn.State,
        seq: Batch,
    ):
        rms_forward_fn = maybe_double_remat(
            nn.RMSNorm.__call__, prevent_cse=True, policy_remat=self.config.remat_rms, policy_remat_bwd=self.config.remat_rms_bwd
        )

        input_embeds = jax.vmap(self.wte)(seq.input_ids.astype(jnp.int32))
        input_embeds = input_embeds.astype(self.compute_dtype)
        hidden_states = self.dropout(input_embeds)
        ttt_target = self.ip_ttt_target_call(seq, input_embeds=hidden_states)
        outputs: BaseModelOutput = self.h(
            hidden_states,
            state=state,
            seq=seq,
            ttt_target=ttt_target,
        )
        hidden_states = outputs.last_hidden_state
        hidden_states = jax.vmap(lambda x: rms_forward_fn(self.ln_f, x))(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states, state=outputs.state)


class MetaModel(eqx.Module):
    """Higher level model that includes all the trainable parameters and methods the full forward pass."""

    class Output(eqx.Module):
        lm_output: CausalLM.Output
        state: nn.State

    class MetricType(StrEnum):
        """Different metrics that can be logged to wandb, we pass these around in a dict."""

        loss = auto()
        token_nll_loss = auto()
        outer_grad_norm = auto()
        prime_grad_norm = auto()
        pretrained_grad_norm = auto()

    config: Config = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)
    state_dtype: jnp.dtype = eqx.field(static=True)

    step_index: nn.StateIndex
    language_model: CausalLM

    def __init__(
        self,
        config: Config,
        *,
        key: PRNGKeyArray,
    ):
        self.config = config
        self.compute_dtype = get_float_dtype_by_name(self.config.model.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(self.config.model.param_dtype)
        self.state_dtype = get_float_dtype_by_name(self.config.model.state_dtype)

        self.step_index = nn.StateIndex(jnp.array(0, dtype=jnp.int32))

        self.language_model = CausalLM(config.model, key=key)

    def get_ilr_multiplier(self, step: jnp.ndarray):
        if self.config.training.ilr_warmup_steps == 0:
            ilr_multiplier = 1.0
        else:
            assert self.config.training.ilr_warmup_steps > 0
            progress = jnp.minimum(1.0, 1.0 * (step + 1) / self.config.training.ilr_warmup_steps)
            ilr = self.config.training.ilr_init + (self.config.training.optimizer_inner.lr - self.config.training.ilr_init) * progress
            ilr_multiplier = (ilr / self.config.training.optimizer_inner.lr).astype(self.state_dtype)

        return ilr_multiplier

    def inner_optimizer(self, state: nn.State):
        step = state.get(self.step_index)
        ilr_multiplier = self.get_ilr_multiplier(step)
        optimizer, _optimizer_info = make_optimizer(self.config.training.optimizer_inner, ilr_multiplier)
        return optimizer

    def __call__(self, seq: Batch, state: nn.State) -> MetaModel.Output:
        pass

    class InnerLoopStepResult(eqx.Module):
        new_model: MetaModel
        new_optimizer_state: OptState
        new_state: nn.State
        metrics: dict[MetaModel.MetricType, jnp.ndarray]

        def __iter__(self):
            return iter((self.new_model, self.new_optimizer_state, self.new_state, self.metrics))

    def inner_loop_step(
        self,
        opt_state: OptState,
        state_tuple: tuple[nn.State, nn.State, nn.State],
        seq: Batch,
        prefix_outputs: jnp.ndarray,
    ) -> InnerLoopStepResult:
        """
        Perform a single step of inner loop training.

        Args:
            inner_optimizer_mut: The inner optimizer to update with. The model weights and optimizer state are updated in-place.
            seq: The sequence to train on.

        Returns:
            loss: The cross-entropy loss for the sequence.
            token_nll_loss: The token-level negative log likelihood loss.
        """

        M = MetaModel.MetricType
        md: dict[MetaModel.MetricType, jnp.ndarray] = {}

        state_all, suffix_state = state_tuple

        value_and_grad_fn = eqx.filter_value_and_grad(MetaModel.lm_loss, has_aux=True)

        (_loss_with_aux, (md[M.loss], md[M.token_nll_loss], new_suffix_state)), grads = value_and_grad_fn(
            self, seq, suffix_state, prefix_outputs=prefix_outputs
        )

        inner_grads = grads.inner_parameters()
        updates, new_optimizer_state = self.inner_optimizer(state_all).update(inner_grads, opt_state, self.inner_parameters())
        new_model = filter_apply_updates(self, updates)

        new_state_tuple = (state_all, new_suffix_state)

        return MetaModel.InnerLoopStepResult(new_model=new_model, new_optimizer_state=new_optimizer_state, new_state=new_state_tuple, metrics=md)

    def lm_loss(
        self, seq: Batch, state: nn.State, *, prefix_outputs: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, nn.State]]:
        if prefix_outputs is None:
            lm_outputs = self.language_model(
                seq=seq,
                state=state,
            )
        else:
            lm_outputs = self.language_model.suffix_call(prefix_outputs=prefix_outputs, state=state, seq=seq)

        loss, loss_pure_ce = cross_entropy_loss_and_accuracy(lm_outputs.logits, seq.target_tokens, seq.loss_masks)
        token_nll_loss = -token_log_probs(lm_outputs.logits, seq.target_tokens)

        return loss, (loss_pure_ce, token_nll_loss, lm_outputs.new_state)

    def loss_for_sequence(self, seq: Batch, state: nn.State, *, train_mode: str | None = None) -> tuple[jnp.ndarray, dict[MetricType, jnp.ndarray]]:
        """
        Process a sequence of data and compute the loss for the sequence.

        In the case of meta-learning, the inner loop will copy the weights so that this function does not mutate its inputs.

        Args:
            seq: Sequence of data to process. Should have no leading batch dimension. [T]

        Returns:
            loss: Loss for the sequence. []
            token_nll_loss: The token-level negative log likelihood loss for each token in the sequence. [T]
        """
        cfg = self.config

        block_collection = self.language_model.model.h.blocks
        prime_storage = self.language_model.model.h.prime_storage if cfg.model.prime else None
        new_collection = BlockCollectionSplit(
            cfg.model,
            block_collection=block_collection,
            prime_storage=prime_storage,
            key=jrandom.PRNGKey(0),
        )

        state_prefix_suffix = state.substate(self.language_model.model.h.blocks)

        state_prefix, state_suffix = BlockCollectionSplit.split_state(state_prefix_suffix, cfg.model.suffix_len)
        state_all = clone_pytree(state)

        self: MetaModel = eqx.tree_at(lambda m: m.language_model.model.h, self, new_collection)

        seqlen = cfg.training.seq_length
        tokens_per_chunk = cfg.model.mini_batch_size

        assert seqlen % tokens_per_chunk == 0, f"For now, seqlen {seqlen} must be divisible by chunk {tokens_per_chunk}"

        M = MetaModel.MetricType
        effective_mode = train_mode if train_mode is not None else cfg.training.train_mode

        if effective_mode == "meta":
            model: MetaModel = jax.tree.map(lambda p: p.astype(self.state_dtype), self)
            inner_opt_state = model.inner_optimizer(state_all).init(model.inner_parameters())

            xt_embed = self.language_model.wte_call(seq.input_ids)
            prefix_output = eqx.filter_checkpoint(self.language_model.prefix_call)(
                self.language_model.model.h.prefix_blocks, xt_embed, state_prefix, seq
            ).last_hidden_state

            def process_suffix_chunk(model__opt_state__state, inputs: tuple[Batch, jnp.ndarray]):
                model_inner, inner_opt_state, state_tuple = model__opt_state__state
                suffix_chunk, prefix_chunk = inputs

                spec_inner = get_filter_spec(model_inner, self.config.training.spec_inner, "inner parameters")
                inner_params, _ = eqx.partition(model_inner, spec_inner)
                _, outer_params = eqx.partition(model, spec_inner)
                model_inner: MetaModel = eqx.combine(inner_params, outer_params)

                new_model, inner_opt_state, state_tuple, metrics = MetaModel.inner_loop_step(
                    model_inner, inner_opt_state, state_tuple, suffix_chunk, prefix_chunk
                )

                return (new_model, inner_opt_state, state_tuple), metrics

            seq = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)
            prefix_output = tree_rearrange(prefix_output, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)

            carry, metrics = scan_remat_chunk(
                eqx.filter_checkpoint(process_suffix_chunk, prevent_cse=False),
                (model, inner_opt_state, (state_all, state_suffix)),
                (seq, prefix_output),
                remat_n_loops=cfg.training.inner_remat_freq,
                unroll=cfg.model.unroll_inner_scan,
            )

            loss = metrics[M.loss].mean()

        elif effective_mode == "pretrain":
            metrics: dict[MetaModel.MetricType, jnp.ndarray] = {}

            seq = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)

            def process_one_window(state, seq_chunk):
                loss, (loss_pure_ce, token_nll_loss, state) = self.lm_loss(seq_chunk, state)
                return state, (loss, loss_pure_ce, token_nll_loss)

            _state, (loss, metrics[M.loss], metrics[M.token_nll_loss]) = scan_remat_chunk(
                process_one_window,
                (state_prefix, state_suffix),
                seq,
                remat_n_loops=cfg.training.inner_remat_freq,
                unroll=cfg.model.unroll_inner_scan,
            )
            loss = loss.mean()

        else:
            raise NotImplementedError(f"Training mode {effective_mode} not implemented")

        # Flatten window into data dimension
        metrics = jax.tree.map(lambda x: x if x.ndim == 1 else rearrange(x, "window data ... -> (window data) ..."), metrics)
        return loss, metrics

    def weights(self):
        """
        Get all weights of the model, including frozen parameters.
        """
        return eqx.filter(self, eqx.is_inexact_array)

    def trainable_parameters(self):
        """
        Get parameters that are trainable, excluding frozen parameters.
        """
        return filter_parameters(self.weights(), self.config.training.spec_outer, "outer parameters")

    def inner_parameters(self):
        """
        Get all inner parameters of the model.
        """
        return filter_parameters(self.weights(), self.config.training.spec_inner, "inner parameters")

    def with_ip_ttt_disabled(self) -> MetaModel:
        """Return a copy with all SwiGLUMLPWithIPTTT layers set to ttt_disabled=True."""
        if not self.config.model.ip_ttt:
            return self
        ff = self.language_model.model.h.blocks.feed_forward
        new_disabled = jnp.ones_like(ff.ttt_disabled)
        return eqx.tree_at(
            lambda m: m.language_model.model.h.blocks.feed_forward.ttt_disabled,
            self,
            new_disabled,
        )


class CausalLM(eqx.Module):
    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)

    model: TransformerModel
    lm_head: NormalLinear | None

    class Output(eqx.Module):
        last_hidden_states: jnp.ndarray
        logits: jnp.ndarray
        new_state: nn.State

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.config = config
        self.compute_dtype = get_float_dtype_by_name(self.config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(self.config.param_dtype)
        key_model, key_word_embeddings = jrandom.split(key, 2)

        self.model = TransformerModel(self.config, key=key_model)

        if not self.config.tie_word_embeddings:
            self.lm_head = NormalLinear(
                self.config,
                in_features=config.hidden_size,
                out_features=config.output_size,
                std=config.initializer_range,
                key=key_word_embeddings,
                name="lm_head",
            )
        else:
            self.lm_head = None

    def wte_call(self, input_ids: jnp.ndarray):
        hidden_states = self.model.wte_call(input_ids)
        return hidden_states

    def prefix_call(self, prefix: Block, hidden_states: jnp.ndarray, state: nn.State, seq: Batch):
        outputs = self.model.prefix_call(prefix, hidden_states, state, seq)
        hidden_states = outputs.last_hidden_state
        assert hidden_states.dtype == self.compute_dtype, "The hidden_states before lm_head should be in compute_dtype"
        return outputs

    def suffix_call(self, prefix_outputs: jnp.ndarray, state: nn.State, seq: Batch):
        outputs = self.model.suffix_call(
            prefix_outputs,
            state,
            seq,
        )
        hidden_states = outputs.last_hidden_state
        assert hidden_states.dtype == self.compute_dtype, "The hidden_states before lm_head should be in compute_dtype"

        if self.config.tie_word_embeddings:
            shared_kernel = self.model.wte.weight.T
            hidden_states, shared_kernel = promote_dtype(hidden_states, shared_kernel, dtype=self.compute_dtype)
            lm_logits = hidden_states @ shared_kernel
        else:
            lm_logits = self.lm_head(hidden_states)

        return CausalLM.Output(last_hidden_states=hidden_states, logits=lm_logits, new_state=outputs.state)

    def wte_disembed_call(self, hidden_states: jnp.ndarray):
        if self.config.tie_word_embeddings:
            shared_kernel = self.model.wte.weight.T
            hidden_states, shared_kernel = promote_dtype(hidden_states, shared_kernel, dtype=self.compute_dtype)
            lm_logits = hidden_states @ shared_kernel
        else:
            lm_logits = self.lm_head(hidden_states)

        return lm_logits

    def __call__(
        self,
        state: nn.State,
        seq: Batch,
    ) -> CausalLM.Output:
        outputs = self.model(
            state,
            seq,
        )
        hidden_states = outputs.last_hidden_state
        assert hidden_states.dtype == self.compute_dtype, "The hidden_states before lm_head should be in compute_dtype"

        if self.config.tie_word_embeddings:
            shared_kernel = self.model.wte.weight.T
            hidden_states, shared_kernel = promote_dtype(hidden_states, shared_kernel, dtype=self.compute_dtype)
            lm_logits = hidden_states @ shared_kernel
        else:
            lm_logits = self.lm_head(hidden_states)

        return CausalLM.Output(last_hidden_states=hidden_states, logits=lm_logits, new_state=outputs.state)
