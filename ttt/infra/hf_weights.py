"""Utilities for loading HuggingFace safetensors weights into MetaModel."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from huggingface_hub import hf_hub_download
from safetensors.flax import load_file
from tqdm import tqdm

from ttt.config import ModelConfig
from ttt.model.transformer import MetaModel

SMOLLM2_135M_MODEL_CONFIG = ModelConfig(
    name="smollm2-135m",
    vocab_size=49152,
    bos_token_id=0,
    eos_token_id=0,
    num_hidden_layers=30,
    hidden_size=576,
    num_attention_heads=9,
    num_key_value_heads=3,
    intermediate_size=1536,
    initializer_range=0.041666666666666664,
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,
    qk_norm=False,
    post_norm=False,
)


def download_safetensors(repo_id: str) -> Path:
    path = hf_hub_download(repo_id, "model.safetensors")
    return Path(path)


def _hf_to_interleaved_qk(arr: jax.Array, num_heads: int, head_dim: int) -> jax.Array:
    """Permute Q/K weights from HF rotate_half RoPE convention to interleaved (complex-number) convention.

    HF pairs (i, i+D/2) per head; this codebase pairs (2i, 2i+1).
    We reorder the output dimension so that the interleaved RoPE
    rotates the same element pairs the HF model was trained with.
    """
    half = head_dim // 2
    perm = jnp.stack([jnp.arange(half), jnp.arange(half, head_dim)], axis=-1).reshape(-1)
    full_perm = jnp.concatenate([perm + h * head_dim for h in range(num_heads)])
    return arr[..., full_perm]


def build_weight_map(hf: dict[str, jax.Array], num_layers: int,
                     num_attention_heads: int | None = None,
                     num_kv_heads: int | None = None) -> dict:
    """
    Map HuggingFace flat key-value weights into the structure expected by MetaModel.weights().

    HF linear weights are stored as (out_features, in_features),
    while NormalLinear stores (in_features, out_features) — so we transpose.
    Per-layer weights are stacked along a new leading axis for vmapped BlockCollection.

    If num_attention_heads / num_kv_heads are provided, the Q and K projection
    weights are permuted from HF's rotate_half RoPE convention to the
    interleaved convention used by this codebase.
    """

    def stack_layers(pattern: str, transpose: bool = False) -> jax.Array:
        arrays = []
        for i in range(num_layers):
            key = pattern.format(i)
            arr = hf[key]
            if transpose:
                arr = arr.T
            arrays.append(arr)
        return jnp.stack(arrays, axis=0)

    wq = stack_layers("model.layers.{}.self_attn.q_proj.weight", transpose=True)
    wk = stack_layers("model.layers.{}.self_attn.k_proj.weight", transpose=True)

    if num_attention_heads is not None:
        head_dim = wq.shape[-1] // num_attention_heads
        wq = _hf_to_interleaved_qk(wq, num_attention_heads, head_dim)
        kv_heads = num_kv_heads if num_kv_heads is not None else num_attention_heads
        wk = _hf_to_interleaved_qk(wk, kv_heads, head_dim)

    return {
        "embed": hf["model.embed_tokens.weight"],
        "ln_f": hf["model.norm.weight"],
        "wq": wq,
        "wk": wk,
        "wv": stack_layers("model.layers.{}.self_attn.v_proj.weight", transpose=True),
        "wo": stack_layers("model.layers.{}.self_attn.o_proj.weight", transpose=True),
        "w1": stack_layers("model.layers.{}.mlp.gate_proj.weight", transpose=True),
        "w2": stack_layers("model.layers.{}.mlp.down_proj.weight", transpose=True),
        "w3": stack_layers("model.layers.{}.mlp.up_proj.weight", transpose=True),
        "seq_norm": stack_layers("model.layers.{}.input_layernorm.weight"),
        "ffn_norm": stack_layers("model.layers.{}.post_attention_layernorm.weight"),
    }


def inject_weights(model: MetaModel, wmap: dict) -> MetaModel:
    """Replace random-init weights in the MetaModel with the HF weights."""
    replacements = [
        (lambda m: m.language_model.model.wte.weight, wmap["embed"]),
        (lambda m: m.language_model.model.ln_f.weight, wmap["ln_f"]),
        (lambda m: m.language_model.model.h.blocks.seq_modeling_block.wq.weight, wmap["wq"]),
        (lambda m: m.language_model.model.h.blocks.seq_modeling_block.wk.weight, wmap["wk"]),
        (lambda m: m.language_model.model.h.blocks.seq_modeling_block.wv.weight, wmap["wv"]),
        (lambda m: m.language_model.model.h.blocks.seq_modeling_block.wo.weight, wmap["wo"]),
        (lambda m: m.language_model.model.h.blocks.feed_forward.w1.weight, wmap["w1"]),
        (lambda m: m.language_model.model.h.blocks.feed_forward.w2.weight, wmap["w2"]),
        (lambda m: m.language_model.model.h.blocks.feed_forward.w3.weight, wmap["w3"]),
        (lambda m: m.language_model.model.h.blocks.seq_norm.weight, wmap["seq_norm"]),
        (lambda m: m.language_model.model.h.blocks.ffn_norm.weight, wmap["ffn_norm"]),
    ]

    for where_fn, value in tqdm(replacements, desc="Injecting weights"):
        model = eqx.tree_at(where_fn, model, value)

    return model


def verify_shapes(model: MetaModel, wmap: dict):
    """Sanity-check that HF weight shapes match the model."""
    checks = [
        ("embed", model.language_model.model.wte.weight),
        ("ln_f", model.language_model.model.ln_f.weight),
        ("wq", model.language_model.model.h.blocks.seq_modeling_block.wq.weight),
        ("wk", model.language_model.model.h.blocks.seq_modeling_block.wk.weight),
        ("wv", model.language_model.model.h.blocks.seq_modeling_block.wv.weight),
        ("wo", model.language_model.model.h.blocks.seq_modeling_block.wo.weight),
        ("w1", model.language_model.model.h.blocks.feed_forward.w1.weight),
        ("w2", model.language_model.model.h.blocks.feed_forward.w2.weight),
        ("w3", model.language_model.model.h.blocks.feed_forward.w3.weight),
        ("seq_norm", model.language_model.model.h.blocks.seq_norm.weight),
        ("ffn_norm", model.language_model.model.h.blocks.ffn_norm.weight),
    ]
    for name, model_param in checks:
        hf_shape = wmap[name].shape
        model_shape = model_param.shape
        assert hf_shape == model_shape, f"Shape mismatch for {name}: HF={hf_shape}, model={model_shape}"


def load_hf_into_model(repo_id: str, model: MetaModel, model_cfg) -> MetaModel:
    """Download HF weights and inject them into an existing MetaModel."""
    safetensors_path = download_safetensors(repo_id)
    hf_weights = load_file(str(safetensors_path))
    wmap = build_weight_map(
        hf_weights,
        num_layers=model_cfg.num_hidden_layers,
        num_attention_heads=model_cfg.num_attention_heads,
        num_kv_heads=model_cfg.num_key_value_heads,
    )
    verify_shapes(model, wmap)
    return inject_weights(model, wmap)
