"""
Test SmolLM2-135M inference using the TTT pipeline.

Downloads HF weights, injects them into a MetaModel, and runs autoregressive
text generation through the real CausalLM forward pass.

Usage (from e2e/):
    pixi run python -m scripts.test_smollm2_inference --prompt "The capital of France is"
    pixi run python -m scripts.test_smollm2_inference --prompt "Once upon a time" --max-tokens 100 --temperature 0.7
"""

import argparse
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
from safetensors.flax import load_file
from transformers import AutoTokenizer

from ttt.infra.hf_weights import SMOLLM2_135M_MODEL_CONFIG, build_weight_map, download_safetensors, inject_weights
from ttt.config import Config
from ttt.model.data import Batch
from ttt.model.transformer import MetaModel

HF_REPO = "HuggingFaceTB/SmolLM2-135M"


def sample_next_token(logits, temperature, key):
    if temperature <= 0:
        return jnp.argmax(logits, axis=-1)
    return jax.random.categorical(key, logits / temperature)


def generate(model, state, token_ids, max_new_tokens, temperature, eos_token_id, key):
    for i in range(max_new_tokens):
        seq = Batch(
            input_ids=token_ids,
            target_tokens=jnp.zeros_like(token_ids),
            loss_masks=jnp.zeros_like(token_ids),
        )
        output = model.language_model(state=state, seq=seq)
        state = output.new_state
        next_logits = output.logits[-1]

        key, subkey = jax.random.split(key)
        next_token = sample_next_token(next_logits, temperature, subkey)

        token_ids = jnp.concatenate([token_ids, next_token[None]])
        sys.stdout.write(f"\r  Generated {i + 1}/{max_new_tokens} tokens")
        sys.stdout.flush()

        if int(next_token) == eos_token_id:
            break

    print()
    return token_ids


def main():
    parser = argparse.ArgumentParser(description="SmolLM2-135M inference via TTT pipeline")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--hf-repo", type=str, default=HF_REPO, help="HuggingFace model repo ID")
    args = parser.parse_args()

    print(f"Downloading {args.hf_repo}...")
    safetensors_path = download_safetensors(args.hf_repo)

    print("Loading safetensors as JAX arrays...")
    hf_weights = load_file(str(safetensors_path))
    print(f"  Loaded {len(hf_weights)} tensors")

    print("Creating model with SmolLM2-135M config...")
    cfg = Config(model=SMOLLM2_135M_MODEL_CONFIG)
    model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=jax.random.PRNGKey(0))

    print("Building weight map and injecting HF weights...")
    wmap = build_weight_map(
        hf_weights, num_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        num_kv_heads=cfg.model.num_key_value_heads,
    )
    model = inject_weights(model, wmap)

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(model.weights()))
    print(f"Model ready — {num_params:,} parameters")

    print(f"Loading tokenizer from {args.hf_repo}...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_repo)

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    token_ids = jnp.array(prompt_ids, dtype=jnp.int32)
    print(f'\nPrompt: "{args.prompt}" ({len(prompt_ids)} tokens)')
    print(f"Generating up to {args.max_tokens} tokens (temperature={args.temperature})...\n")

    key = jax.random.PRNGKey(42)
    output_ids = generate(
        model, state, token_ids,
        args.max_tokens, args.temperature,
        tokenizer.eos_token_id or 0, key,
    )

    output_text = tokenizer.decode(output_ids.tolist(), skip_special_tokens=False)
    print(f"--- Output ---\n{output_text}\n--------------")


if __name__ == "__main__":
    main()
