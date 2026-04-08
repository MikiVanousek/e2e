"""
Convert HuggingFace SmolLM2-135M (safetensors) to an Orbax checkpoint
compatible with the TTT-E2E training pipeline.

Usage:
    python -m scripts.convert_hf_to_orbax --output-dir ./checkpoints/demo/smollm2-135m-hf-converted
"""

import argparse
from pathlib import Path

import equinox as eqx
import jax
from safetensors.flax import load_file

from ttt.config import Config
from ttt.infra.hf_weights import SMOLLM2_135M_MODEL_CONFIG, build_weight_map, download_safetensors, inject_weights, verify_shapes
from ttt.model.transformer import MetaModel

HF_REPO = "HuggingFaceTB/SmolLM2-135M"


def save_orbax(model: MetaModel, output_dir: str):
    import orbax.checkpoint as ocp

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    handler_registry = ocp.DefaultCheckpointHandlerRegistry()
    handler_registry.add("model_weights", ocp.args.StandardRestore, ocp.StandardCheckpointHandler)
    handler_registry.add("model_weights", ocp.args.StandardSave, ocp.StandardCheckpointHandler)

    manager = ocp.CheckpointManager(
        output_path,
        handler_registry=handler_registry,
    )

    model_weights = model.weights()
    manager.save(
        step=0,
        args=ocp.args.Composite(
            model_weights=ocp.args.StandardSave(model_weights),
        ),
        force=True,
    )
    manager.wait_until_finished()
    manager.close()
    print(f"Checkpoint saved to {output_path}/0/")


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace SmolLM2-135M to Orbax checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for the Orbax checkpoint")
    parser.add_argument("--hf-repo", type=str, default=HF_REPO, help="HuggingFace model repo ID")
    args = parser.parse_args()

    print(f"Downloading {args.hf_repo}...")
    safetensors_path = download_safetensors(args.hf_repo)

    print(f"Loading safetensors from {safetensors_path}...")
    hf_weights = load_file(str(safetensors_path))
    print(f"  Loaded {len(hf_weights)} tensors")

    print("Creating model with SmolLM2-135M config...")
    cfg = Config(model=SMOLLM2_135M_MODEL_CONFIG)
    model, _state = eqx.nn.make_with_state(MetaModel)(cfg, key=jax.random.PRNGKey(0))

    print("Building weight map...")
    wmap = build_weight_map(
        hf_weights, num_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        num_kv_heads=cfg.model.num_key_value_heads,
    )

    verify_shapes(model, wmap)

    print("Injecting HF weights into model...")
    model = inject_weights(model, wmap)

    print("Saving Orbax checkpoint...")
    save_orbax(model, args.output_dir)

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(model.weights()))
    print(f"Done. Total parameters: {num_params:,}")


if __name__ == "__main__":
    main()
