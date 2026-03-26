"""Modal entrypoint: download HF dataset, then train the E2E TTT model."""
import os
import subprocess
import textwrap
from pathlib import Path

import modal

E2E_DIR = Path(__file__).resolve().parent

app = modal.App("e2e-ttt-train")
hf_cache_volume = modal.Volume.from_name("e2e-hf-cache", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("e2e-checkpoints", create_if_missing=True)
jax_cache_volume = modal.Volume.from_name("e2e-jax-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .add_local_file(str(E2E_DIR / "pyproject.toml"), remote_path="/app/pyproject.toml", copy=True)
    .add_local_file(str(E2E_DIR / "uv.lock"), remote_path="/app/uv.lock", copy=True)
    .run_commands("cd /app && /root/.local/bin/uv sync --exact --no-install-project")
    .add_local_dir(str(E2E_DIR), remote_path="/app", copy=True)
    .run_commands("cd /app && /root/.local/bin/uv sync --exact")
)


@app.function(
    image=image,
    timeout=6 * 3600,
    secrets=[modal.Secret.from_name("default")],
    volumes={"/data": hf_cache_volume},
    cpu=4,
    memory=32768,
)
def download_dataset(hf_dataset: str = "HuggingFaceFW/fineweb-edu", hf_subset: str = "sample-10BT", split: str = "train"):
    """Download HF dataset to the cache volume (CPU only, no GPU)."""
    hf_cache_volume.reload()
    env = os.environ.copy()
    env["HF_HOME"] = "/data"
    script = f"""
from datasets import load_dataset
ds = load_dataset({hf_dataset!r}, {hf_subset!r}, split={split!r}, cache_dir="/data")
print(f"Downloaded {{len(ds)}} rows")
"""
    subprocess.run(
        ["/root/.local/bin/uv", "run", "--exact", "python", "-c", script],
        check=True, cwd="/app", env=env,
    )
    hf_cache_volume.commit()
    print("Dataset cached to volume")


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 3600,
    secrets=[modal.Secret.from_name("default")],
    volumes={"/data": hf_cache_volume, "/checkpoints": checkpoint_volume, "/jax_cache": jax_cache_volume},
)
def train(
    experiment: str,
    wandb_entity: str = "miki-aisle",
    wandb_project: str = "e2e-ttt",
    wandb_name: str = "",
    fast_compile: bool = False,
):
    hf_cache_volume.reload()
    jax_cache_volume.reload()
    checkpoint_volume.reload()

    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
    if fast_compile:
        env["XLA_FLAGS"] = " ".join([
            env.get("XLA_FLAGS", ""),
            "--xla_gpu_autotune_level=0",
            "--xla_gpu_enable_triton_gemm=false",
        ]).strip()

    cmd = [
        "/root/.local/bin/uv", "run", "--exact", "train",
        "+deploy=interactive",
        f"+experiment={experiment}",
        "training.checkpoint_path=/checkpoints",
        f"training.wandb_entity={wandb_entity}",
        f"training.wandb_project={wandb_project}",
        f"training.wandb_key={os.environ['WANDB_API_KEY']}",
        "backend.num_devices=1",
        "backend.compilation_cache_dir=/jax_cache",
        "dataset.hf_cache_dir=/data",
    ]
    if wandb_name:
        cmd.append(f"training.wandb_display_name={wandb_name}")

    subprocess.run(cmd, check=True, cwd="/app", env=env)
    jax_cache_volume.commit()
    checkpoint_volume.commit()


@app.local_entrypoint()
def main(
    experiment: str = "125m/pretrain/simple",
    wandb_entity: str = "miki-aisle",
    wandb_project: str = "e2e-ttt",
    wandb_name: str = "",
    fast_compile: bool = False,
):
    """Download dataset (CPU) then train (GPU)."""
    download_dataset.remote()
    train.remote(
        experiment=experiment,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        fast_compile=fast_compile,
    )
