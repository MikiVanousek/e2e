"""Modal entrypoint: train the E2E TTT model on a single GPU."""
import os
import subprocess
import textwrap
import threading
from pathlib import Path

import modal


def _run_uv(args: list[str]) -> None:
    subprocess.run(["/root/.local/bin/uv", *args], check=True, cwd="/app")

E2E_DIR = Path(__file__).resolve().parent

app = modal.App("e2e-ttt-train")
dataset_volume = modal.Volume.from_name("llama3-dataset-vol")
checkpoint_volume = modal.Volume.from_name("e2e-checkpoints", create_if_missing=True)
cache_volume = modal.Volume.from_name("e2e-jax-cache", create_if_missing=True)
chunks_volume = modal.Volume.from_name("e2e-chunks", create_if_missing=True)

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


def _preprocess_base_cmd(experiment: str) -> list[str]:
    return [
        "/root/.local/bin/uv", "run", "--exact", "preprocess",
        "+deploy=interactive",
        f"+experiment={experiment}",
        "deploy_paths.data.dclm_filter_8k=/data/data.zarr",
        "deploy_paths.data.books3=/data/books3",
        "training.chunks_dir=/chunks",
    ]


def _preprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    return env


@app.function(
    image=image,
    timeout=12 * 3600,
    secrets=[modal.Secret.from_name("default")],
    volumes={"/data": dataset_volume, "/chunks": chunks_volume},
    cpu=4,
    memory=32768,
)
def preprocess_single_chunk(experiment: str, chunk_idx: int):
    dataset_volume.reload()
    chunks_volume.reload()

    cmd = [*_preprocess_base_cmd(experiment), f"training.preprocess_chunk_idx={chunk_idx}"]
    result = subprocess.run(cmd, cwd="/app", env=_preprocess_env())
    if result.returncode != 0:
        raise RuntimeError(f"Preprocessing chunk {chunk_idx} failed with exit code {result.returncode}")
    chunks_volume.commit()
    print(f"Committed chunk {chunk_idx}")


@app.function(
    image=image,
    timeout=12 * 3600,
    secrets=[modal.Secret.from_name("default")],
    volumes={"/data": dataset_volume, "/chunks": chunks_volume},
    cpu=4,
    memory=32768,
)
def preprocess_data(experiment: str):
    dataset_volume.reload()

    cmd = [*_preprocess_base_cmd(experiment), "training.preprocess_chunk_idx=-2"]
    result = subprocess.run(cmd, cwd="/app", env=_preprocess_env(), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Count chunks failed: {result.stderr}")
    for line in result.stdout.splitlines():
        if line.startswith("CHUNK_COUNT="):
            num_chunks = int(line.split("=", 1)[1])
            break
    else:
        raise RuntimeError(f"Could not parse chunk count from output:\n{result.stdout}")

    print(f"Dispatching {num_chunks} chunk(s) in parallel")
    for _ in preprocess_single_chunk.starmap([(experiment, i) for i in range(num_chunks)]):
        pass
    print(f"All {num_chunks} chunks done")


GPU_COUNT = 1


@app.function(
    image=image,
    gpu=f"H200:{GPU_COUNT}",
    timeout=12 * 3600,
    secrets=[modal.Secret.from_name("default")],
    volumes={"/data": dataset_volume, "/checkpoints": checkpoint_volume, "/jax_cache": cache_volume, "/chunks": chunks_volume},
)
def train(experiment: str, run_name: str, wandb_entity: str = "miki-aisle", wandb_project: str = "e2e-ttt", fast_compile: bool = False, eval_only: bool = False, mem_profile: bool = False):
    dataset_volume.reload()
    cache_volume.reload()
    checkpoint_volume.reload()
    chunks_volume.reload()

    env = os.environ.copy()
    if mem_profile:
        env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    else:
        env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
    xla_extra = []
    if fast_compile:
        xla_extra += [
            "--xla_gpu_autotune_level=0",
            "--xla_gpu_enable_triton_gemm=false",
        ]
    if GPU_COUNT == 1:
        xla_extra.append("--xla_gpu_enable_latency_hiding_scheduler=false")
    if xla_extra:
        env["XLA_FLAGS"] = " ".join([env.get("XLA_FLAGS", ""), *xla_extra]).strip()

    cmd = [
        "/root/.local/bin/uv", "run", "--exact", "train",
        "+deploy=interactive",
        f"+experiment={experiment}",
        "deploy_paths.data.dclm_filter_8k=/data/data.zarr",
        "deploy_paths.data.books3=/data/books3",
        "training.checkpoint_path=/checkpoints",
        f"training.run_name={run_name}",
        f"training.wandb_entity={wandb_entity}",
        f"training.wandb_project={wandb_project}",
        f"training.wandb_key={os.environ['WANDB_API_KEY']}",
        f"backend.num_devices={GPU_COUNT}",
        "backend.compilation_cache_dir=/jax_cache",
        "training.chunks_dir=/chunks",
    ]
    if eval_only:
        cmd.append("training.eval_mode=True")

    stop_commit = threading.Event()

    def _periodic_commit():
        while not stop_commit.wait(timeout=120):
            checkpoint_volume.commit()

    commit_thread = threading.Thread(target=_periodic_commit, daemon=True)
    commit_thread.start()
    try:
        subprocess.run(cmd, check=True, cwd="/app", env=env)
    finally:
        stop_commit.set()
        commit_thread.join()
        checkpoint_volume.commit()
    cache_volume.commit()


@app.function(
    image=image,
    gpu="H100",
    timeout=30 * 60,
)
def debug_versions():
    _run_uv(["run", "--exact", "python", "-c", _debug_probe_script()])


@app.local_entrypoint()
def main(
    experiment: str = "125m/pretrain/simple",
    run_name: str = "",
    wandb_entity: str = "miki-aisle",
    wandb_project: str = "e2e-ttt",
    fast_compile: bool = False,
    eval_only: bool = False,
    mem_profile: bool = False,
):
    """Preprocess data if needed (CPU, no GPU), then launch GPU training."""
    assert run_name, "--run-name is required"
    preprocess_data.remote(experiment=experiment)
    train.remote(
        experiment=experiment, run_name=run_name, wandb_entity=wandb_entity,
        wandb_project=wandb_project, fast_compile=fast_compile, eval_only=eval_only,
        mem_profile=mem_profile,
    )


# @app.local_entrypoint()
# def debug():
#     debug_versions.remote()
