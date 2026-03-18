"""Modal entrypoint: train the E2E TTT model on a single GPU."""
import os
import subprocess
import textwrap
from pathlib import Path

import modal


def _run_uv(args: list[str]) -> None:
    subprocess.run(["/root/.local/bin/uv", *args], check=True, cwd="/app")


def _debug_probe_script() -> str:
    return textwrap.dedent(
        """
        import importlib.metadata as metadata
        import os
        import platform
        import shutil
        import subprocess
        import sys

        import jax
        import jax.numpy as jnp
        import jaxlib

        packages = [
            "jax",
            "jaxlib",
            "jax-cuda12-plugin",
            "jax-cuda12-pjrt",
            "nvidia-cublas-cu12",
            "nvidia-cuda-cupti-cu12",
            "nvidia-cuda-nvcc-cu12",
            "nvidia-cuda-runtime-cu12",
            "nvidia-cudnn-cu12",
            "nvidia-nccl-cu12",
        ]

        print("== Platform ==")
        print(f"python={sys.version}")
        print(f"platform={platform.platform()}")
        print()

        print("== Package versions ==")
        for package in packages:
            try:
                dist = metadata.distribution(package)
                print(f"{package}={dist.version} @ {dist.locate_file('')}")
            except metadata.PackageNotFoundError:
                print(f"{package}=<not installed>")
        print()

        print("== Environment ==")
        for key in ["CUDA_VISIBLE_DEVICES", "JAX_PLATFORM_NAME", "XLA_FLAGS", "LD_LIBRARY_PATH", "PATH"]:
            print(f"{key}={os.environ.get(key, '')}")
        print()

        print("== Driver / tools ==")
        for cmd in (["nvidia-smi"], ["nvcc", "--version"]):
            tool = shutil.which(cmd[0])
            if tool is None:
                print(f"{cmd[0]}=<not on PATH>")
                continue
            print(f"$ {' '.join(cmd)}")
            try:
                print(subprocess.check_output(cmd, text=True))
            except subprocess.CalledProcessError as exc:
                print(exc.output)
                raise
        print()

        print("== JAX runtime ==")
        print(f"jax={jax.__version__}")
        print(f"jaxlib={jaxlib.__version__}")
        print(f"default_backend={jax.default_backend()}")
        print(f"devices={jax.devices()}")
        print()

        print("== cuDNN attention probe ==")
        batch = 1
        seq_len = 128
        num_heads = 12
        head_dim = 64
        q = jnp.ones((batch, seq_len, num_heads, head_dim), dtype=jnp.bfloat16)
        k = jnp.ones((batch, seq_len, num_heads, head_dim), dtype=jnp.bfloat16)
        v = jnp.ones((batch, seq_len, num_heads, head_dim), dtype=jnp.bfloat16)

        @jax.jit
        def run_attention(q, k, v):
            return jax.nn.dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                implementation="cudnn",
            )

        out = run_attention(q, k, v)
        out.block_until_ready()
        print(f"attention_output_shape={out.shape}")
        print(f"attention_output_dtype={out.dtype}")
        print("cudnn_attention=ok")
        """
    ).strip()

E2E_DIR = Path(__file__).resolve().parent

app = modal.App("e2e-ttt-train")
dataset_volume = modal.Volume.from_name("llama3-dataset-vol")
checkpoint_volume = modal.Volume.from_name("e2e-checkpoints", create_if_missing=True)
cache_volume = modal.Volume.from_name("e2e-jax-cache", create_if_missing=True)

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


GPU_COUNT = 1
@app.function(
    image=image,
    gpu=f"B200:{GPU_COUNT}",
    timeout=6 * 3600,
    secrets=[modal.Secret.from_name("default")],
    volumes={"/data": dataset_volume, "/checkpoints": checkpoint_volume, "/jax_cache": cache_volume},
)
def train(experiment: str, run_name: str, wandb_entity: str = "miki-aisle", wandb_project: str = "e2e-ttt", fast_compile: bool = False):
    dataset_volume.reload()
    cache_volume.reload()

    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
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
        "deploy_paths.data.dclm_filter_8k=/data/data.zarr",
        "deploy_paths.data.books3=/data/books3",
        "training.checkpoint_path=/checkpoints",
        f"training.run_name={run_name}",
        f"training.wandb_entity={wandb_entity}",
        f"training.wandb_project={wandb_project}",
        f"training.wandb_key={os.environ['WANDB_API_KEY']}",
        f"backend.num_devices={GPU_COUNT}",
        "backend.compilation_cache_dir=/jax_cache",
    ]

    subprocess.run(cmd, check=True, cwd="/app", env=env)
    cache_volume.commit()
    checkpoint_volume.commit()


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
):
    assert run_name, "--run-name is required"
    train.remote(experiment=experiment, run_name=run_name, wandb_entity=wandb_entity, wandb_project=wandb_project, fast_compile=fast_compile)


@app.local_entrypoint()
def debug():
    debug_versions.remote()
