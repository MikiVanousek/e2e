"""Modal entrypoint: download HF dataset, then train the E2E TTT model."""
import os
import shlex
import subprocess
import textwrap
from pathlib import Path

import modal

E2E_DIR = Path(__file__).resolve().parent

app = modal.App("e2e-ttt-train")
hf_cache_volume = modal.Volume.from_name("e2e-hf-cache", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("e2e-checkpoints", create_if_missing=True)
jax_cache_volume = modal.Volume.from_name("e2e-jax-cache", create_if_missing=True)
ruler_volume = modal.Volume.from_name("e2e-ruler", create_if_missing=True)
kv_recall_volume = modal.Volume.from_name("e2e-kv-recall", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("curl", "git", "wget")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .add_local_file(str(E2E_DIR / "pyproject.toml"), remote_path="/app/pyproject.toml", copy=True)
    .add_local_file(str(E2E_DIR / "uv.lock"), remote_path="/app/uv.lock", copy=True)
    .run_commands("cd /app && /root/.local/bin/uv sync --exact --no-install-project")
    .add_local_dir(str(E2E_DIR), remote_path="/app", copy=True)
    .run_commands("cd /app && /root/.local/bin/uv sync --exact")
    .run_commands("cd /app && /root/.local/bin/uv pip install beautifulsoup4 html2text nltk pandas tenacity wonderwords")
)


@app.function(
    image=image,
    timeout=6 * 3600,
    secrets=[modal.Secret.from_name("default")],
    volumes={"/data": hf_cache_volume},
    cpu=64,
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


def _run_train(
    experiment: str,
    wandb_entity: str = "miki-aisle",
    wandb_project: str | None = None,
    extra_args: str = "",
    fast_compile: bool = False,
    resume: bool = False,
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
        f"training.wandb_key={os.environ['WANDB_API_KEY']}",
        "backend.num_devices=1",
        "backend.compilation_cache_dir=/jax_cache",
        "dataset.hf_cache_dir=/data",
    ]
    if wandb_project is not None:
        cmd.append(f"training.wandb_project={wandb_project}")
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    if resume:
        cmd.extend([
            "training.load_part=all",
            f"training.resume_exp_name={Path(experiment).name}",
        ])
    subprocess.run(cmd, check=True, cwd="/app", env=env)
    jax_cache_volume.commit()
    checkpoint_volume.commit()


@app.function(
    image=image,
    gpu="H200",
    timeout=24 * 3600,
    secrets=[modal.Secret.from_name("default")],
    volumes={"/data": hf_cache_volume, "/checkpoints": checkpoint_volume, "/jax_cache": jax_cache_volume},
)
def train(
    experiment: str,
    wandb_entity: str = "miki-aisle",
    wandb_project: str | None = None,
    extra_args: str = "",
    fast_compile: bool = False,
    resume: bool = False,
):
    _run_train(experiment, wandb_entity, wandb_project, extra_args, fast_compile, resume)


@app.function(
    image=image,
    gpu="B200",
    timeout=24 * 3600,
    secrets=[modal.Secret.from_name("default")],
    volumes={"/data": hf_cache_volume, "/checkpoints": checkpoint_volume, "/jax_cache": jax_cache_volume},
)
def train_b200(
    experiment: str,
    wandb_entity: str = "miki-aisle",
    wandb_project: str | None = None,
    extra_args: str = "",
    fast_compile: bool = False,
    resume: bool = False,
):
    _run_train(experiment, wandb_entity, wandb_project, extra_args, fast_compile, resume)


@app.function(
    image=image,
    gpu="H200",
    timeout=24 * 3600,
    secrets=[modal.Secret.from_name("default")],
    volumes={
        "/data": hf_cache_volume,
        "/checkpoints": checkpoint_volume,
        "/jax_cache": jax_cache_volume,
        "/ruler": ruler_volume,
    },
)
def ruler(
    experiment: str,
    checkpoint_exp_name: str | None = None,
    tasks: str = "niah_single_1,vt",
    num_samples: int = 1,
    max_seq_length: int = 131072,
    tokens_to_generate_limit: int = 0,
    output_root: str = "/ruler/results",
    e2e_ttt_off: bool = False,
    tokenizer_name: str = "meta-llama/Llama-3.1-8B",
    download_aux_data: bool = False,
    validate_e2e_only: bool = False,
    wandb_entity: str = "miki-aisle",
    wandb_project: str = "thesis-125m",
):
    hf_cache_volume.reload()
    jax_cache_volume.reload()
    checkpoint_volume.reload()
    ruler_volume.reload()

    env = os.environ.copy()
    env["HF_HOME"] = "/data"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

    exp_name = checkpoint_exp_name or Path(experiment).name
    cmd = [
        "/app/.venv/bin/python", "-m", "ttt.ruler",
        "--experiment", experiment,
        "--checkpoint-exp-name", exp_name,
        "--checkpoint-path", "/checkpoints",
        "--output-root", output_root,
        "--ruler-dir", "/ruler/NVIDIA_RULER",
        "--tasks", tasks,
        "--num-samples", str(num_samples),
        "--max-seq-length", str(max_seq_length),
        "--tokenizer-name", tokenizer_name,
        "--num-devices", "1",
        "--jax-cache-dir", "/jax_cache",
        "--wandb-entity", wandb_entity,
        "--wandb-project", wandb_project,
        "--wandb-key", os.environ.get("WANDB_API_KEY", ""),
    ]
    if tokens_to_generate_limit > 0:
        cmd.extend(["--tokens-to-generate-limit", str(tokens_to_generate_limit)])
    if e2e_ttt_off:
        cmd.append("--e2e-ttt-off")
    if validate_e2e_only:
        cmd.append("--validate-e2e-only")
    if not download_aux_data:
        cmd.append("--no-download-aux-data")

    subprocess.run(cmd, check=True, cwd="/app", env=env)
    jax_cache_volume.commit()
    ruler_volume.commit()


@app.function(
    image=image,
    gpu="H200",
    timeout=24 * 3600,
    secrets=[modal.Secret.from_name("default")],
    volumes={
        "/checkpoints": checkpoint_volume,
        "/jax_cache": jax_cache_volume,
        "/kv_recall": kv_recall_volume,
    },
)
def kv_recall(
    labels: str = "fa,swa",
    max_docs: int = 4,
    output_dir: str = "/kv_recall/latest",
):
    checkpoint_volume.reload()
    jax_cache_volume.reload()
    kv_recall_volume.reload()

    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
    cmd = [
        "/root/.local/bin/uv",
        "run",
        "--exact",
        "python",
        "-m",
        "ttt.eval_kv_recall",
        "--labels",
        labels,
        "--max-docs",
        str(max_docs),
        "--checkpoint-root",
        "/checkpoints",
        "--output-dir",
        output_dir,
        "--jax-cache-dir",
        "/jax_cache",
    ]
    subprocess.run(cmd, check=True, cwd="/app", env=env)
    jax_cache_volume.commit()
    kv_recall_volume.commit()


@app.function(
    image=image,
    timeout=6 * 3600,
    secrets=[modal.Secret.from_name("default")],
    volumes={"/data": hf_cache_volume},
    cpu=32,
    memory=32768,
)
def preprocess_dataset(experiment: str, extra_args: str = ""):
    """Download, tokenize, and filter dataset (CPU only, no GPU)."""
    hf_cache_volume.reload()
    cmd = [
        "/root/.local/bin/uv", "run", "--exact", "preprocess",
        "+deploy=interactive",
        f"+experiment={experiment}",
        "dataset.hf_cache_dir=/data",
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    subprocess.run(cmd, check=True, cwd="/app", env=os.environ.copy())
    hf_cache_volume.commit()


@app.local_entrypoint()
def preprocess_main(
    experiment: str = "125m/extension/ext-16K-125m-ip-ttt-next-current-layer",
    extra_args: str = "",
    wait: bool = True,
):
    """Preprocess one or more comma-separated experiments on CPU."""
    handles = [
        preprocess_dataset.spawn(
            experiment=exp.strip(),
            extra_args=extra_args,
        )
        for exp in experiment.split(",")
    ]
    if not wait:
        print(f"Spawned {len(handles)} preprocessing job(s).")
        return
    for h in handles:
        h.get()


@app.local_entrypoint()
def ruler_main(
    experiment: str = "125m/extension/ext-128K-125m-fa,125m/extension/ext-128K-125m-swa",
    tasks: str = "niah_single_1,vt",
    num_samples: int = 1,
    max_seq_length: int = 131072,
    tokens_to_generate_limit: int = 0,
    wait: bool = True,
):
    """Run RULER for one or more comma-separated experiments."""
    handles = [
        ruler.spawn(
            experiment=exp.strip(),
            tasks=tasks,
            num_samples=num_samples,
            max_seq_length=max_seq_length,
            tokens_to_generate_limit=tokens_to_generate_limit,
        )
        for exp in experiment.split(",")
    ]
    if not wait:
        print(f"Spawned {len(handles)} RULER job(s).")
        return
    for h in handles:
        h.get()


@app.local_entrypoint()
def kv_recall_main(
    labels: str = "fa,swa",
    max_docs: int = 4,
    output_dir: str = "/kv_recall/latest",
    wait: bool = True,
):
    """Run synthetic KV recall eval for selected comma-separated labels."""
    handle = kv_recall.spawn(labels=labels, max_docs=max_docs, output_dir=output_dir)
    if not wait:
        print("Spawned KV recall eval job.")
        return
    handle.get()


@app.local_entrypoint()
def main(
    experiment: str = "125m/pretrain/simple",
    wandb_entity: str = "miki-aisle",
    wandb_project: str | None = None,
    extra_args: str = "",
    fast_compile: bool = False,
    resume: bool = False,
    wait: bool = True,
):
    """Train one or more experiments in parallel. `experiment` accepts a comma-separated list."""
    # download_dataset.remote()
    handles = []
    for exp in experiment.split(","):
        exp = exp.strip()
        train_fn = train_b200 if "e2e" in Path(exp).name else train
        handles.append(train_fn.spawn(
            experiment=exp,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            extra_args=extra_args,
            fast_compile=fast_compile,
            resume=resume,
        ))
    if not wait:
        print(f"Spawned {len(handles)} training job(s).")
        return
    for h in handles:
        h.get()
