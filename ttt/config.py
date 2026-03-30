from dataclasses import dataclass, field
from enum import StrEnum

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass(unsafe_hash=True, eq=True)
class JaxDistributedConfig:
    distributed: bool = False
    coordinator_address: str | None = None
    num_processes: int | None = None
    process_id: int | None = None
    local_device_ids: str | None = None
    backend: str | None = None  # cpu, gpu, tpu
    num_devices: int | None = None
    compilation_cache_dir: str | None = "/tmp/jax_cache"


@dataclass(unsafe_hash=True, eq=True)
class CheckpointConfig:
    float_dtype: str = "bf16"
    save_optimizer_state: bool = True
    checkpoint_dir: str = MISSING
    resume_checkpoint_dir: str = MISSING


class OptimizerType(StrEnum):
    adamw = "adamw"
    sgd = "sgd"


@dataclass(unsafe_hash=True, eq=True)
class OptimizerConfig:
    optimizer_type: OptimizerType = MISSING  # adamw, sgd
    init_lr: float = MISSING
    end_lr: float = MISSING
    lr: float = MISSING
    lr_warmup_steps: int = MISSING
    lr_decay_steps: int = MISSING
    b1: float = MISSING
    b2: float = MISSING
    clip_gradient: float = MISSING
    weight_decay: float = MISSING
    bf16_momentum: bool = MISSING


@dataclass(unsafe_hash=True, eq=True)
class AdamWOptimizerConfig(OptimizerConfig):
    optimizer_type: OptimizerType = "adamw"
    init_lr: float = 0.0
    end_lr: float = 1e-5
    lr: float = 0.01
    lr_warmup_steps: int = 2000
    lr_decay_steps: int = 500000
    b1: float = 0.9
    b2: float = 0.95
    clip_gradient: float = 1.0
    weight_decay: float = 0.1
    bf16_momentum: bool = False
    emb_wd: bool = True


@dataclass(unsafe_hash=True, eq=True)
class SGDOptimizerConfig(OptimizerConfig):
    optimizer_type: OptimizerType = "sgd"
    lr: float = 0.01
    clip_gradient: float = 0.0


@dataclass(unsafe_hash=True, eq=True)
class ModelConfig:
    class SeqModelingBlockType(StrEnum):
        self_attention = "self_attention"
        SWA = "SWA"

    name: str = "unnamed"
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mini_batch_size: int = 1024
    sliding_window_size: int = 1024
    seq_len: int = 131072
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    bos_token_id: int = 1
    eos_token_id: int = 2
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    tie_word_embeddings: bool = False
    remat_block: str = ""
    remat_block_bwd: str = ""
    remat_prefix_block: str = "nothing_saveable"
    remat_attention: str = ""
    remat_attention_bwd: str = ""
    remat_mlp: str = ""
    remat_mlp_bwd: str = ""
    remat_rms: str = ""
    remat_rms_bwd: str = ""
    remat_multiple_gd: str = ""
    seq_modeling_block: str = "self_attention"
    rope_theta: float = 10000.0
    output_size: int = vocab_size
    compute_dtype: str = "bf16"
    param_dtype: str = "fp32"
    state_dtype: str = "fp32"
    unroll_block_scan: bool = False
    unroll_inner_scan: bool = False
    force_flash: bool = False
    suffix_len: int = 0
    prime: bool = False
    qk_norm: bool = True
    pre_norm: bool = True
    post_norm: bool = True
    feed_forward_prime: str = "swiglu"  # Only "swiglu" is supported.


@dataclass(unsafe_hash=True, eq=True)
class DatasetConfig:
    hf_dataset: str = "HuggingFaceFW/fineweb-edu"
    hf_subset: str = "sample-10BT"
    hf_text_column: str = "text"
    hf_cache_dir: str = ""


@dataclass(unsafe_hash=True, eq=True)
class TrainingConfig:
    class LoadPart(StrEnum):
        all = "all"
        params = "params"
        none = "none"

    class TrainMode(StrEnum):
        pretrain = "pretrain"
        meta = "meta"

    log_wandb: bool = True
    wandb_entity: str = MISSING
    wandb_project: str = MISSING
    wandb_key: str = MISSING
    model_seed: int = 0
    data_seed: int = 0
    load_part: LoadPart = LoadPart.none  # params, all, none
    total_steps: int = 2500
    break_step: int = -1
    save_milestone_freq: int = 2500
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf"
    dataset_name: str = ""
    seq_length: int = 1024
    global_batch_size: int = 8
    accum_steps: int = 1
    loader_workers: int = 32
    dummy_dataset: bool = False
    checkpoint_path: str = "./checkpoints"
    exp_dir: str = "./experiments"
    exp_folder: str = "demo"
    exp_name: str = MISSING
    run_name: str = MISSING
    wandb_display_name: str = ""
    resume_exp_name: str = ""
    resume_step: int | None = None
    eval_mode: bool = False
    train_mode: TrainMode = "pretrain"
    data_split: str = "train[5%:]"
    eval_split: str = "train[:5%]"
    inner_remat_freq: int = 1
    optimizer_outer: OptimizerConfig = field(default_factory=AdamWOptimizerConfig)
    optimizer_inner: OptimizerConfig | None = field(default_factory=SGDOptimizerConfig)
    spec_outer: list[str] = field(default_factory=lambda: ["**"])
    spec_inner: list[str] = field(default_factory=lambda: ["**"])
    "Specs are a list of dot-expressions with globs to index into the model. For instance, `['language_model.*.weight', 'language_model.**.bias]` would match every weight in every direct submodule of the language_model, and every bias parameter in the entire language_model."
    n_data_parallel: int | None = None  # Default to num_devices / n_state_parallel
    n_state_parallel: int = 1
    ilr_warmup_steps: int = 0
    ilr_init: float = 1.0
    eval_batch_size: int = 8
    max_eval_batches: int = 150
    num_evals: int = 1
    skip_tokens: int = 0
    chunks_dir: str = ""
    preprocess_chunk_idx: int = -1


@dataclass(unsafe_hash=True, eq=True)
class DeployPathsConfig:
    checkpoint: str = MISSING


@dataclass(unsafe_hash=True, eq=True)
class Config:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backend: JaxDistributedConfig = field(default_factory=JaxDistributedConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    deploy_paths: DeployPathsConfig = field(default_factory=DeployPathsConfig)


def register_configs():
    cs = ConfigStore.instance()
    cs.store(group="training", name="base_training", node=TrainingConfig)
    cs.store(group="dataset", name="base_dataset", node=DatasetConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="backend", name="base_backend", node=JaxDistributedConfig)
    cs.store(group="checkpoint", name="base_checkpoint", node=CheckpointConfig)
    cs.store(group="deploy_paths", name="base_deploy_paths", node=DeployPathsConfig)
    cs.store(name="base_config", node=Config)
