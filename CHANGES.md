# Changes from upstream `test-time-training/e2e`

Fork of [test-time-training/e2e](https://github.com/test-time-training/e2e) at commit `a4fc478` (Merge pull request #5 — release checkpoints).
All changes below are on top of that base.

---

## 1. Modal Cloud Training Infrastructure (`modal_train.py`)

The upstream repo assumed a Slurm/submitit cluster with local GPUs and GCS data.
A complete Modal-based training pipeline was built from scratch:

- **Docker image**: switched from `debian_slim` to `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` for reliable cuDNN support. Dependencies installed in two stages (lock file first, then full project) for faster image rebuilds.
- **GPU targeting**: iterated through H100 → B200 → H200, settled on H100 for cost/availability.
- **Volume management**: Modal volumes for HF dataset cache (`e2e-hf-cache`), checkpoints (`e2e-checkpoints`), and XLA compilation cache (`e2e-jax-cache`). Earlier iterations also had a `llama3-dataset-vol` for the original Zarr data and `e2e-chunks` for preprocessed chunks (both later removed).
- **Dataset download function**: CPU-only Modal function `download_dataset` that pre-downloads HuggingFace datasets to the cache volume before GPU training starts.
- **XLA flags**: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, optional `--xla_gpu_autotune_level=0` for faster compilation, and `--xla_gpu_enable_latency_hiding_scheduler=false` for single-GPU runs.
- **Entry point**: `main()` sequentially calls `download_dataset` then `train`, passing experiment config, wandb credentials, and optional `wandb_project` override.
- **Removed**: the original `modal_download_dataset.py` (GCS-based), the debug probe script, and the zarr chunk preprocessing pipeline that were intermediate solutions.

## 2. Dataset Pipeline Rewrite (`ttt/dataloader/lm_dataset.py`)

The upstream data loader read from a pre-tokenized Zarr array on GCS. This was replaced entirely:

- **`HFTokenizedDataset`**: New `grain.RandomAccessDataSource` that loads any HuggingFace dataset, tokenizes it with `AutoTokenizer`, and filters by minimum length — all cached by HF `datasets` Arrow cache so only the first run pays the cost.
- **Truncated tokenizer support**: when `vocab_size < tokenizer.vocab_size`, builds a truncated tokenizer via `tokenizers` library (see `retokenizer.py`) and applies it during the map step.
- **Epoch accounting**: prints `epochs` count and asserts `<= 5` to prevent accidental overtraining.
- **Removed**: `Dataset` (Zarr-based), `ChunkedDataset` (.npy chunk-based), `save_chunk()`, `chunks_volume_key()`, `Retokenizer` class, and all `zarr`/`zarr.codecs` imports.
- **Config-driven**: dataset params (`hf_dataset`, `hf_subset`, `hf_text_column`, `hf_cache_dir`) moved to a new `DatasetConfig` dataclass with a default YAML (`configs/dataset/fineweb-edu-10bt.yaml`).

## 3. Configuration System Overhaul (`ttt/config.py`, `configs/`)

### New config dataclasses
- **`DatasetConfig`**: `hf_dataset`, `hf_subset`, `hf_text_column`, `hf_cache_dir`. Registered as a Hydra config group with default `fineweb-edu-10bt`.

### TrainingConfig changes
- **Added**: `run_name`, `wandb_display_name` (later simplified to just `exp_name`), `wandb_tags`, `skip_tokens`, `num_evals`, `max_eval_batches`, `chunks_dir`, `preprocess_chunk_idx`, `eval_batch_size`.
- **Changed defaults**: `tokenizer_name` from `meta-llama/Llama-2-7b-hf` → `meta-llama/Llama-3.1-8B`, `data_split` from `train` → `train[5%:]`, `eval_split` from `val` → `train[:5%]`, `num_evals` from `1` → `6`.
- **Removed**: `dataset_path`, `dataset_name` (as MISSING required), `save_milestone_freq`, the `DeployPathsConfig.Data` sub-dataclass (no more GCS paths).

### ModelConfig changes
- **Added**: `num_key_value_heads` (for GQA/MQA support), `prime_zero_init` (zero-initialize the prime FFN output).

### Experiment configs
- **New model definitions**: `14m.yaml` (6-layer, 384-hidden, 8192 vocab), `toy_small.yaml` (2-layer debug), `smollm2-135m.yaml` (for SmolLM2-135M with GQA).
- **14m experiments** (created and iterated on):
  - `pretrain-14m-e2e.yaml` — E2E TTT pretraining at 14M scale
  - `pretrain-14m-fa.yaml` / `pretrain-14m-fa-match-long.yaml` — Full-attention baselines
  - `pretrain-14m-e2e-from-fa.yaml` — E2E TTT initialized from a full-attention checkpoint
  - `pretrain-14m-e2e-long.yaml` / `pretrain-14m-e2e-longest.yaml` — Extended training schedules
  - `ext-14m-e2e-32K.yaml` / `ext-14m-fa-32K.yaml` — 32K context-length extension fine-tuning
- **125m experiments**: added `accum_steps: 8` to `pretrain-125m-e2e.yaml`, `wandb_project`/`wandb_tags`.
- **SmolLM2-135M**: `pretrain-smollm2-135m-e2e-from-hf.yaml` — E2E TTT starting from HF pretrained weights.
- **Training schedules**: new `14m/pretrain-8K.yaml` (Chinchilla-scaled: 280M tokens, 534 steps) and `14m/ext.yaml`.
- **Deploy configs**: removed `data.dclm_filter_8k` and `data.books3` path requirements from `interactive.yaml` and `submitit.yaml`.

## 4. W&B Logging Rewrite (`ttt/infra/wandb_utils.py`)

- **Simplified run management**: every launch creates a new W&B run instead of trying to find/resume existing runs by name. Runs sharing the same `exp_name` are grouped via W&B's `group` field.
- **Display name**: auto-generated as `{exp_name}-{YYYYMMDD-HHMM}`.
- **Tags support**: `wandb_tags` list passed through to `wandb.init()`.
- **Context manager**: `WandbLogger` supports `__enter__`/`__exit__` to auto-finish runs with proper exit codes (added then later removed in favor of explicit cleanup).
- **Auth fix**: `wandb.login(key=...)` replaced with `os.environ["WANDB_API_KEY"] = ...` to fix authentication issues.

## 5. Evaluation System (`ttt/model/loop.py`)

- **Configurable eval count**: `num_evals` config controls how many evaluations are run during training (evenly spaced).
- **Max eval batches**: `max_eval_batches` caps evaluation length for faster iteration.
- **95% confidence intervals**: evaluation now computes and logs CI for loss.
- **Eval batch size**: separate `eval_batch_size` config (defaults to larger than train batch for speed).
- **Bits-per-byte metric**: added `bits_per_byte` evaluation using UTF-8 byte counting with `tiktoken` encoder, providing a tokenizer-independent perplexity metric.
- **Gradient norm decomposition**: when using prime layers, separately tracks and logs `prime_grad_norm` and `pretrained_grad_norm` to monitor how much the gradient comes from prime vs pretrained parameters.

## 6. Checkpoint System (`ttt/infra/checkpoint.py`)

- **Shape mismatch errors**: improved error message for checkpoint shape mismatches to suggest checking `intermediate_size`.
- **Duplicate save guard**: `save_checkpoint` now skips if the step already exists in the manager.
- **Type annotation fix**: `unify_dict_with_eqx_module` changed from PEP 695 syntax to standard typing for broader compatibility.
- **Eval-mode optimization**: skips restoring optimizer and loop state when `eval_mode=True`.

## 7. Training Loop (`ttt/train.py`)

- **Data skipping**: `skip_tokens` config allows resuming training past already-seen data (for E2E-from-FA experiments). Implemented as dataset slicing rather than iterating through skipped batches.
- **XLA compilation cache**: enabled persistent cache with `min_entry_size_bytes=0` and `min_compile_time_secs=0` for aggressive caching.
- **Memory breakdown logging**: `log_memory_breakdown()` reports model weights, optimizer state, and activation peak memory at start and after first step.
- **Not-found parameter logging**: when loading a checkpoint with mismatched architecture, logs which parameters were initialized from scratch.
- **Removed**: `_BatchPrefetcher` class (replaced by grain's built-in prefetch), timing instrumentation, JAX profiler trace upload, and the `_resolve_chunks_dir` helper.

## 8. Model Architecture (`ttt/model/`)

### Attention (`ttt/model/attention.py`)
- **Grouped Query Attention (GQA)**: added `num_key_value_heads` support with KV head repetition (`_repeat_kv`), enabling models like SmolLM2-135M that use fewer KV heads than query heads.

### Transformer (`ttt/model/transformer.py`)
- **Prime zero-init**: optional `prime_zero_init` that zero-initializes the `w2` weight of the prime FFN, so the prime contribution starts at zero.
- **New metrics**: `prime_grad_norm` and `pretrained_grad_norm` in `MetricType` enum.

### Data (`ttt/model/data.py`)
- Minor dtype change in `Batch`.

## 9. HuggingFace Weight Loading (`ttt/infra/hf_weights.py`, `scripts/`)

Entirely new infrastructure to load pretrained HF models into the E2E TTT framework:

- **`hf_weights.py`**: Downloads safetensors from HuggingFace Hub, builds a weight map that translates HF parameter names to TTT model paths, verifies shapes, and injects weights via `eqx.tree_at`. Includes `SMOLLM2_135M_MODEL_CONFIG` constant.
- **`convert_hf_to_orbax.py`**: Script to convert HF safetensors to Orbax checkpoint format.
- **`test_smollm2_inference.py`**: End-to-end inference test — downloads SmolLM2-135M, injects into MetaModel, and runs autoregressive generation.

## 10. Dependencies (`pyproject.toml`)

- **Added**: `transformers>=4.40`, `datasets`, `safetensors`, `tiktoken`, `tokenizers`.
- **New script entry**: `preprocess = "ttt.train:preprocess_main"` (for the chunked preprocessing pipeline, later simplified).

## 11. Utility Additions

- **`ttt/utils/memory_utils.py`**: GPU memory introspection — reports model, optimizer, activation, and peak memory usage. Logs to both console and W&B.
- **`ttt/dataloader/retokenizer.py`**: Vocabulary truncation via `tokenizers` library for training with smaller vocab sizes than the base tokenizer.
- **`ttt/utils/jax_utils.py`**: Added `safe_sqrt` helper for gradient norm decomposition.
