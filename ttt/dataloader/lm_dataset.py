import grain.python as grain
import jax
import numpy as np

from ttt.model.data import Batch


def _doc_indices(n_docs: int, split: str, eval_fraction: float) -> range:
    eval_count = int(round(n_docs * eval_fraction))
    if eval_fraction > 0 and n_docs > 0:
        eval_count = max(1, eval_count)

    if split == "eval":
        return range(0, eval_count)
    if split == "train":
        return range(eval_count, n_docs)
    return range(0, n_docs)


class HFTokenizedDataset(grain.RandomAccessDataSource):
    """HuggingFace dataset, tokenized and filtered via cached .map()/.filter().

    Rows are deterministic chunks from source_seq_len-token blocks drawn from a
    single filtered document pool.
    Tokenization and filtering are cached by HF datasets -- only the first run
    pays the cost; subsequent runs load instantly from the Arrow cache.
    """

    def __init__(self, *, hf_dataset: str, hf_subset: str | None, hf_text_column: str,
                 split: str, seq_len: int, tokenizer_name: str,
                 vocab_size: int | None = None,
                 min_seq_len: int = 0,
                 cache_dir: str | None = None, num_proc: int = 8,
                 data_partition: str = "all",
                 source_seq_len: int = 0,
                 eval_fraction: float = 0.05):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if not 0 <= eval_fraction < 1:
            raise ValueError(f"eval_fraction must be in [0, 1), got {eval_fraction}")
        if data_partition not in {"train", "eval", "all"}:
            raise ValueError(f"data_partition must be train, eval, or all; got {data_partition!r}")

        source_seq_len = source_seq_len if source_seq_len > 0 else max(seq_len, min_seq_len)
        if source_seq_len % seq_len != 0:
            raise ValueError(f"source_seq_len={source_seq_len} must be divisible by seq_len={seq_len}")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        filter_seq_len = source_seq_len
        min_len = filter_seq_len + 1

        if vocab_size is not None and vocab_size < tokenizer.vocab_size:
            from ttt.dataloader.retokenizer import _make_truncated_tokenizer
            truncated = _make_truncated_tokenizer(tokenizer, vocab_size)
            encode_fn = lambda texts: [truncated.encode(t).ids for t in texts]
            print(f"Using truncated tokenizer: {tokenizer.vocab_size} → {vocab_size}")
        else:
            encode_fn = lambda texts: tokenizer(texts, add_special_tokens=False)["input_ids"]

        ds = load_dataset(hf_dataset, hf_subset or None, split=split, cache_dir=cache_dir or None)
        ds = ds.map(
            lambda rows: {"input_ids": [ids for ids in encode_fn(rows[hf_text_column]) if len(ids) >= min_len]},
            batched=True,
            remove_columns=ds.column_names,
            num_proc=num_proc,
            desc="Tokenizing + filtering",
        )

        self._ds = ds
        self.seq_len = seq_len
        self.source_seq_len = source_seq_len
        self.min_seq_len = filter_seq_len
        self._chunk_index: list[tuple[int, int]] = []

        doc_indices = _doc_indices(len(ds), data_partition, eval_fraction)
        chunks_per_source = source_seq_len // seq_len
        for doc_idx in doc_indices:
            token_count = len(ds[doc_idx]["input_ids"])
            n_source_blocks = (token_count - 1) // source_seq_len
            for source_idx in range(n_source_blocks):
                source_start = source_idx * source_seq_len
                for chunk_idx in range(chunks_per_source):
                    self._chunk_index.append((doc_idx, source_start + chunk_idx * seq_len))

        print(
            "HFTokenizedDataset: "
            f"{len(ds):,} filtered documents with >={min_len} tokens; "
            f"{len(doc_indices):,} {data_partition} documents; "
            f"{len(self._chunk_index):,} chunks of {seq_len + 1:,} tokens"
        )

    def __getitem__(self, idx):
        doc_idx, start = self._chunk_index[idx]
        end = start + self.seq_len + 1
        return np.array(self._ds[doc_idx]["input_ids"][start:end], dtype=np.int32)

    def __len__(self):
        return len(self._chunk_index)

    @property
    def document_ids(self) -> np.ndarray:
        return np.asarray([doc_idx for doc_idx, _ in self._chunk_index], dtype=np.int64)


class SyntheticKVDataset(grain.RandomAccessDataSource):
    """On-the-fly token associative-recall documents.

    Each document gets a fresh table of disjoint query/key token pairs. Training
    examples alternate query, key, query, key, ... and only key targets are
    supervised.
    """

    def __init__(
        self,
        *,
        seq_len: int,
        vocab_size: int,
        bos_token_id: int,
        eos_token_id: int,
        num_pairs: int = 32768,
        num_docs: int = 8192,
        seed: int = 0,
        data_partition: str = "all",
        eval_fraction: float = 0.05,
    ):
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if not 0 <= eval_fraction < 1:
            raise ValueError(f"eval_fraction must be in [0, 1), got {eval_fraction}")
        if data_partition not in {"train", "eval", "all"}:
            raise ValueError(f"data_partition must be train, eval, or all; got {data_partition!r}")
        if num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive, got {num_pairs}")
        if num_docs <= 0:
            raise ValueError(f"num_docs must be positive, got {num_docs}")

        excluded = {token for token in (bos_token_id, eos_token_id) if 0 <= token < vocab_size}
        self._available_tokens = np.array([token for token in range(vocab_size) if token not in excluded], dtype=np.int32)
        if 2 * num_pairs > len(self._available_tokens):
            raise ValueError(
                f"synthetic_num_pairs={num_pairs} requires {2 * num_pairs} distinct non-special tokens, "
                f"but vocab_size={vocab_size} leaves only {len(self._available_tokens)}"
            )

        self.seq_len = seq_len
        self.num_pairs = num_pairs
        self.seed = seed
        self._doc_indices = list(_doc_indices(num_docs, data_partition, eval_fraction))
        ranks = np.arange(1, num_pairs + 1, dtype=np.float64)
        self._pair_probs = (1.0 / ranks) / np.sum(1.0 / ranks)

        print(
            "SyntheticKVDataset: "
            f"{num_docs:,} docs; {len(self._doc_indices):,} {data_partition} docs; "
            f"{num_pairs:,} pairs/doc; {seq_len + 1:,} tokens/sample"
        )

    def __getitem__(self, idx):
        doc_idx = self._doc_indices[idx]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, doc_idx]))
        pair_tokens = rng.choice(self._available_tokens, size=2 * self.num_pairs, replace=False)
        query_tokens = pair_tokens[:self.num_pairs]
        key_tokens = pair_tokens[self.num_pairs:]

        token_count = self.seq_len + 1
        sample_count = (token_count + 1) // 2
        pair_indices = rng.choice(self.num_pairs, size=sample_count, replace=True, p=self._pair_probs)

        tokens = np.empty(token_count, dtype=np.int32)
        tokens[0::2] = query_tokens[pair_indices]
        tokens[1::2] = key_tokens[pair_indices[:token_count // 2]]
        loss_masks = (np.arange(self.seq_len) % 2 == 0)
        return tokens, loss_masks

    def __len__(self):
        return len(self._doc_indices)


class HFNcaRawDataset(grain.RandomAccessDataSource):
    """Raw NCA rollout rows from Hugging Face, patch-tokenized on load."""

    def __init__(
        self,
        *,
        hf_dataset: str,
        hf_subset: str | None,
        split: str,
        seq_len: int,
        patch_size: int = 2,
        num_colors: int = 10,
        mask_delimiters: bool = True,
        cache_dir: str | None = None,
        data_partition: str = "all",
    ):
        from datasets import load_dataset

        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if num_colors <= 1:
            raise ValueError(f"num_colors must be >1, got {num_colors}")
        if data_partition not in {"train", "eval", "all"}:
            raise ValueError(f"data_partition must be train, eval, or all; got {data_partition!r}")

        effective_split = "validation" if data_partition == "eval" and split == "train" else split
        self._ds = load_dataset(hf_dataset, hf_subset or None, split=effective_split, cache_dir=cache_dir or None)
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_colors = num_colors
        self.mask_delimiters = mask_delimiters
        self.start_token = num_colors ** (patch_size * patch_size)
        self.end_token = self.start_token + 1

        print(
            "HFNcaRawDataset: "
            f"{len(self._ds):,} {effective_split} rows; patch={patch_size}; "
            f"vocab={self.end_token + 1:,}; {seq_len + 1:,} tokens/sample"
        )

    def _encode_rollout(self, row) -> np.ndarray:
        recorded_steps = int(row["recorded_steps"])
        grid_size = int(row["grid_size"])
        raw_rollout = row["rollout"]
        if isinstance(raw_rollout, bytes | bytearray | memoryview):
            rollout = np.frombuffer(raw_rollout, dtype=np.uint8).astype(np.int32).reshape(recorded_steps, grid_size, grid_size)
        else:
            rollout = np.asarray(raw_rollout, dtype=np.int32).reshape(recorded_steps, grid_size, grid_size)

        if grid_size % self.patch_size != 0:
            raise ValueError(f"grid_size={grid_size} must be divisible by patch_size={self.patch_size}")

        n_h = grid_size // self.patch_size
        n_w = grid_size // self.patch_size
        patches = rollout.reshape(recorded_steps, n_h, self.patch_size, n_w, self.patch_size)
        patches = patches.transpose(0, 1, 3, 2, 4).reshape(recorded_steps, n_h * n_w, self.patch_size * self.patch_size)
        powers = (self.num_colors ** np.arange(self.patch_size * self.patch_size, dtype=np.int32))
        patch_tokens = np.sum(patches * powers, axis=-1, dtype=np.int32)

        grid_tokens = np.empty((recorded_steps, n_h * n_w + 2), dtype=np.int32)
        grid_tokens[:, 0] = self.start_token
        grid_tokens[:, 1:-1] = patch_tokens
        grid_tokens[:, -1] = self.end_token
        return grid_tokens.reshape(-1)

    def __getitem__(self, idx):
        tokens = self._encode_rollout(self._ds[int(idx)])
        token_count = self.seq_len + 1
        if len(tokens) < token_count:
            raise ValueError(f"NCA row has {len(tokens):,} tokens, but seq_len={self.seq_len} requires {token_count:,}")

        tokens = tokens[:token_count]
        if not self.mask_delimiters:
            return tokens

        target_tokens = tokens[1:]
        loss_masks = (target_tokens != self.start_token) & (target_tokens != self.end_token)
        return tokens, loss_masks

    def __len__(self):
        return len(self._ds)


class DummyDataset(grain.RandomAccessDataSource):
    def __init__(self, *, seq_len: int, num_tokens: int = 2**25):
        self.seq_len = seq_len
        self.num_tokens = num_tokens

    def __getitem__(self, idx):
        return np.random.randint(0, 20, (self.seq_len + 1,), dtype=np.int32)

    def __len__(self):
        return (self.num_tokens - self.seq_len - 1) // self.seq_len


def _to_batch(data: np.ndarray | tuple[np.ndarray, np.ndarray], *, bos_token_id: int, eos_token_id: int) -> Batch:
    if isinstance(data, tuple):
        tokens, loss_masks = data
        return Batch(
            input_ids=np.asarray(tokens[:-1]),
            target_tokens=np.asarray(tokens[1:]),
            loss_masks=np.asarray(loss_masks, dtype=bool),
        )

    tokens = np.asarray(data)
    return Batch(
        input_ids=tokens[:-1],
        target_tokens=tokens[1:],
        loss_masks=np.ones_like(tokens[1:], dtype=bool),
    )


def lm_dataset(
    *,
    dataset_kind: str = "hf",
    hf_dataset: str,
    hf_subset: str | None,
    hf_text_column: str,
    split: str,
    seq_len: int,
    global_batch_size: int,
    bos_token_id: int,
    eos_token_id: int,
    tokenizer_name: str,
    vocab_size: int | None = None,
    min_seq_len: int = 0,
    total_steps: int | None = None,
    seed=None,
    repeat: bool,
    shard_index: int | None = None,
    shard_count: int | None = None,
    shuffle: bool = True,
    cache_dir: str | None = None,
    data_partition: str = "all",
    source_seq_len: int = 0,
    eval_fraction: float = 0.05,
    synthetic_num_pairs: int = 32768,
    synthetic_num_docs: int = 8192,
    synthetic_seed: int = 0,
    nca_patch_size: int = 2,
    nca_num_colors: int = 10,
    nca_mask_delimiters: bool = True,
    return_source: bool = False,
) -> grain.MapDataset | tuple[grain.MapDataset, grain.RandomAccessDataSource]:
    if shard_index is None:
        shard_index = jax.process_index()
    if shard_count is None:
        shard_count = jax.process_count()

    assert global_batch_size % shard_count == 0
    host_batch_size = global_batch_size // shard_count

    if dataset_kind == "hf":
        source = HFTokenizedDataset(
            hf_dataset=hf_dataset, hf_subset=hf_subset, hf_text_column=hf_text_column,
            split=split, seq_len=seq_len, tokenizer_name=tokenizer_name,
            vocab_size=vocab_size, min_seq_len=min_seq_len, cache_dir=cache_dir,
            data_partition=data_partition,
            source_seq_len=source_seq_len, eval_fraction=eval_fraction,
        )
    elif dataset_kind == "synthetic_kv":
        if vocab_size is None:
            raise ValueError("vocab_size is required for dataset_kind='synthetic_kv'")
        source = SyntheticKVDataset(
            seq_len=seq_len,
            vocab_size=vocab_size,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            num_pairs=synthetic_num_pairs,
            num_docs=synthetic_num_docs,
            seed=synthetic_seed,
            data_partition=data_partition,
            eval_fraction=eval_fraction,
        )
    elif dataset_kind == "nca_hf_raw":
        source = HFNcaRawDataset(
            hf_dataset=hf_dataset,
            hf_subset=hf_subset,
            split=split,
            seq_len=seq_len,
            patch_size=nca_patch_size,
            num_colors=nca_num_colors,
            mask_delimiters=nca_mask_delimiters,
            cache_dir=cache_dir,
            data_partition=data_partition,
        )
    else:
        raise ValueError(f"Unknown dataset_kind={dataset_kind!r}")
    dataset = grain.MapDataset.source(source)

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    dataset = dataset.map(
        lambda data: _to_batch(data, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
    ).batch(batch_size=host_batch_size, drop_remainder=True)

    dataset_length = len(source)
    steps_per_epoch = dataset_length // global_batch_size

    if total_steps is not None and steps_per_epoch > 0:
        epochs = total_steps / steps_per_epoch
        print(f"Training for {epochs:.2f} epochs ({total_steps} steps, {steps_per_epoch} steps/epoch)")
        assert epochs <= 5, f"Too many epochs ({epochs:.2f} > 5). Reduce total_steps or increase dataset size."

    if repeat:
        print(f"Repeating dataset. Length {dataset_length}.")
        dataset = dataset.repeat()
    else:
        dataset_length = len(dataset)
        trimmed_length = (dataset_length // shard_count) * shard_count
        dataset = dataset[:trimmed_length]
        print(f"Trimming dataset. Length {dataset_length} → {trimmed_length}.")

    dataset = dataset[shard_index::shard_count]
    if return_source:
        return dataset, source
    return dataset


def dummy_dataset(
    seq_len: int,
    global_batch_size: int,
    bos_token_id: int,
    eos_token_id: int,
    repeat: bool = False,
    num_tokens: int = 2**25,
):
    shard_index = jax.process_index()
    shard_count = jax.process_count()

    dataset = grain.MapDataset.source(DummyDataset(seq_len=seq_len, num_tokens=num_tokens))

    host_batch_size = global_batch_size // shard_count
    dataset = dataset.map(
        lambda data: _to_batch(data, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
    ).batch(batch_size=host_batch_size, drop_remainder=True)

    if repeat:
        print("Repeating dataset.")
        dataset = dataset.repeat()

    dataset = dataset[shard_index::shard_count]
    return dataset
