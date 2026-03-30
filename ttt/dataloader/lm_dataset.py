import grain.python as grain
import jax
import numpy as np

from ttt.model.data import Batch


class HFTokenizedDataset(grain.RandomAccessDataSource):
    """HuggingFace dataset, tokenized and filtered via cached .map()/.filter().

    Each row is one document truncated to seq_len+1 tokens.
    Tokenization and filtering are cached by HF datasets -- only the first run
    pays the cost; subsequent runs load instantly from the Arrow cache.
    """

    def __init__(self, *, hf_dataset: str, hf_subset: str | None, hf_text_column: str,
                 split: str, seq_len: int, tokenizer_name: str,
                 vocab_size: int | None = None,
                 cache_dir: str | None = None, num_proc: int = 4):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        min_len = seq_len + 1

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
        print(f"HFTokenizedDataset: {len(ds):,} documents with >={min_len} tokens")

    def __getitem__(self, idx):
        return np.array(self._ds[idx]["input_ids"][:self.seq_len + 1], dtype=np.int32)

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


def _to_batch(data: np.ndarray, *, bos_token_id: int, eos_token_id: int) -> Batch:
    tokens = np.asarray(data)
    return Batch(
        input_ids=tokens[:-1],
        target_tokens=tokens[1:],
        loss_masks=(tokens[1:] != bos_token_id),
    )


def lm_dataset(
    *,
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
    total_steps: int | None = None,
    seed=None,
    repeat: bool,
    shard_index: int | None = None,
    shard_count: int | None = None,
    shuffle: bool = True,
    cache_dir: str | None = None,
) -> grain.MapDataset:
    if shard_index is None:
        shard_index = jax.process_index()
    if shard_count is None:
        shard_count = jax.process_count()

    assert global_batch_size % shard_count == 0
    host_batch_size = global_batch_size // shard_count

    source = HFTokenizedDataset(
        hf_dataset=hf_dataset, hf_subset=hf_subset, hf_text_column=hf_text_column,
        split=split, seq_len=seq_len, tokenizer_name=tokenizer_name,
        vocab_size=vocab_size, cache_dir=cache_dir,
    )
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
