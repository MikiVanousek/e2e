from __future__ import annotations

from pathlib import Path

import grain.python as grain
import jax
import numpy as np
import zarr.codecs
import zarr.storage
from tqdm import tqdm

from ttt.dataloader.retokenizer import LLAMA_3_VOCAB_SIZE, Retokenizer
from ttt.model.data import Batch

CHUNK_SIZE = 25_000


def chunks_volume_key(vocab_size: int, dataset_name: str, seq_len: int, split: str, chunk_size: int = CHUNK_SIZE) -> str:
    return f"v{vocab_size}-{dataset_name}-{seq_len}-{split}-c{chunk_size}"


class Dataset(grain.RandomAccessDataSource):
    def __init__(self, *, path: str, split: str, seq_len: int, retokenizer: Retokenizer | None = None):
        codec = zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle)

        store = zarr.storage.LocalStore(path, read_only=True)

        self._dataset = zarr.open_array(store, path=f"/{split}", codec=codec)

        self.split = self._dataset
        self.seq_len = seq_len
        self.retokenizer = retokenizer

    def __getitem__(self, idx):
        sample = self.split[idx * self.seq_len : (idx + 1) * self.seq_len + 1]
        assert len(sample) == (self.seq_len + 1), "Loader got a sequence with the wrong length!"

        if self.retokenizer is not None:
            sample = self.retokenizer(sample)[: self.seq_len + 1]
            assert len(sample) == (self.seq_len + 1), "Retokenization produced fewer tokens than expected!"

        return sample

    def __len__(self):
        return (self.split.shape[0] - 1) // self.seq_len


class ChunkedDataset(grain.RandomAccessDataSource):
    """Reads pre-retokenized .npy chunk files into RAM. Each chunk is (CHUNK_SIZE, seq_len+1) int32."""

    def __init__(self, *, chunks_dir: str, seq_len: int):
        chunks_path = Path(chunks_dir)
        chunk_files = sorted(chunks_path.glob("chunk_*.npy"))
        assert chunk_files, f"No chunk files found in {chunks_dir}"

        arrays = [np.load(str(f)) for f in chunk_files]
        self._data = np.concatenate(arrays, axis=0)
        assert self._data.shape[1] == seq_len + 1, f"Chunk seq dim {self._data.shape[1]} != expected {seq_len + 1}"
        print(f"ChunkedDataset: loaded {len(chunk_files)} chunks, {self._data.shape} ({self._data.nbytes / 1e9:.1f} GB)")

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return self._data.shape[0]


class DummyDataset(grain.RandomAccessDataSource):
    def __init__(self, *, seq_len: int, num_tokens: int = 2**25):
        self.seq_len = seq_len
        self.num_tokens = num_tokens

    def __getitem__(self, idx):
        sample = np.random.randint(0, 20, (self.seq_len + 1,), dtype=np.int32)
        return sample

    def __len__(self):
        return (self.num_tokens - self.seq_len - 1) // self.seq_len


def _to_batch(
    data: np.ndarray,
    *,
    bos_token_id: int,
    eos_token_id: int,
) -> Batch:
    tokens = np.asarray(data)
    return Batch(
        input_ids=tokens[:-1],
        target_tokens=tokens[1:],
        loss_masks=(tokens[1:] != bos_token_id),
    )


def lm_dataset(
    *,
    path: str,
    split: str,
    seq_len: int,
    global_batch_size: int,
    bos_token_id: int,
    eos_token_id: int,
    seed=None,
    repeat: bool,
    shard_index: int | None = None,
    shard_count: int | None = None,
    shuffle: bool = True,
    vocab_size: int | None = None,
    tokenizer_name: str | None = None,
    chunks_dir: str | None = None,
) -> grain.MapDataset:
    if shard_index is None:
        shard_index = jax.process_index()
    if shard_count is None:
        shard_count = jax.process_count()

    assert global_batch_size % shard_count == 0
    host_batch_size = global_batch_size // shard_count

    if chunks_dir is not None:
        source = ChunkedDataset(chunks_dir=chunks_dir, seq_len=seq_len)
    else:
        retokenizer = None
        if vocab_size is not None and vocab_size < LLAMA_3_VOCAB_SIZE:
            assert tokenizer_name is not None, "tokenizer_name is required when vocab_size < LLAMA_3_VOCAB_SIZE"
            retokenizer = Retokenizer(tokenizer_name, vocab_size, new_bos=bos_token_id, new_eos=eos_token_id)
        source = Dataset(path=path, split=split, seq_len=seq_len, retokenizer=retokenizer)

    dataset = grain.MapDataset.source(source)

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    dataset = dataset.map(
        lambda data: _to_batch(
            data,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
    ).batch(batch_size=host_batch_size, drop_remainder=True)

    dataset_length = len(source)

    if repeat:
        print(f"Repeating dataset. Length {dataset_length}.")
        dataset = dataset.repeat()
    else:
        dataset_length = len(dataset)
        trimmed_length = (dataset_length // shard_count) * shard_count  # Drop remainder
        dataset = dataset[:trimmed_length]
        print(f"Trimming dataset. Initial length {dataset_length}. New length {trimmed_length}.")

    dataset = dataset[shard_index::shard_count]

    return dataset


def save_chunk(
    *,
    path: str,
    split: str,
    seq_len: int,
    output_dir: str,
    chunk_idx: int,
    chunk_size: int = CHUNK_SIZE,
    vocab_size: int | None = None,
    tokenizer_name: str | None = None,
    bos_token_id: int,
    eos_token_id: int,
) -> bool:
    """Extract one chunk of sequences from Zarr (with optional retokenization) into a .npy file.

    Returns True if the chunk was created, False if it already existed.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    chunk_file = out / f"chunk_{chunk_idx:03d}.npy"
    if chunk_file.exists():
        print(f"Skipping existing {chunk_file}")
        return False

    retokenizer = None
    if vocab_size is not None and vocab_size < LLAMA_3_VOCAB_SIZE:
        assert tokenizer_name is not None, "tokenizer_name is required when vocab_size < LLAMA_3_VOCAB_SIZE"
        retokenizer = Retokenizer(tokenizer_name, vocab_size, new_bos=bos_token_id, new_eos=eos_token_id)

    source = Dataset(path=path, split=split, seq_len=seq_len, retokenizer=retokenizer)

    start = chunk_idx * chunk_size
    end = min(start + chunk_size, len(source))
    assert start < len(source), f"chunk {chunk_idx} starts at {start} but source only has {len(source)} sequences"

    data = np.stack([source[i] for i in tqdm(range(start, end), desc=f"chunk_{chunk_idx:03d}")])
    np.save(str(chunk_file), data)
    print(f"Saved {chunk_file}: {data.shape} ({data.nbytes / 1e9:.1f} GB)")
    return True


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

    dataset = grain.MapDataset.source(
        DummyDataset(seq_len=seq_len, num_tokens=num_tokens),
    )

    host_batch_size = global_batch_size // shard_count
    dataset = dataset.map(
        lambda data: _to_batch(
            data,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
    ).batch(batch_size=host_batch_size, drop_remainder=True)

    if repeat:
        print("Repeating dataset.")
        dataset = dataset.repeat()

    dataset = dataset[shard_index::shard_count]
    return dataset
