import json

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import AutoTokenizer

LLAMA_3_VOCAB_SIZE = 128256
LLAMA_3_BOS = 128000
LLAMA_3_EOS = 128001


class Retokenizer:
    """Decodes Llama 3 token IDs and re-encodes with a truncated BPE vocabulary."""

    def __init__(self, tokenizer_name: str, target_vocab_size: int, new_bos: int, new_eos: int):
        self.original = AutoTokenizer.from_pretrained(tokenizer_name)
        self.truncated = _make_truncated_tokenizer(self.original, target_vocab_size)
        self.new_bos = new_bos
        self.new_eos = new_eos

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        result = []
        segment: list[int] = []

        for tid in token_ids.tolist():
            if tid in (LLAMA_3_BOS, LLAMA_3_EOS):
                if segment:
                    text = self.original.decode(segment)
                    result.extend(self.truncated.encode(text).ids)
                    segment = []
                result.append(self.new_bos if tid == LLAMA_3_BOS else self.new_eos)
            else:
                segment.append(tid)

        if segment:
            text = self.original.decode(segment)
            result.extend(self.truncated.encode(text).ids)

        return np.array(result, dtype=np.int32)


def _make_truncated_tokenizer(original, target_vocab_size: int) -> Tokenizer:
    """Create a smaller BPE tokenizer by keeping only merges whose components exist."""
    tok_json = json.loads(original.backend_tokenizer.to_str())
    original_merges = tok_json["model"]["merges"]
    original_vocab = tok_json["model"]["vocab"]

    def _parts(m) -> tuple[str, str]:
        return tuple(m.split(" ")) if isinstance(m, str) else tuple(m)

    all_merge_results = {"".join(_parts(m)) for m in original_merges}
    added_strs = {t["content"] for t in tok_json.get("added_tokens", [])}
    base_tokens = sorted(
        [(tok, vid) for tok, vid in original_vocab.items() if tok not in all_merge_results and tok not in added_strs],
        key=lambda x: x[1],
    )

    num_merges = target_vocab_size - len(base_tokens)
    assert num_merges > 0

    # Walk merges in order, only keeping those whose components already exist
    # and whose result isn't already in the vocab (different pairs can produce
    # the same string, e.g. ("a","bc") and ("ab","c") both yield "abc").
    valid_tokens = {tok for tok, _ in base_tokens}
    kept_merges: list[tuple[str, str]] = []
    for m in original_merges:
        if len(kept_merges) >= num_merges:
            break
        a, b = _parts(m)
        result = a + b
        if a in valid_tokens and b in valid_tokens and result not in valid_tokens:
            valid_tokens.add(result)
            kept_merges.append((a, b))

    # Build vocab: base tokens 0..N-1, then merge results N..N+K-1
    new_vocab = {tok: i for i, (tok, _) in enumerate(base_tokens)}
    for i, (a, b) in enumerate(kept_merges):
        new_vocab[a + b] = len(base_tokens) + i

    # Construct directly via the BPE model API instead of JSON round-tripping
    bpe = BPE(vocab=new_vocab, merges=kept_merges)
    tok = Tokenizer(bpe)
    tok.pre_tokenizer = original.backend_tokenizer.pre_tokenizer
    tok.decoder = original.backend_tokenizer.decoder

    return tok
