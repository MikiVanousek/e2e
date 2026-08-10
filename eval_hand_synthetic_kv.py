import argparse
import importlib
import math
import sys
import types
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

try:
    importlib.import_module("grain.python")
except ModuleNotFoundError:
    grain_package = types.ModuleType("grain")
    grain_package.__path__ = []
    grain_python = types.ModuleType("grain.python")
    grain_python.MapDataset = object
    grain_python.RandomAccessDataSource = object
    grain_package.python = grain_python
    sys.modules["grain"] = grain_package
    sys.modules["grain.python"] = grain_python

from ttt.dataloader.lm_dataset import SyntheticKVDataset


@dataclass
class EvalStats:
    literal_loss_sum: float = 0.0
    bayes_loss_sum: float = 0.0
    scored_tokens: int = 0
    first_occurrences: int = 0
    repeated_occurrences: int = 0
    repeated_correct: int = 0

    def add(self, other: "EvalStats") -> None:
        self.literal_loss_sum += other.literal_loss_sum
        self.bayes_loss_sum += other.bayes_loss_sum
        self.scored_tokens += other.scored_tokens
        self.first_occurrences += other.first_occurrences
        self.repeated_occurrences += other.repeated_occurrences
        self.repeated_correct += other.repeated_correct


@dataclass(frozen=True)
class SparseLinear:
    """Sparse fixed weight matrix y = W x, stored as coordinate triples."""

    name: str
    out_features: int
    in_features: int
    rows: np.ndarray
    cols: np.ndarray
    values: np.ndarray

    @classmethod
    def from_pairs(
        cls,
        name: str,
        *,
        out_features: int,
        in_features: int,
        pairs: list[tuple[int, int, float]],
    ) -> "SparseLinear":
        rows, cols, values = zip(*pairs) if pairs else ([], [], [])
        return cls(
            name=name,
            out_features=out_features,
            in_features=in_features,
            rows=np.asarray(rows, dtype=np.int32),
            cols=np.asarray(cols, dtype=np.int32),
            values=np.asarray(values, dtype=np.float32),
        )

    @property
    def nnz(self) -> int:
        return int(self.values.size)


@dataclass(frozen=True)
class FixedWeightInductionTransformerWeights:
    """Explicit sparse weights for the two-head induction circuit.

    Residual stream layout:
      token one-hot | position one-hot | constant | previous-token slot | retrieved-token slot

    Layer 1 is a previous-token head: position t attends to position t-1 and
    writes that token into the previous-token slot.

    Layer 2 is an induction head: current query attends to earlier positions
    whose previous-token slot equals the query, then writes the matching key
    token into the retrieved-token slot. A learned null/sink key handles the
    no-match case and leaves the retrieved-token slot at zero.
    """

    vocab_size: int
    max_seq_len: int
    attention_logit: float = 40.0
    output_logit: float = 80.0
    w_q1: SparseLinear = field(init=False)
    w_k1: SparseLinear = field(init=False)
    w_v1: SparseLinear = field(init=False)
    w_o1: SparseLinear = field(init=False)
    w_q2: SparseLinear = field(init=False)
    w_k2: SparseLinear = field(init=False)
    w_v2: SparseLinear = field(init=False)
    w_o2: SparseLinear = field(init=False)

    def __post_init__(self) -> None:
        vocab = self.vocab_size
        seq = self.max_seq_len
        token_offset = 0
        pos_offset = token_offset + vocab
        const_idx = pos_offset + seq
        prev_offset = const_idx + 1
        retrieved_offset = prev_offset + vocab
        residual_width = retrieved_offset + vocab

        scale = 2.0 * self.attention_logit
        bias = -self.attention_logit

        q1_pairs = [(t - 1, pos_offset + t, scale) for t in range(1, seq)]
        q1_pairs.append((seq, const_idx, 1.0))
        k1_pairs = [(t, pos_offset + t, 1.0) for t in range(seq)]
        k1_pairs.append((seq, const_idx, bias))

        q2_pairs = [(tok, token_offset + tok, scale) for tok in range(vocab)]
        q2_pairs.append((vocab, const_idx, 1.0))
        k2_pairs = [(tok, prev_offset + tok, 1.0) for tok in range(vocab)]
        k2_pairs.append((vocab, const_idx, bias))

        object.__setattr__(self, "w_q1", SparseLinear.from_pairs("layer1.prev_token.W_Q", out_features=seq + 1, in_features=residual_width, pairs=q1_pairs))
        object.__setattr__(self, "w_k1", SparseLinear.from_pairs("layer1.prev_token.W_K", out_features=seq + 1, in_features=residual_width, pairs=k1_pairs))
        object.__setattr__(self, "w_v1", SparseLinear.from_pairs(
            "layer1.prev_token.W_V",
            out_features=vocab,
            in_features=residual_width,
            pairs=[(tok, token_offset + tok, 1.0) for tok in range(vocab)],
        ))
        object.__setattr__(self, "w_o1", SparseLinear.from_pairs(
            "layer1.prev_token.W_O",
            out_features=residual_width,
            in_features=vocab,
            pairs=[(prev_offset + tok, tok, 1.0) for tok in range(vocab)],
        ))
        object.__setattr__(self, "w_q2", SparseLinear.from_pairs("layer2.induction.W_Q", out_features=vocab + 1, in_features=residual_width, pairs=q2_pairs))
        object.__setattr__(self, "w_k2", SparseLinear.from_pairs("layer2.induction.W_K", out_features=vocab + 1, in_features=residual_width, pairs=k2_pairs))
        object.__setattr__(self, "w_v2", SparseLinear.from_pairs(
            "layer2.induction.W_V",
            out_features=vocab,
            in_features=residual_width,
            pairs=[(tok, token_offset + tok, 1.0) for tok in range(vocab)],
        ))
        object.__setattr__(self, "w_o2", SparseLinear.from_pairs(
            "layer2.induction.W_O",
            out_features=residual_width,
            in_features=vocab,
            pairs=[(retrieved_offset + tok, tok, 1.0) for tok in range(vocab)],
        ))

    def summary(self) -> str:
        matrices = [self.w_q1, self.w_k1, self.w_v1, self.w_o1, self.w_q2, self.w_k2, self.w_v2, self.w_o2]
        return "\n".join(f"{w.name}: shape=({w.out_features}, {w.in_features}), nnz={w.nnz:,}" for w in matrices)


class FixedWeightInductionTransformer:
    """No-training literal fixed-weight transformer for SyntheticKVDataset.

    Layer 1 copies each query token into the following key position's memory slot.
    Layer 2 retrieves the previous key whose memory slot matches the current query.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        bos_token_id: int,
        eos_token_id: int,
        max_seq_len: int,
        attention_logit: float = 40.0,
        output_logit: float = 80.0,
    ):
        self.vocab_size = vocab_size
        self.special_tokens = {token for token in (bos_token_id, eos_token_id) if 0 <= token < vocab_size}
        self.usable_tokens = vocab_size - len(self.special_tokens)
        self.output_logit = output_logit
        self.weights = FixedWeightInductionTransformerWeights(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            attention_logit=attention_logit,
            output_logit=output_logit,
        )

    def _literal_target_loss(self, *, was_copied: bool) -> float:
        if not was_copied:
            return math.log(self.usable_tokens)
        return math.log1p((self.usable_tokens - 1) * math.exp(-self.output_logit))

    def _layer1_previous_token_slot(self, tokens: np.ndarray) -> np.ndarray:
        previous_token_slot = np.full_like(tokens, fill_value=-1)
        previous_token_slot[1:] = tokens[:-1]
        return previous_token_slot

    def _layer2_retrieve_key(self, *, tokens: np.ndarray, previous_token_slot: np.ndarray, pos: int) -> tuple[int, bool]:
        query = tokens[pos]
        matching_key_positions = np.flatnonzero(previous_token_slot[: pos + 1] == query)
        if matching_key_positions.size == 0:
            return -1, False
        return int(tokens[int(matching_key_positions[-1])]), True

    def score_document(self, tokens: np.ndarray, loss_masks: np.ndarray) -> EvalStats:
        memory: dict[int, int] = {}
        previous_token_slot = self._layer1_previous_token_slot(tokens)
        stats = EvalStats()

        for pos, should_score in enumerate(loss_masks):
            if not should_score:
                continue

            query = int(tokens[pos])
            target_key = int(tokens[pos + 1])
            retrieved_key, has_match = self._layer2_retrieve_key(tokens=tokens, previous_token_slot=previous_token_slot, pos=pos)
            stats.scored_tokens += 1

            if has_match:
                stats.repeated_occurrences += 1
                copied_correctly = retrieved_key == target_key
                stats.repeated_correct += int(copied_correctly)
                stats.literal_loss_sum += self._literal_target_loss(was_copied=copied_correctly)
            else:
                stats.first_occurrences += 1
                remaining_possible_keys = self.usable_tokens - 2 * len(memory) - 1
                stats.literal_loss_sum += self._literal_target_loss(was_copied=False)
                stats.bayes_loss_sum += math.log(remaining_possible_keys)
                memory[query] = target_key

        return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the hand-built no-training transformer on synthetic KV sequences.")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--bos-token-id", type=int, default=0)
    parser.add_argument("--eos-token-id", type=int, default=1)
    parser.add_argument("--num-pairs", type=int, default=4095)
    parser.add_argument("--num-docs", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("train", "eval", "all"), default="eval")
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--max-docs", type=int, default=64, help="Number of generated documents to score. Use 0 for the full split.")
    parser.add_argument("--attention-logit", type=float, default=40.0)
    parser.add_argument("--output-logit", type=float, default=80.0)
    parser.add_argument("--print-weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = SyntheticKVDataset(
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
        num_pairs=args.num_pairs,
        num_docs=args.num_docs,
        seed=args.seed,
        data_partition=args.split,
        eval_fraction=args.eval_fraction,
    )
    model = FixedWeightInductionTransformer(
        vocab_size=args.vocab_size,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
        max_seq_len=args.seq_len + 1,
        attention_logit=args.attention_logit,
        output_logit=args.output_logit,
    )
    if args.print_weights:
        print(model.weights.summary())

    n_docs = len(dataset) if args.max_docs == 0 else min(args.max_docs, len(dataset))
    total = EvalStats()
    for i in tqdm(range(n_docs), desc="Scoring synthetic KV docs"):
        tokens, loss_masks = dataset[i]
        total.add(model.score_document(np.asarray(tokens), np.asarray(loss_masks)))

    literal_loss = total.literal_loss_sum / total.scored_tokens
    literal_bpb = literal_loss / math.log(2.0)
    bayes_loss = total.bayes_loss_sum / total.scored_tokens
    bayes_bpb = bayes_loss / math.log(2.0)
    repeated_accuracy = total.repeated_correct / max(total.repeated_occurrences, 1)
    copyable_fraction = total.repeated_occurrences / total.scored_tokens
    first_fraction = total.first_occurrences / total.scored_tokens

    print(f"docs: {n_docs}")
    print(f"scored key predictions: {total.scored_tokens:,}")
    print(f"literal transformer loss: {literal_loss:.6f} nats/token")
    print(f"literal transformer bpb: {literal_bpb:.6f}")
    print(f"bayes first-occurrence lower-bound loss: {bayes_loss:.6f} nats/token")
    print(f"bayes first-occurrence lower-bound bpb: {bayes_bpb:.6f}")
    print(f"first occurrence fraction: {first_fraction:.6f}")
    print(f"copyable repeated fraction: {copyable_fraction:.6f}")
    print(f"repeated copy accuracy: {repeated_accuracy:.6f}")


if __name__ == "__main__":
    main()
