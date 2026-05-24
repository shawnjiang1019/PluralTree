"""Batch packaging for training.

Each training step samples a batch of triples from the triple list.
Since the full tree is encoded once per epoch (not per batch), the
collate step just packages triple ids into tensors.
"""

from __future__ import annotations

import random
import torch
from torch import Tensor

from data.negative_sampler import NegativeSampler


class TripleBatchSampler:
    """Samples batches of (positive, negative) triple pairs for training."""

    def __init__(
        self,
        triples: list[tuple[int, int, int]],
        sampler: NegativeSampler,
        batch_size: int = 128,
        n_negative: int = 10,
        seed: int = 0,
    ):
        self.triples   = triples
        self.sampler   = sampler
        self.batch_size = batch_size
        self.n_negative = n_negative
        self.rng = random.Random(seed)
        self._indices = list(range(len(triples)))

    def __len__(self) -> int:
        return max(1, len(self.triples) // self.batch_size)

    def __iter__(self):
        self.rng.shuffle(self._indices)
        for start in range(0, len(self._indices), self.batch_size):
            batch_idx = self._indices[start : start + self.batch_size]
            pos_triples = [self.triples[i] for i in batch_idx]
            neg_triples = self.sampler.sample_negatives(
                pos_triples, n_negative=self.n_negative
            )
            yield _pack_batch(pos_triples, neg_triples, self.n_negative)


def _pack_batch(
    pos_triples: list[tuple[int, int, int]],
    neg_triples: list[tuple[int, int, int]],
    n_negative: int,
) -> dict[str, Tensor]:
    """Pack positive and negative triples into tensors.

    Returns dict with:
        pos_s, pos_r, pos_o : (B,) LongTensors for positive triples
        neg_s, neg_r, neg_o : (B * n_negative,) LongTensors for negatives
    """
    def unpack(triples):
        if not triples:
            e = torch.zeros(0, dtype=torch.long)
            return e, e, e
        s, r, o = zip(*triples)
        return (
            torch.tensor(s, dtype=torch.long),
            torch.tensor(r, dtype=torch.long),
            torch.tensor(o, dtype=torch.long),
        )

    pos_s, pos_r, pos_o = unpack(pos_triples)
    neg_s, neg_r, neg_o = unpack(neg_triples)

    return {
        "pos_s": pos_s,
        "pos_r": pos_r,
        "pos_o": pos_o,
        "neg_s": neg_s,
        "neg_r": neg_r,
        "neg_o": neg_o,
    }
