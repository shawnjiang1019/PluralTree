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
            pos = [self.triples[i] for i in batch_idx]
            ps, pr, po = zip(*pos)
            pos_s = torch.tensor(ps, dtype=torch.long)
            pos_r = torch.tensor(pr, dtype=torch.long)
            pos_o = torch.tensor(po, dtype=torch.long)
            neg_s, neg_r, neg_o = self.sampler.sample_negatives_tensor(
                pos_s, pos_r, pos_o, self.n_negative
            )
            yield {
                "pos_s": pos_s, "pos_r": pos_r, "pos_o": pos_o,
                "neg_s": neg_s, "neg_r": neg_r, "neg_o": neg_o,
            }
