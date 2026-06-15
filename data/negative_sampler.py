"""Type-constrained negative sampler for link prediction.

For each positive triple (s, r, o), corrupts the object with a random
entity of the same semantic type. For example:
    practiced_in corruptions sample from country entities only
    located_in   corruptions sample from region entities only

Filtered evaluation: known positive triples are excluded from the
negative pool to avoid false negatives during evaluation.
"""

from __future__ import annotations

import torch
from torch import Tensor


class NegativeSampler:
    """Type-constrained triple corruption for training and evaluation."""

    def __init__(
        self,
        type_constraints: dict[int, list[int]],
        all_triples: list[tuple[int, int, int]],
    ):
        """
        Args:
            type_constraints: relation_id → list of valid object entity ids
            all_triples: all known positive triples (for filtered evaluation)
        """
        self.type_constraints = type_constraints

        # Build a set of known positives for fast lookup
        self.known_positives: set[tuple[int, int, int]] = set(all_triples)

        # Precompute a candidate tensor per relation for vectorized sampling.
        self._cand_tensors: dict[int, Tensor] = {
            r: torch.tensor(ids, dtype=torch.long)
            for r, ids in type_constraints.items() if ids
        }

    def sample_negatives_tensor(
        self,
        pos_s: Tensor,
        pos_r: Tensor,
        pos_o: Tensor,
        n_negative: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Vectorized object-corruption negatives (training).

        Replaces the per-triple Python ``rng.choice`` + retry loop with grouped
        tensor sampling: one ``torch.randint`` per relation present in the batch,
        with collisions against the true object resampled in a few vectorized
        passes. Unfiltered (other known positives are allowed as negatives, which
        is standard for training throughput).

        Args:
            pos_s, pos_r, pos_o: (B,) long tensors for the positive triples.
            n_negative: negatives per positive.

        Returns:
            (neg_s, neg_r, neg_o), each (M,) with
            M = (#triples whose relation has candidates) * n_negative.
        """
        device = pos_s.device
        keep = torch.tensor(
            [int(r) in self._cand_tensors for r in pos_r.tolist()], dtype=torch.bool
        )
        if not bool(keep.any()):
            empty = torch.zeros(0, dtype=torch.long, device=device)
            return empty, empty, empty

        s, r, o = pos_s[keep], pos_r[keep], pos_o[keep]
        neg_s  = s.repeat_interleave(n_negative)
        neg_r  = r.repeat_interleave(n_negative)
        true_o = o.repeat_interleave(n_negative)
        neg_o  = torch.empty_like(true_o)

        for r_id in torch.unique(neg_r).tolist():
            cand = self._cand_tensors[int(r_id)].to(device)
            mask = neg_r == r_id
            k = int(mask.sum())
            sampled = cand[torch.randint(len(cand), (k,), device=device)]
            t_o = true_o[mask]
            for _ in range(10):                    # resample collisions w/ true object
                coll = sampled == t_o
                if not bool(coll.any()):
                    break
                sampled[coll] = cand[torch.randint(len(cand), (int(coll.sum()),), device=device)]
            neg_o[mask] = sampled

        return neg_s, neg_r, neg_o
