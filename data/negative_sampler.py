"""Type-constrained negative sampler for link prediction.

For each positive triple (s, r, o), corrupts the object with a random
entity of the same semantic type. For example:
    practiced_in corruptions sample from country entities only
    located_in   corruptions sample from region entities only

Filtered evaluation: known positive triples are excluded from the
negative pool to avoid false negatives during evaluation.
"""

from __future__ import annotations

import random
import torch
from torch import Tensor


class NegativeSampler:
    """Type-constrained triple corruption for training and evaluation."""

    def __init__(
        self,
        type_constraints: dict[int, list[int]],
        all_triples: list[tuple[int, int, int]],
        seed: int = 0,
    ):
        """
        Args:
            type_constraints: relation_id → list of valid object entity ids
            all_triples: all known positive triples (for filtered evaluation)
            seed: random seed
        """
        self.type_constraints = type_constraints
        self.rng = random.Random(seed)

        # Build a set of known positives for fast lookup
        self.known_positives: set[tuple[int, int, int]] = set(all_triples)

    def sample_negatives(
        self,
        triples: list[tuple[int, int, int]],
        n_negative: int = 1,
        filtered: bool = False,
    ) -> list[tuple[int, int, int]]:
        """Generate negative triples by corrupting the object.

        Args:
            triples: positive triples to corrupt
            n_negative: number of negatives per positive
            filtered: if True, skip known positives when sampling

        Returns:
            List of negative triples, length = len(triples) * n_negative
        """
        negatives: list[tuple[int, int, int]] = []
        for s, r, o in triples:
            candidates = self.type_constraints.get(r, [])
            if not candidates:
                continue
            for _ in range(n_negative):
                neg_o = self._sample_one(s, r, o, candidates, filtered)
                negatives.append((s, r, neg_o))
        return negatives

    def _sample_one(
        self,
        s: int,
        r: int,
        o: int,
        candidates: list[int],
        filtered: bool,
    ) -> int:
        """Sample one negative object, avoiding the true object."""
        for _ in range(100):  # max attempts
            neg = self.rng.choice(candidates)
            if neg == o:
                continue
            if filtered and (s, r, neg) in self.known_positives:
                continue
            return neg
        # Fallback: return any candidate different from o
        for neg in candidates:
            if neg != o:
                return neg
        return candidates[0]

    def get_all_candidates(self, r: int, exclude: set[tuple[int, int, int]] | None = None) -> Tensor:
        """Get all valid object candidates for a relation (for ranking evaluation).

        Args:
            r: relation id
            exclude: set of (s, r, o) triples to exclude from candidates

        Returns:
            LongTensor of candidate object entity ids
        """
        candidates = self.type_constraints.get(r, [])
        if exclude is None:
            return torch.tensor(candidates, dtype=torch.long)
        filtered = [c for c in candidates if c not in {o for s_, r_, o in exclude if r_ == r}]
        return torch.tensor(filtered, dtype=torch.long)

    def rank_candidates(
        self,
        s_id: int,
        r_id: int,
        o_id: int,
        scores: Tensor,
        candidate_ids: Tensor,
        filtered: bool = True,
    ) -> int:
        """Compute the rank of the correct object among candidates.

        Args:
            s_id: subject entity id
            r_id: relation id
            o_id: correct object entity id
            scores: (num_candidates,) score for each candidate
            candidate_ids: (num_candidates,) entity ids corresponding to scores
            filtered: exclude known positives other than (s_id, r_id, o_id)

        Returns:
            1-based rank of the correct object
        """
        id_to_idx = {cid.item(): i for i, cid in enumerate(candidate_ids)}
        correct_idx = id_to_idx.get(o_id, -1)
        if correct_idx == -1:
            return len(candidate_ids) + 1

        correct_score = scores[correct_idx].item()

        # Count how many candidates score higher than correct
        rank = 1
        for i, (cid, score) in enumerate(zip(candidate_ids.tolist(), scores.tolist())):
            if i == correct_idx:
                continue
            if filtered and (s_id, r_id, cid) in self.known_positives and cid != o_id:
                continue
            if score > correct_score:
                rank += 1

        return rank
