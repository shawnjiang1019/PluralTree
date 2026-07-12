"""Filtered link prediction evaluation: MRR and Hits@k.

For each test triple (s, r, o):
  1. Score (s, r, o) against all type-valid candidate objects
  2. Remove known positives from the candidate set (filtered setting)
  3. Rank the correct object among remaining candidates
  4. Compute MRR and Hits@{1, 3, 10}

The ranking is vectorized so it scales to large candidate sets (e.g. WN18RR
ranks against all ~40K entities). The filtered competitors of each query are
masked with tensor ops instead of a per-candidate Python loop.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor

from data.loaders.culturalbench import CulturalGraph
from data.negative_sampler import NegativeSampler
from training.scoring import HyperbolicLinkPredictor


def evaluate_link_prediction(
    h_all: Tensor,
    triples: list[tuple[int, int, int]],
    predictor: HyperbolicLinkPredictor,
    graph: CulturalGraph,
    neg_sampler: NegativeSampler,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Compute filtered MRR and Hits@{1,3,10} over a set of triples.

    Args:
        h_all:      (N, d_hidden) all entity hidden states on ball
        triples:    list of (s_id, r_id, o_id) to evaluate
        predictor:  scoring head
        graph:      CulturalGraph (for type constraints)
        neg_sampler: for filtered ranking
        device:     torch device

    Returns:
        dict with keys mrr, hits@1, hits@3, hits@10
    """
    device = torch.device(device)
    h_all = h_all.to(device)
    n_entities = h_all.shape[0]

    # (s, r) -> set of all known-true objects, for filtered ranking.
    true_objs: dict[tuple[int, int], set[int]] = defaultdict(set)
    for s, r, o in neg_sampler.known_positives:
        true_objs[(s, r)].add(o)

    # Per-relation candidate tensors, id->position maps, and the gathered candidate
    # embeddings — cached by the identity of the candidate list (relations may share
    # one big list, e.g. all ~40K entities on WN18RR). Caching h_cands here avoids
    # re-gathering the full (K, d) candidate matrix on every single triple.
    cand_cache: dict[int, tuple[Tensor, Tensor, Tensor]] = {}

    def candidates_for(r_id: int):
        ids = graph.type_constraints.get(r_id, [])
        if not ids:
            return None
        key = id(ids)
        cached = cand_cache.get(key)
        if cached is None:
            cand = torch.tensor(ids, dtype=torch.long, device=device)
            id2pos = torch.full((n_entities,), -1, dtype=torch.long, device=device)
            id2pos[cand] = torch.arange(cand.numel(), device=device)
            h_cands = h_all[cand]                          # (K, d) — gather once
            cand_cache[key] = (cand, id2pos, h_cands)
            cached = cand_cache[key]
        return cached

    ranks: list[int] = []
    for s_id, r_id, o_id in triples:
        cc = candidates_for(r_id)
        if cc is None:
            continue
        cand, id2pos, h_cands = cc

        correct_pos = id2pos[o_id].item()
        if correct_pos < 0:                       # true object not a candidate
            ranks.append(cand.numel() + 1)
            continue

        scores = predictor.score_all_candidates(h_all[s_id], r_id, h_cands)  # (K,)
        correct_score = scores[correct_pos]

        # Filtered: mask out OTHER known-true objects so they don't count as
        # beating the correct one. The correct object itself is kept.
        others = true_objs.get((s_id, r_id))
        if others:
            other_ids = torch.tensor([o for o in others if o != o_id],
                                     dtype=torch.long, device=device)
            if other_ids.numel():
                pos = id2pos[other_ids]
                pos = pos[pos >= 0]
                scores[pos] = float("-inf")

        rank = int((scores > correct_score).sum().item()) + 1
        ranks.append(rank)

    if not ranks:
        return {"mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0}

    r = torch.tensor(ranks, dtype=torch.float)
    return {
        "mrr":     (1.0 / r).mean().item(),
        "hits@1":  (r <= 1).float().mean().item(),
        "hits@3":  (r <= 3).float().mean().item(),
        "hits@10": (r <= 10).float().mean().item(),
    }
