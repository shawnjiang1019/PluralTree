"""Filtered link prediction evaluation: MRR and Hits@k.

For each test triple (s, r, o):
  1. Score (s, r, o) against all type-valid candidate objects
  2. Remove known positives from the candidate set (filtered setting)
  3. Rank the correct object among remaining candidates
  4. Compute MRR and Hits@{1, 3, 10}
"""

from __future__ import annotations

import torch
from torch import Tensor

from data.culturalbench import CulturalGraph
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

    ranks: list[int] = []

    for s_id, r_id, o_id in triples:
        # Get all type-valid candidates for this relation
        candidate_ids = torch.tensor(
            graph.type_constraints.get(r_id, []), dtype=torch.long, device=device
        )
        if candidate_ids.numel() == 0:
            continue

        h_s = h_all[s_id]               # (d,)
        h_cands = h_all[candidate_ids]  # (K, d)

        scores = predictor.score_all_candidates(h_s, r_id, h_cands)  # (K,)

        # Filtered ranking: remove known positives except the query triple itself
        correct_score = scores[candidate_ids.tolist().index(o_id)].item() \
            if o_id in candidate_ids.tolist() else float("-inf")

        rank = 1
        for i, (cid, score) in enumerate(zip(candidate_ids.tolist(), scores.tolist())):
            if cid == o_id:
                continue
            # Skip known positives (filtered setting)
            if (s_id, r_id, cid) in neg_sampler.known_positives:
                continue
            if score > correct_score:
                rank += 1

        ranks.append(rank)

    if not ranks:
        return {"mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0}

    mrr    = sum(1.0 / r for r in ranks) / len(ranks)
    hits1  = sum(1 for r in ranks if r <= 1)  / len(ranks)
    hits3  = sum(1 for r in ranks if r <= 3)  / len(ranks)
    hits10 = sum(1 for r in ranks if r <= 10) / len(ranks)

    return {
        "mrr":     mrr,
        "hits@1":  hits1,
        "hits@3":  hits3,
        "hits@10": hits10,
    }
