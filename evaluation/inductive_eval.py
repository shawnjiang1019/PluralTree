"""GraIL-style inductive evaluation: Hits@10 vs 50 sampled negatives + AUC-PR.

This matches the original GraIL protocol (Teru et al., ICML 2020): for each test
triple, rank the true tail against ``n_neg`` randomly sampled (filtered) negative
tails and report Hits@10; AUC-PR is computed over the positives vs. one sampled
negative each. Use alongside the vectorized ``evaluate_link_prediction`` (filtered
MRR / Hits@k against all entities) for the modern protocol.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor


def _auc_pr(scores: Tensor, labels: Tensor) -> float:
    """Average precision (area under the precision-recall curve)."""
    order = torch.argsort(scores, descending=True)
    lab = labels[order].float()
    tp = torch.cumsum(lab, dim=0)
    fp = torch.cumsum(1.0 - lab, dim=0)
    precision = tp / (tp + fp).clamp(min=1e-12)
    total_pos = lab.sum().clamp(min=1e-12)
    recall = tp / total_pos
    recall_prev = torch.cat([torch.zeros(1), recall[:-1]])
    return float(((recall - recall_prev) * precision).sum())


def grail_eval(
    h_all: Tensor,
    triples: list[tuple[int, int, int]],
    predictor,
    graph,
    n_neg: int = 50,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, float]:
    """Hits@10 / MRR against n_neg sampled negatives, plus AUC-PR.

    Args:
        h_all: (N, d) embeddings for the inductive graph's entities.
        triples: held-out test triples (s, r, o) to rank.
        predictor: scoring head (score_all_candidates).
        graph: the inductive CulturalGraph (for candidates + filtering).
        n_neg: negatives per triple (GraIL uses 50).
    """
    device = torch.device(device)
    h_all = h_all.to(device)
    g = torch.Generator().manual_seed(seed)

    true_objs: dict[tuple[int, int], set[int]] = defaultdict(set)
    for s, r, o in graph.all_triples:
        true_objs[(s, r)].add(o)

    # All relations share one candidate list on WN18RR (no type system).
    any_rel = next(iter(graph.relation_vocab.values()))
    cand_ids = graph.type_constraints.get(any_rel, [])
    cand = torch.tensor(cand_ids, dtype=torch.long, device=device)
    n_cand = cand.numel()

    ranks: list[int] = []
    pos_scores: list[float] = []
    neg_scores: list[float] = []

    for s, r, o in triples:
        # sample n_neg filtered negative tails
        negs: list[int] = []
        tries = 0
        while len(negs) < n_neg and tries < n_neg * 50:
            j = int(torch.randint(n_cand, (1,), generator=g).item())
            e = int(cand[j].item())
            if e != o and e not in true_objs[(s, r)]:
                negs.append(e)
            tries += 1
        if not negs:
            continue
        cset = torch.tensor([o] + negs, dtype=torch.long, device=device)
        scores = predictor.score_all_candidates(h_all[s], r, h_all[cset])  # (1+|negs|,)
        true_score = scores[0]
        rank = int((scores[1:] > true_score).sum().item()) + 1
        ranks.append(rank)
        pos_scores.append(float(true_score))
        neg_scores.append(float(scores[1]))  # one negative per positive for AUC-PR

    if not ranks:
        return {}

    r_t = torch.tensor(ranks, dtype=torch.float)
    auc_pr = _auc_pr(
        torch.tensor(pos_scores + neg_scores),
        torch.tensor([1.0] * len(pos_scores) + [0.0] * len(neg_scores)),
    )
    return {
        "hits@10_50neg": (r_t <= 10).float().mean().item(),
        "mrr_50neg":     (1.0 / r_t).mean().item(),
        "auc_pr":        auc_pr,
        "n":             float(len(ranks)),
    }
