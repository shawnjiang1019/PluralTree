"""Representativeness metrics for the plurality track (GlobalOpinionQA / OpinionQA).

Unlike link prediction, the target here is a *distribution over answer options per
group* (country), with no single correct answer — different groups legitimately
disagree (see ``docs/EXPERIMENTS.md`` §E4). We score how faithfully a model
reproduces each group's distribution, following Durmus et al. 2023 ("Towards
Measuring the Representation of Subjective Global Opinions in Language Models").

The core quantity is the **Jensen-Shannon distance** between the model's predicted
option distribution and the human distribution for a question, turned into a
similarity ``1 - JSdist`` and averaged per country:

    R_c = mean_{q in Q_c} ( 1 - sqrt(JSD(P_model(q) || P_country_c(q))) )       in [0, 1]

higher = more representative of country c.

This module is *intrinsic to the distributions*: like ``structure_metrics`` takes a
frozen ``h_all``, this takes already-computed option distributions. How the
predicted distribution is produced (a distributional head over the per-(question,
country) existence ``h_{q|c}``) and how the human targets are loaded (the survey
``selections`` field) live elsewhere — this file only compares distributions.

Each question may have a different number of options, so inputs are *lists* of
1-D tensors (one per question), not a single padded matrix. ``pred[c][i]`` must be
aligned with ``target[c][i]``: same question, same option ordering, both summing
to 1.

Returned keys:
    repr_macro   mean over countries of R_c (each country weighted equally).
    repr_micro   mean over all (question, country) pairs (large countries dominate).
    js_macro     mean JS *distance* over countries (= 1 - repr_macro view; lower better).
    n_countries  number of countries with at least one scored question.
    n_pairs      total number of (question, country) pairs scored.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Jensen-Shannon divergence / distance (base 2, so JSD in [0, 1])
# ---------------------------------------------------------------------------
def _kl_nats(p: Tensor, q: Tensor, eps: float) -> Tensor:
    """KL(p || q) in nats, summed over the last dim. xlogy(0, ·) = 0 handles p=0."""
    q = q.clamp_min(eps)
    return torch.xlogy(p, p / q).sum(dim=-1)


def js_divergence(p: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    """Jensen-Shannon divergence in bits (base 2), shape (...) over batched rows.

    p, q: (..., O) distributions over the same O options (each row sums to 1).
    Returns JSD in [0, 1].
    """
    m = 0.5 * (p + q)
    jsd_nats = 0.5 * _kl_nats(p, m, eps) + 0.5 * _kl_nats(q, m, eps)
    return (jsd_nats / _LN2).clamp_min(0.0)


def js_distance(p: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    """JS distance = sqrt(JSD), a true metric in [0, 1]."""
    return js_divergence(p, q, eps).sqrt()


# ---------------------------------------------------------------------------
# Representativeness
# ---------------------------------------------------------------------------
def compute_representativeness(
    pred: dict[int, list[Tensor]],
    target: dict[int, list[Tensor]],
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Representativeness of predicted distributions against human survey targets.

    Args:
        pred:   country_id -> list of predicted option distributions (1-D, sum 1).
        target: country_id -> list of human option distributions, aligned 1:1 with
                ``pred[country_id]`` (same question, same option ordering).
        eps:    numerical floor for the KL terms.

    Returns:
        dict of the aggregate metrics documented in the module docstring.
    """
    per_country = representativeness_per_country(pred, target, eps=eps)

    if not per_country:
        return {"repr_macro": float("nan"), "repr_micro": float("nan"),
                "js_macro": float("nan"), "n_countries": 0, "n_pairs": 0}

    # macro: each country equal weight
    repr_macro = sum(per_country.values()) / len(per_country)

    # micro: weight by number of questions per country
    total_sim, n_pairs = 0.0, 0
    for c in pred:
        if c not in target:
            continue
        for p, t in zip(pred[c], target[c]):
            total_sim += 1.0 - float(js_distance(p, t, eps))
            n_pairs += 1
    repr_micro = total_sim / n_pairs if n_pairs else float("nan")

    return {
        "repr_macro":  repr_macro,
        "repr_micro":  repr_micro,
        "js_macro":    1.0 - repr_macro,
        "n_countries": len(per_country),
        "n_pairs":     n_pairs,
    }


def representativeness_per_country(
    pred: dict[int, list[Tensor]],
    target: dict[int, list[Tensor]],
    *,
    eps: float = 1e-12,
) -> dict[int, float]:
    """Per-country R_c = mean_q (1 - JSdist). Useful for ranking which groups the
    model represents well vs. poorly (the paper's per-country breakdown)."""
    out: dict[int, float] = {}
    for c, preds in pred.items():
        if c not in target or not preds:
            continue
        sims = [1.0 - float(js_distance(p, t, eps))
                for p, t in zip(preds, target[c])]
        if sims:
            out[c] = sum(sims) / len(sims)
    return out
