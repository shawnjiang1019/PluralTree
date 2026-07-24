"""Graph-grounded pluralism reward for the GRPO alignment phase.

The RL reward must NOT use the OvertonBench human judge -- those 60 questions are
held-out eval and would leak. Instead it is grounded in the graph's own subgroup
answer distributions (see docs/grpo_alignment.txt):

  positions_from_subtree(graph, anchor)
      walk the anchor's subtree to its opinion leaves; each leaf is one subgroup's
      distribution over the SAME survey options. Aggregate to a population
      distribution over positions -- prevalence(option) = mean subgroup prob.

  coverage_reward(response, positions, embed_fn, cfg)
      recall_w   prevalence-weighted fraction of real positions expressed
      precision  fraction of response units that map to a real position
      verbosity  penalty for expressing MORE positions than the distribution
                 supports (target = exp(entropy(prevalence))); this is the
                 `route` "commit on consensus" lesson turned into a gradient.
      reward = recall_w - L_prec*(1-precision) - L_verb*max(0,n_expr-target)/target

Everything here is pure and testable offline: embed_fn is injected (default lazily
loads the held-out mpnet embedder, deliberately NOT the scout's MiniLM, so the
reward does not credit retrieval-shaped text). Run the self-test:

    python -m alignment.reward --selftest
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

EmbedFn = Callable[[Sequence[str]], np.ndarray]      # texts -> (n, d) unit-normalized

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_BOLD = re.compile(r"\*+")


@dataclass(frozen=True)
class Position:
    """One population-level position: an answer option and its prevalence."""
    option: str          # short option label, for display/logging
    embed_text: str      # "<question> <option>" -- what we semantically match against
    prevalence: float    # mean subgroup probability of this option (pre-normalization)


@dataclass
class RewardConfig:
    match_thr: float = 0.50       # cosine >= thr counts a position as expressed
    l_precision: float = 0.20     # weight on the (1 - precision) penalty
    l_verbose: float = 0.30       # weight on the over-enumeration penalty
    min_prevalence: float = 0.05  # drop positions below this mean prob (noise floor)
    max_units: int = 40           # cap response units scored (bound cost)


# ---------------------------------------------------------------------------
# Positions from the graph
# ---------------------------------------------------------------------------
def _leaf_positions(texts: list[str], dist: list[float]) -> list[tuple[str, str, float]]:
    """(option, embed_text, prob) for one opinion leaf.

    opinion_texts stores '<question> <option>' per option; strip the shared
    question prefix to recover the option label, keep the full text for matching.
    """
    if not texts or not dist:
        return []
    pref = os.path.commonprefix(texts)
    out = []
    for t, p in zip(texts, dist):
        opt = t[len(pref):].strip(" \t\"'-") or t.strip()
        out.append((opt, t.strip(), float(p)))
    return out


def positions_from_subtree(graph, anchor: int,
                           min_prevalence: float = 0.05) -> list[Position]:
    """Aggregate the anchor subtree's opinion leaves into population positions.

    Descends children_indices from ``anchor`` to every opinion leaf (a node with
    an opinion_dist entry), then averages each option's probability across the
    subgroups to a population prevalence. Options come from the same survey
    question so they share a vocabulary; aggregation keys on the option label.
    """
    opinion_dist = getattr(graph, "opinion_dist", {})
    opinion_texts = getattr(graph, "opinion_texts", {})
    children = graph.children_indices

    agg: dict[str, list] = {}        # option -> [sum_prob, count, embed_text]
    stack, seen = [anchor], set()
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        if nid in opinion_dist:                       # opinion leaf
            for opt, emb, p in _leaf_positions(opinion_texts.get(nid, []),
                                               opinion_dist[nid]):
                slot = agg.setdefault(opt, [0.0, 0, emb])
                slot[0] += p
                slot[1] += 1
            continue
        if nid < len(children):
            stack.extend(children[nid])

    positions = []
    for opt, (s, c, emb) in agg.items():
        prev = s / c if c else 0.0
        if prev >= min_prevalence:
            positions.append(Position(option=opt, embed_text=emb, prevalence=prev))
    positions.sort(key=lambda p: p.prevalence, reverse=True)
    return positions


# ---------------------------------------------------------------------------
# Response -> units
# ---------------------------------------------------------------------------
def split_units(text: str, max_units: int = 40) -> list[str]:
    """Response -> position units (enumerated items / bullets, else sentences).

    Mirrors scripts/analysis/measure_pole_collapse.split_units so the reward and
    the diagnostic measure the same thing.
    """
    units = []
    for raw in (text or "").splitlines():
        s = _BOLD.sub("", _BULLET.sub("", raw.strip())).strip()
        if len(s.split()) >= 8:
            units.append(s)
    if len(units) < 3:
        units = [s.strip() for s in re.split(r"(?<=[.!?])\s+", _BOLD.sub("", text or ""))
                 if len(s.split()) >= 8]
    return units[:max_units]


def _effective_positions(prev: np.ndarray) -> float:
    """exp(Shannon entropy of the prevalence distribution) = how many positions
    the population actually spreads over. ~1 on consensus, larger when contested."""
    p = prev / prev.sum()
    p = p[p > 1e-12]
    return float(math.exp(-(p * np.log(p)).sum()))


# ---------------------------------------------------------------------------
# The reward
# ---------------------------------------------------------------------------
def coverage_reward(response: str, positions: list[Position], embed_fn: EmbedFn,
                    cfg: RewardConfig = RewardConfig()) -> tuple[float, dict]:
    """Graph-grounded pluralism reward in [0, 1] + a breakdown dict.

    Returns (reward, {recall, precision, verbosity_penalty, n_units, n_expressed,
    target}). A response with no scorable units gets 0.0.
    """
    units = split_units(response, cfg.max_units)
    if not units or not positions:
        return 0.0, {"recall": 0.0, "precision": 0.0, "verbosity_penalty": 0.0,
                     "n_units": len(units), "n_expressed": 0, "target": 0.0}

    U = np.asarray(embed_fn(units), dtype=float)
    P = np.asarray(embed_fn([p.embed_text for p in positions]), dtype=float)
    sim = U @ P.T                                    # (units, positions), both unit-norm
    prev = np.array([p.prevalence for p in positions])

    # recall: prevalence-weighted, a position is expressed if any unit matches it
    pos_best = sim.max(axis=0)                        # best unit per position
    expressed = pos_best >= cfg.match_thr
    recall_w = float((prev * expressed).sum() / prev.sum())

    # precision: fraction of units that land on SOME real position (anti-padding)
    unit_best = sim.max(axis=1)
    precision = float((unit_best >= cfg.match_thr).mean())

    # verbosity: expressing more positions than the distribution supports
    n_expr = int(expressed.sum())
    target = _effective_positions(prev)
    verbosity_penalty = max(0.0, n_expr - target) / target if target > 0 else 0.0

    reward = recall_w - cfg.l_precision * (1.0 - precision) \
        - cfg.l_verbose * verbosity_penalty
    reward = float(min(1.0, max(0.0, reward)))
    return reward, {"recall": recall_w, "precision": precision,
                    "verbosity_penalty": verbosity_penalty, "n_units": len(units),
                    "n_expressed": n_expr, "target": target}


def batch_rewards(responses: Sequence[str], positions: list[Position],
                  embed_fn: EmbedFn, cfg: RewardConfig = RewardConfig()) -> list[float]:
    """Reward for each response in a GRPO group (same question => same positions)."""
    return [coverage_reward(r, positions, embed_fn, cfg)[0] for r in responses]


# ---------------------------------------------------------------------------
# Default embedder (held-out mpnet; lazily loaded)
# ---------------------------------------------------------------------------
def default_embed_fn(model_name: str = "sentence-transformers/all-mpnet-base-v2") -> EmbedFn:
    """Held-out embedder for the reward -- NOT the scout's MiniLM (avoids crediting
    retrieval-shaped text). Loaded once, closed over."""
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model_name)

    def _embed(texts: Sequence[str]) -> np.ndarray:
        return enc.encode(list(texts), normalize_embeddings=True,
                          show_progress_bar=False, batch_size=64)
    return _embed


# ---------------------------------------------------------------------------
# Self-test (no model download, no graph, no GPU)
# ---------------------------------------------------------------------------
def _selftest() -> None:
    """Deterministic hashing embedder + a synthetic 3-position distribution.

    Checks the reward orders responses as the design requires:
      contested Q  -> covering all 3 positions beats covering 1
      consensus Q  -> committing to the 1 dominant position beats enumerating 3
    """
    rng = np.random.default_rng(0)
    vocab = {}

    def fake_embed(texts):
        # bag-of-words hashed to a fixed space, L2-normalized -> shared words = high cos
        V = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                V[i, vocab.setdefault(w, rng.integers(0, 64))] += 1.0
            n = np.linalg.norm(V[i])
            if n:
                V[i] /= n
        return V

    # embed_text is dominated by the distinctive option token (repeated) so the
    # bag-of-words stub matches on the option, not on a shared stem -- a real
    # embedder does this via token salience; the stub needs it made explicit.
    def pos(opt, prev):
        return Position(option=opt, embed_text=f"{opt} {opt} {opt} {opt}",
                        prevalence=prev)

    contested = [pos("alpha", 0.34), pos("beta", 0.33), pos("gamma", 0.33)]
    broad = ("Some people say alpha alpha is right here for sure. "
             "Others firmly hold that beta beta is the answer instead. "
             "A third group argues gamma gamma is clearly correct now.")
    narrow = "The chosen answer here is alpha alpha alpha without any doubt today."
    cfg = RewardConfig()
    r_broad = coverage_reward(broad, contested, fake_embed, cfg)[0]
    r_narrow = coverage_reward(narrow, contested, fake_embed, cfg)[0]
    assert r_broad > r_narrow, (r_broad, r_narrow)

    consensus = [pos("alpha", 0.90), pos("beta", 0.06), pos("gamma", 0.04)]
    r_broad_c = coverage_reward(broad, consensus, fake_embed, cfg)[0]
    r_narrow_c = coverage_reward(narrow, consensus, fake_embed, cfg)[0]
    assert r_narrow_c > r_broad_c, (r_narrow_c, r_broad_c)

    # advantage sanity
    from alignment.advantage import group_relative_advantage
    adv = group_relative_advantage([0.2, 0.8, 0.5, 0.5])
    assert abs(sum(adv)) < 1e-6 and adv[1] == max(adv)
    print("reward self-test OK")
    print(f"  contested:  broad {r_broad:.3f} > narrow {r_narrow:.3f}")
    print(f"  consensus:  narrow {r_narrow_c:.3f} > broad {r_broad_c:.3f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Graph-grounded pluralism reward")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.error("nothing to do; pass --selftest (or import the module)")
