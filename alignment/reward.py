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
    """Defaults for coverage_reward (v2). See coverage_reward_v1 for the original.

    Three corrections over v1, each traced to a measured failure:

    weight="uniform"     v1 weighted recall by prevalence, but OvertonScore is an
                         UNWEIGHTED mean over clusters (judge_overtonbench.py:
                         `covered / len(cluster_ratings)`). Prevalence weighting
                         systematically favors majority subgroups -- backwards for
                         pluralism, and it compounds with the fact that minority
                         positions may also be the expensive ones to articulate.
    min_depth_words      v1 marked a position "expressed" on a single unit clearing
                         cosine 0.50, so a one-clause namedrop scored like a full
                         articulation. The `route` condition did exactly that and
                         scored 0.072 vs baseline 0.507 -- i.e. v1 CANNOT SEE the
                         failure mode that dominates this benchmark. Coverage
                         requires depth per position, so expressing a position now
                         also requires min_depth_words of text about it.
    multiplicative       v1 subtracted penalties additively. GR3 (arXiv:2603.10535)
                         shows additive length/quality penalties create a
                         compensatory effect -- the policy inflates one term to pay
                         for another. Factors in [0,1] cannot be compensated away.
    """
    match_thr: float = 0.50       # cosine >= thr counts a position as mentioned
    min_depth_words: int = 60     # ...AND this many words about it to count as covered.
    #                               PROVISIONAL: route spent ~30 words/position and
    #                               covered nothing; baseline ~110-165 and covered ~half.
    #                               Fit properly against OvertonBench's own human-rated
    #                               reference responses (judge-free) before trusting the
    #                               absolute scale -- see fit_depth_threshold() docstring.
    weight: str = "uniform"       # "uniform" (matches the eval) or "prevalence"
    l_precision: float = 0.50     # exponent on the precision factor
    l_verbose: float = 0.0        # OFF by default -- see below. >0 re-enables it.
    #   v1 used this to encode "commit on consensus". Two reasons it is off now:
    #   (1) OvertonScore is MONOTONE in cluster coverage -- covering 3/3 clusters
    #       scores 1.0 however lopsided the population is -- so the eval cannot
    #       reward commitment, and a reward that does will train away from it.
    #   (2) It is driven by GRAPH prevalence entropy, but a skewed survey option
    #       distribution does not imply few human viewpoint clusters (the same
    #       graph-vs-human gap that measured corr +0.20). Suppressing breadth on
    #       that basis penalizes coverage the eval would have credited.
    #   Its anti-spam role is now covered by min_depth_words: enumerate ten
    #   positions shallowly and none of them clears the depth bar.
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
def coverage_reward_v1(response: str, positions: list[Position], embed_fn: EmbedFn,
                       cfg: RewardConfig = RewardConfig()) -> tuple[float, dict]:
    """ORIGINAL reward. Kept only for comparability with already-computed scores.

    Do NOT train on this: it is depth-blind (a one-clause mention scores like a
    full articulation), prevalence-weighted where the eval is uniform, and uses
    additive penalties. See RewardConfig for the evidence on each. Use
    coverage_reward() instead.
    """
    units = split_units(response, cfg.max_units)
    if not units or not positions:
        return 0.0, {"recall": 0.0, "precision": 0.0, "verbosity_penalty": 0.0,
                     "n_units": len(units), "n_expressed": 0, "target": 0.0}

    U = np.asarray(embed_fn(units), dtype=float)
    P = np.asarray(embed_fn([p.embed_text for p in positions]), dtype=float)
    sim = U @ P.T                                    # (units, positions), both unit-norm
    prev = np.array([p.prevalence for p in positions])

    pos_best = sim.max(axis=0)                        # best unit per position
    expressed = pos_best >= cfg.match_thr
    recall_w = float((prev * expressed).sum() / prev.sum())

    unit_best = sim.max(axis=1)
    precision = float((unit_best >= cfg.match_thr).mean())

    n_expr = int(expressed.sum())
    target = _effective_positions(prev)
    verbosity_penalty = max(0.0, n_expr - target) / target if target > 0 else 0.0

    reward = recall_w - 0.20 * (1.0 - precision) - 0.30 * verbosity_penalty
    reward = float(min(1.0, max(0.0, reward)))
    return reward, {"recall": recall_w, "precision": precision,
                    "verbosity_penalty": verbosity_penalty, "n_units": len(units),
                    "n_expressed": n_expr, "target": target}


def position_depths(units: list[str], sim: np.ndarray, cfg: RewardConfig) -> np.ndarray:
    """Words of text primarily ABOUT each position.

    Each unit is assigned to its single best-matching position (argmax), and only
    if that match clears match_thr; its word count accrues to that position. Note
    this is a different rule from `pos_best`: a position can be *mentioned* by a
    unit whose argmax lies elsewhere. Mentioned != covered -- that distinction is
    the whole point of the depth term.
    """
    depths = np.zeros(sim.shape[1])
    if not len(units):
        return depths
    assign = sim.argmax(axis=1)
    best = sim.max(axis=1)
    for u_i, (p_i, s) in enumerate(zip(assign, best)):
        if s >= cfg.match_thr:
            depths[p_i] += len(units[u_i].split())
    return depths


def coverage_reward(response: str, positions: list[Position], embed_fn: EmbedFn,
                    cfg: RewardConfig = RewardConfig()) -> tuple[float, dict]:
    """Graph-grounded pluralism reward in [0, 1] + a breakdown dict.

        covered_p = (best-unit cosine >= match_thr) AND (depth_p >= min_depth_words)
        recall    = weighted fraction of positions covered (uniform by default)
        reward    = recall * precision^l_prec * 1/(1 + l_verb * verbosity_penalty)

    Multiplicative by construction, so a weak factor cannot be compensated by
    inflating another (GR3, arXiv:2603.10535), and the result stays in [0,1]
    without clamping. A response with no scorable units gets 0.0.
    """
    units = split_units(response, cfg.max_units)
    if not units or not positions:
        return 0.0, {"recall": 0.0, "precision": 0.0, "verbosity_penalty": 0.0,
                     "n_units": len(units), "n_expressed": 0, "target": 0.0,
                     "n_mentioned": 0, "mean_depth": 0.0}

    U = np.asarray(embed_fn(units), dtype=float)
    P = np.asarray(embed_fn([p.embed_text for p in positions]), dtype=float)
    sim = U @ P.T                                    # (units, positions), both unit-norm
    prev = np.array([p.prevalence for p in positions])

    # covered = mentioned AND articulated. `route` (0.072) satisfied only the first.
    mentioned = sim.max(axis=0) >= cfg.match_thr
    depths = position_depths(units, sim, cfg)
    expressed = mentioned & (depths >= cfg.min_depth_words)

    # uniform by default: OvertonScore counts every cluster equally, so weighting
    # by prevalence here would train the policy to serve the majority.
    w = np.ones_like(prev) if cfg.weight == "uniform" else prev
    recall = float((w * expressed).sum() / w.sum())

    # precision: fraction of units landing on SOME real position (anti-padding)
    precision = float((sim.max(axis=1) >= cfg.match_thr).mean())

    # verbosity: expressing more positions than the distribution supports
    n_expr = int(expressed.sum())
    target = _effective_positions(prev)
    verbosity_penalty = max(0.0, n_expr - target) / target if target > 0 else 0.0

    reward = float(recall
                   * (precision ** cfg.l_precision)
                   / (1.0 + cfg.l_verbose * verbosity_penalty))
    return reward, {"recall": recall, "precision": precision,
                    "verbosity_penalty": verbosity_penalty, "n_units": len(units),
                    "n_expressed": n_expr, "target": target,
                    "n_mentioned": int(mentioned.sum()),
                    "mean_depth": float(depths[mentioned].mean()) if mentioned.any() else 0.0}


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
                # deterministic, collision-free: each distinct word gets its own
                # dimension, so option tokens can never alias onto each other
                V[i, vocab.setdefault(w, len(vocab) % 64)] += 1.0
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

    def para(opt, n):
        """One position articulated at ~n words (the option token repeated)."""
        return " ".join([opt] * n) + "."

    # deep = each position genuinely articulated; route_like = all three NAMED but
    # none articulated -- exactly the pattern that scored 0.072 on OvertonBench.
    cfg = RewardConfig(min_depth_words=20)
    deep = "\n".join(para(o, 30) for o in ("alpha", "beta", "gamma"))
    route_like = "\n".join(para(o, 8) for o in ("alpha", "beta", "gamma"))
    narrow_deep = para("alpha", 40)

    r_deep = coverage_reward(deep, contested, fake_embed, cfg)[0]
    r_route, bd_route = coverage_reward(route_like, contested, fake_embed, cfg)
    r_narrow = coverage_reward(narrow_deep, contested, fake_embed, cfg)[0]

    # THE fix: v1 cannot tell route_like from deep (both "express" all 3);
    # v2 must, because route_like never clears the depth bar.
    v1_deep = coverage_reward_v1(deep, contested, fake_embed, cfg)[0]
    v1_route = coverage_reward_v1(route_like, contested, fake_embed, cfg)[0]
    assert abs(v1_deep - v1_route) < 1e-9, (v1_deep, v1_route)   # v1: identical
    assert r_deep > r_route, (r_deep, r_route)                   # v2: separated
    assert bd_route["n_mentioned"] == 3 and bd_route["n_expressed"] == 0, bd_route
    assert r_deep > r_narrow, (r_deep, r_narrow)                 # breadth still pays

    # Consensus/skewed question: the eval is MONOTONE in cluster coverage, so
    # covering all three must still beat covering one. (v1 inverted this via its
    # verbosity penalty -- see RewardConfig.l_verbose for why that was wrong.)
    consensus = [pos("alpha", 0.90), pos("beta", 0.06), pos("gamma", 0.04)]
    r_deep_c = coverage_reward(deep, consensus, fake_embed, cfg)[0]
    r_narrow_c = coverage_reward(narrow_deep, consensus, fake_embed, cfg)[0]
    assert r_deep_c > r_narrow_c, (r_deep_c, r_narrow_c)

    # Bug 1: prevalence weighting over-credits covering ONLY the majority option;
    # uniform weighting (what the eval does) does not.
    skew_cfg = RewardConfig(min_depth_words=20, weight="prevalence")
    p = coverage_reward(narrow_deep, consensus, fake_embed, skew_cfg)[0]
    u = coverage_reward(narrow_deep, consensus, fake_embed, cfg)[0]
    assert p > u, (p, u)
    assert abs(u - 1 / 3) < 0.05, u        # uniform: 1 of 3 positions covered
    assert p > 0.85, p                     # prevalence: ~0.90 for the same answer


    # advantage sanity
    from alignment.advantage import group_relative_advantage
    adv = group_relative_advantage([0.2, 0.8, 0.5, 0.5])
    assert abs(sum(adv)) < 1e-6 and adv[1] == max(adv)
    print("reward self-test OK")
    print(f"  contested : deep {r_deep:.3f} > narrow {r_narrow:.3f} "
          f"> route-like {r_route:.3f}")
    print(f"  skewed    : deep {r_deep_c:.3f} > narrow {r_narrow_c:.3f} "
          f"(eval is monotone in coverage; v1 inverted this)")
    print(f"  weighting : uniform {u:.3f} vs prevalence {p:.3f} on a "
          f"majority-only answer")
    print(f"  depth fix : v1 scores deep and route-like IDENTICALLY "
          f"({v1_deep:.3f}); v2 separates {r_deep:.3f} vs {r_route:.3f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Graph-grounded pluralism reward")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.error("nothing to do; pass --selftest (or import the module)")
