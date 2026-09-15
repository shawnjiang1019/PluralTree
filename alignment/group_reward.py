"""Group-diversity reward for GRPO: credit each rollout for what the rest of its group missed.

WHY ACROSS SAMPLES. The largest measured effect in this project is not inside one
answer: union coverage across separately generated answers is 0.66-0.69 while any
single answer averages ~0.51, and merge_v2 already turns several drafts into one
answer. A policy whose rollouts COMPLEMENT each other feeds that directly. The
single-answer coverage_reward ranked same-question answers at 0.15 concordance with
the judge (chance 0.5), and its cluster-shaped fix needs targets that exist only for
the 60 eval questions -- so this reward uses no targets at all.

THE REWARD. For G rollouts to one prompt:

    pool        units of every rollout, greedily clustered across the group
                (bestofk_selection.pool_positions, min_support=1: a viewpoint one
                rollout voices must survive -- that is what novelty is for)
    C_i         pool clusters rollout i ARTICULATES: some unit within sim_thr, and
                >= min_depth_words of its units assigned there (reward.py's depth
                rule, so name-dropping a viewpoint does not count)
    q_i         |C_i| / |pool|                        within-answer substance
    n_i         mode "group":     |C_i minus U_{j!=i} C_j| / |pool|   leave-one-out
                mode "reference": |C_i minus U ref C_r|   / |pool|    vs frozen base
    r_i         q_i * (1 + lambda_div * n_i)

Multiplicative, like coverage_reward: novelty scales substance, it cannot stand in
for it. lambda_div=0 is the no-diversity control the gate compares against.

WHAT EACH MODE CAN AND CANNOT DO. "group" is the marginal contribution to the
group's union, the quantity union@K rewards -- but two rollouts that find the SAME
new viewpoint both score n=0, so a shared discovery is not credited. "reference"
scores novelty against samples of the frozen base model for the same prompt, which
credits shared discoveries and rewards leaving the base model's modes. Neither mode
can separate IDENTICAL rollouts: equal rewards give a zero group-relative advantage
whatever the reward is, so a fully collapsed group carries no gradient.

UNVALIDATED. Nothing here has been checked against the judge. The gate is
scripts/analysis/group_reward_gate.py; do not train on this until it passes.

    python -m alignment.group_reward --selftest
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignment.reward import EmbedFn                                  # noqa: E402
from scripts.analysis.bestofk_selection import (Cluster, embed_units,  # noqa: E402
                                                pool_positions)

MODES = ("group", "reference")


@dataclass(frozen=True)
class GroupRewardConfig:
    sim_thr: float = 0.55         # cosine that both builds pool clusters and marks
    #                               a unit as expressing one (bestofk's default)
    min_depth_words: int = 0      # words a rollout must spend on a cluster for it to
    #                               count; swept {0, 30} by the gate, not assumed
    lambda_div: float = 1.0       # novelty weight; 0 = within-answer coverage only
    mode: str = "group"           # "group" (leave-one-out) or "reference"
    max_units: int = 40


def covered_sets(units: list[list[str]], U: list[np.ndarray],
                 clusters: list[Cluster], cfg: GroupRewardConfig) -> list[set[int]]:
    """Per response, the pool clusters it articulates (mentioned AND deep enough).

    Mention and depth use different assignment rules, as in reward.py: a cluster
    is mentioned by any unit within sim_thr, but a unit's words accrue only to its
    single best-matching cluster.
    """
    if not clusters:
        return [set() for _ in U]
    C = np.stack([c.centroid for c in clusters])
    out = []
    for us, V in zip(units, U):
        if not us or not V.size:
            out.append(set())
            continue
        S = V @ C.T
        mentioned = S.max(axis=0) >= cfg.sim_thr
        depth = np.zeros(len(clusters))
        for k, (j, s) in enumerate(zip(S.argmax(axis=1), S.max(axis=1))):
            if s >= cfg.sim_thr:
                depth[j] += len(us[k].split())
        out.append({int(j) for j in np.flatnonzero(mentioned & (depth >= cfg.min_depth_words))})
    return out


def group_rewards(completions: Sequence[str], embed_fn: EmbedFn,
                  cfg: GroupRewardConfig = GroupRewardConfig(),
                  ref: Sequence[str] | None = None,
                  pre: tuple[list[list[str]], list[np.ndarray]] | None = None
                  ) -> tuple[list[float], list[dict]]:
    """Rewards for ONE group (all completions answer the same prompt).

    ``ref`` are frozen-base samples for the same prompt. Their units join the pool,
    so a viewpoint only the base model voiced is still a cluster; in "reference"
    mode, clusters any ref sample covers are not novel. ``pre`` is embed_units over
    ``list(completions) + list(ref)`` in that order, to score many configs from one
    embedding pass.
    """
    if cfg.mode not in MODES:
        raise ValueError(f"mode {cfg.mode!r} not in {MODES}")
    if cfg.mode == "reference" and not ref:
        raise ValueError("mode 'reference' needs ref samples")
    G = len(completions)
    texts = list(completions) + list(ref or [])
    units, U = pre if pre is not None else embed_units(texts, embed_fn, cfg.max_units)
    clusters = pool_positions(texts, embed_fn, cfg.sim_thr, min_support=1,
                              max_units=cfg.max_units, pre=(units, U))
    cov = covered_sets(units, U, clusters, cfg)
    grp, ref_cov = cov[:G], cov[G:]
    ref_union = set().union(*ref_cov) if ref_cov else set()
    n_pool = len(clusters)

    rewards, breakdowns = [], []
    for i, C_i in enumerate(grp):
        if cfg.mode == "group":
            others = set().union(*(grp[j] for j in range(G) if j != i))
            unique = C_i - others
        else:
            unique = C_i - ref_union
        q = len(C_i) / n_pool if n_pool else 0.0
        n = len(unique) / n_pool if n_pool else 0.0
        rewards.append(q * (1.0 + cfg.lambda_div * n))
        breakdowns.append({"q": q, "novelty": n, "n_covered": len(C_i),
                           "n_unique": len(unique), "n_pool": n_pool,
                           "n_units": len(units[i])})
    return rewards, breakdowns


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
_TOPICS = ("taxes", "guns", "climate", "health")


def _topic_embed(texts):
    """One axis per topic keyword, plus a spare axis for topic-free text. Distinct
    topics are orthogonal, so clustering is exact and the test checks the reward
    logic rather than an embedder."""
    V = np.zeros((len(texts), len(_TOPICS) + 1))
    for i, t in enumerate(texts):
        low = (t or "").lower()
        for k, w in enumerate(_TOPICS):
            V[i, k] = low.count(w)
        if not V[i].any():
            V[i, -1] = 1.0
        V[i] /= np.linalg.norm(V[i])
    return V


def _deep(topic: str, n: int = 3) -> str:
    """n ten-word sentences about one topic, one per line (split_units keeps
    lines of >= 8 words when there are at least 3)."""
    return "\n".join(f"People who care about {topic} argue this point number {k} strongly."
                     for k in range(n))


def _selftest() -> None:
    cfg = GroupRewardConfig(lambda_div=1.0)

    # 1. collapsed group: one cluster, nobody unique, identical rewards
    r, bd = group_rewards([_deep("taxes")] * 4, _topic_embed, cfg)
    assert bd[0]["n_pool"] == 1 and all(b["n_unique"] == 0 for b in bd), bd
    assert len(set(r)) == 1, f"collapsed group must score identically: {r}"

    # 2. complementary: the only rollout on guns out-scores the two sharing taxes
    r, bd = group_rewards([_deep("taxes"), _deep("guns"), _deep("taxes")], _topic_embed, cfg)
    assert r[1] > r[0] == r[2], r
    assert bd[1]["n_unique"] == 1 and bd[0]["n_unique"] == 0, bd

    # lambda_div = 0 removes the novelty credit: equal breadth -> equal reward
    r0, _ = group_rewards([_deep("taxes"), _deep("guns"), _deep("taxes")], _topic_embed,
                         GroupRewardConfig(lambda_div=0.0))
    assert r0[0] == r0[1] == r0[2], r0

    # 3. depth: naming all four topics in one short sentence each loses to one deep topic
    namedrop = "\n".join(f"Some people mention {t} briefly here today." for t in _TOPICS)
    deep_cfg = GroupRewardConfig(min_depth_words=20)
    r, bd = group_rewards([namedrop, _deep("climate")], _topic_embed, deep_cfg)
    assert bd[0]["n_covered"] == 0, f"8-word mentions must not clear depth 20: {bd[0]}"
    assert r[1] > r[0], r

    # 4. reference mode credits leaving the base model's modes, even when shared
    ref = [_deep("taxes"), _deep("taxes")]
    ref_cfg = GroupRewardConfig(mode="reference")
    r, bd = group_rewards([_deep("climate"), _deep("taxes")], _topic_embed, ref_cfg, ref=ref)
    assert r[0] > r[1] and bd[1]["novelty"] == 0.0, (r, bd)
    r_ref, bd_ref = group_rewards([_deep("climate")] * 2, _topic_embed, ref_cfg, ref=ref)
    r_grp, bd_grp = group_rewards([_deep("climate")] * 2, _topic_embed, cfg)
    assert all(b["novelty"] > 0 for b in bd_ref), "a shared discovery is novel vs the base"
    assert all(b["novelty"] == 0 for b in bd_grp), "...but not leave-one-out"

    # 5. degenerate inputs
    r, bd = group_rewards(["", "too short"], _topic_embed, cfg)
    assert r == [0.0, 0.0], r
    try:
        group_rewards([_deep("taxes")], _topic_embed, ref_cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("reference mode without ref must raise")

    print("group_reward self-test OK (collapse, complement, lambda=0, depth, "
          "reference vs leave-one-out, degenerate)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="group-diversity GRPO reward")
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        _selftest()
    else:
        ap.error("--selftest")
