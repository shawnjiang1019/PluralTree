"""Does the GRPO reward rank answers the way the OvertonBench judge does?

This is the experiment that gates the whole RL phase, and it needs no GPU: if
`coverage_reward` cannot order two answers to the SAME question the way the judge
does, then GRPO's advantage -- which is computed strictly within a group of
rollouts sharing one prompt -- is being driven by noise, and training will
optimize something unrelated to the metric.

WITHIN-QUESTION, NOT POOLED. A pooled correlation across all (question,
condition) rows can look healthy purely because some questions are easier than
others, while the within-question ordering is chance. That exact distinction bit
the judge validation itself (pooled rho 0.673 vs within-participant 0.059), so
this script reports both and treats the within-question number as the real one.

PAIRWISE CONCORDANCE is the primary statistic. With only 3-4 conditions per
question a within-question Spearman takes very few values; concordance over
condition PAIRS (does reward order A,B like the judge does?) pools cleanly across
questions and has an unambiguous chance level of 0.5.

Also scores coverage_reward_v1 alongside v2, so the depth/weighting/shape fixes
can be judged on whether they actually improve agreement with the judge.

Positions come from the anchor the scout actually chose, recovered from the
stored `fork_context` -- so this needs no embeddings, only the graph.

    OPINIONQA_DIR=... python scripts/analysis/reward_eval_correlation.py \
        --responses overton_responses_v5.jsonl --scores overton_scores_v5.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# "[fork 1] at 'How POLIDEOLOGY divides opinion on: On balance, do you thin…'"
_ANCHOR_RE = re.compile(r"\[fork \d+\] (?:at|on) '(.+?)'", re.DOTALL)


def parse_anchor_text(fork_context: str) -> str | None:
    """Anchor description of the TOP fork (truncated to 60 chars by describe_node)."""
    m = _ANCHOR_RE.search(fork_context or "")
    if not m:
        return None
    return m.group(1).rstrip("…").rstrip()


def build_anchor_index(graph) -> list[tuple[str, int]]:
    """(entity_text, node_id) for axis nodes, for prefix matching."""
    out = []
    for nid, name in enumerate(graph.id_to_entity):
        if name.startswith("ax:"):
            out.append((graph.entity_text.get(nid, ""), nid))
    return out


def resolve_anchor(prefix: str, index: list[tuple[str, int]]) -> int | None:
    for text, nid in index:
        if text.startswith(prefix):
            return nid
    return None


def _concordance(pairs: list[tuple[float, float]]) -> tuple[float, int]:
    """Fraction of (judge_delta, reward_delta) pairs that agree in sign.

    Pairs where the judge is tied carry no information and are dropped; pairs
    where the REWARD is tied count as disagreement (a reward that cannot
    separate two answers gives GRPO nothing).
    """
    used = [(j, r) for j, r in pairs if abs(j) > 1e-9]
    if not used:
        return float("nan"), 0
    agree = sum(1 for j, r in used if (j > 0) == (r > 0) and abs(r) > 1e-9)
    return agree / len(used), len(used)


def _corr(xs, ys) -> float:
    if len(xs) < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Reward vs judge agreement")
    ap.add_argument("--responses", default="overton_responses_v5.jsonl")
    ap.add_argument("--scores", default="overton_scores_v5.csv")
    ap.add_argument("--seed", type=int, default=42, help="graph split seed")
    ap.add_argument("--min_depth_words", type=int, default=None,
                    help="override RewardConfig.min_depth_words (sweep this)")
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--exclude", default="",
                    help="comma-separated conditions to drop. Run v6 BOTH ways: "
                         "with `route` included the reward only has to notice a "
                         "0.4 collapse (easy); excluding it leaves the near-ties, "
                         "which is the regime GRPO actually operates in -- every "
                         "rollout in a group comes from the SAME policy and they "
                         "resemble each other far more than baseline resembles route")
    ap.add_argument("--out", default="docs/reward_eval_correlation.csv")
    args = ap.parse_args()
    drop = {c.strip() for c in args.exclude.split(",") if c.strip()}

    from alignment.reward import (RewardConfig, coverage_reward,
                                  coverage_reward_v1, default_embed_fn,
                                  positions_from_subtree)
    from data.loaders.opinionqa import load_opinionqa

    # --- load responses + judge scores -------------------------------------
    rows = [json.loads(l) for l in open(args.responses, encoding="utf-8")]
    resp = defaultdict(dict)          # qid -> cond -> response
    ctx = {}                          # qid -> fork_context (from any injected row)
    for r in rows:
        resp[r["question_id"]][r["condition"]] = r.get("response") or ""
        if r.get("fork_context") and r["question_id"] not in ctx:
            ctx[r["question_id"]] = r["fork_context"]

    cov = defaultdict(dict)
    for r in csv.DictReader(open(args.scores, encoding="utf-8")):
        cov[int(r["question_id"])][r["condition"]] = float(r["coverage"])

    graph = load_opinionqa(split_seed=args.seed, leakage_safe=True)
    index = build_anchor_index(graph)
    embed_fn = default_embed_fn(args.embedder)
    cfg = RewardConfig()
    if args.min_depth_words is not None:
        cfg.min_depth_words = args.min_depth_words
    print(f"reward cfg: min_depth_words={cfg.min_depth_words} weight={cfg.weight} "
          f"l_precision={cfg.l_precision} l_verbose={cfg.l_verbose}")

    # --- score every response with both rewards -----------------------------
    out_rows, n_noanchor = [], 0
    pairs_v1, pairs_v2 = [], []       # (judge_delta, reward_delta) within question
    pooled_j, pooled_r1, pooled_r2 = [], [], []
    per_q_pick = {"v1": 0, "v2": 0, "n": 0}

    for qid in sorted(resp):
        conds = [c for c in resp[qid] if c in cov.get(qid, {}) and c not in drop]
        if len(conds) < 2:
            continue
        anchor_txt = parse_anchor_text(ctx.get(qid, ""))
        anchor = resolve_anchor(anchor_txt, index) if anchor_txt else None
        if anchor is None:
            n_noanchor += 1
            continue
        positions = positions_from_subtree(graph, anchor)
        if len(positions) < 2:
            n_noanchor += 1
            continue

        scored = {}
        for c in conds:
            r2 = coverage_reward(resp[qid][c], positions, embed_fn, cfg)[0]
            r1 = coverage_reward_v1(resp[qid][c], positions, embed_fn, cfg)[0]
            scored[c] = (r1, r2)
            j = cov[qid][c]
            pooled_j.append(j); pooled_r1.append(r1); pooled_r2.append(r2)
            out_rows.append({"question_id": qid, "condition": c, "judge_coverage": j,
                             "reward_v1": r1, "reward_v2": r2,
                             "n_positions": len(positions), "anchor": anchor})

        for i in range(len(conds)):
            for k in range(i + 1, len(conds)):
                a, b = conds[i], conds[k]
                jd = cov[qid][a] - cov[qid][b]
                pairs_v1.append((jd, scored[a][0] - scored[b][0]))
                pairs_v2.append((jd, scored[a][1] - scored[b][1]))

        best_j = max(conds, key=lambda c: cov[qid][c])
        if len({cov[qid][c] for c in conds}) > 1:
            per_q_pick["n"] += 1
            if max(conds, key=lambda c: scored[c][0]) == best_j:
                per_q_pick["v1"] += 1
            if max(conds, key=lambda c: scored[c][1]) == best_j:
                per_q_pick["v2"] += 1

    n_q = len({r["question_id"] for r in out_rows})
    print(f"\nscored {len(out_rows)} responses over {n_q} questions "
          f"({n_noanchor} skipped: no resolvable anchor)")

    # --- the numbers --------------------------------------------------------
    c1, n1 = _concordance(pairs_v1)
    c2, n2 = _concordance(pairs_v2)
    print(f"\n=== WITHIN-QUESTION pairwise concordance (chance = 0.500) ===")
    print(f"  reward_v1 (depth-blind) {c1:.3f}   over {n1} condition pairs")
    print(f"  reward_v2 (fixed)       {c2:.3f}   over {n2} condition pairs")
    print("  ^ THIS is what GRPO's advantage consumes. At ~0.5 the reward cannot")
    print("    order rollouts of the same prompt and training optimizes noise.")

    print(f"\n=== picks the judge's best condition (chance = 1/n_conds) ===")
    if per_q_pick["n"]:
        print(f"  reward_v1 {per_q_pick['v1']}/{per_q_pick['n']} "
              f"({per_q_pick['v1']/per_q_pick['n']:.2f})   "
              f"reward_v2 {per_q_pick['v2']}/{per_q_pick['n']} "
              f"({per_q_pick['v2']/per_q_pick['n']:.2f})")

    print(f"\n=== POOLED correlation (for contrast -- NOT the relevant number) ===")
    print(f"  corr(reward_v1, judge) = {_corr(pooled_r1, pooled_j):+.3f}")
    print(f"  corr(reward_v2, judge) = {_corr(pooled_r2, pooled_j):+.3f}")
    print("  ^ can look fine purely from between-question difficulty variance")

    print(f"\n=== reward distributions (is the reward even discriminative?) ===")
    for name, vals in (("v1", pooled_r1), ("v2", pooled_r2), ("judge", pooled_j)):
        zeros = sum(1 for v in vals if v <= 1e-9) / max(1, len(vals))
        print(f"  {name:<6} mean={st.mean(vals):.3f} sd={st.pstdev(vals):.3f} "
              f"frac_zero={zeros:.2f}")

    if out_rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
