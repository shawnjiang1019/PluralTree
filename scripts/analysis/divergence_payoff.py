"""Does injection pay off more where the retrieved subgroups actually disagree?

THE QUESTION. The learned geometry ranks real disagreement well (within-axis
Spearman +0.755 against Jensen-Shannon divergence between subgroup answer
distributions), yet every selection ablation ties. One explanation is that there
is simply not much disagreement to select: the mean JS over same-axis pairs is
0.0154 bits. But the scout injects the ARGMAX pair, not the average one, so that
mean does not settle it -- picking the best of many weak pairs can still land on a
strong one, and the data does reach 0.66.

So this measures the selected pairs, per question, and asks whether the coverage
delta tracks them:

    js_sel     max JS among the forks the scout actually selected
    js_pool    max JS among ALL candidate pairs it considered (the ceiling a
               perfect selector could have reached)
    w_sel      max learned divergence among selected forks (the contrast: the
               quantity the scout optimises)
    delta      coverage(condition) - coverage(baseline), per question

READING.
  corr(js_sel, delta) > 0      disagreement helps when it is there; the limit is
                               the SOURCE DATA, and richer viewpoints are the fix
  corr(js_sel, delta) ~ 0      magnitude is not the lever; the limit is delivery
                               (the model ignores the distinction) or the target
                               gap (demographic splits are not the judge's
                               vote-based clusters)
  js_sel high but delta flat   the strongest available disagreement still does
                               not move coverage -- the most damaging case for
                               divergence-based retrieval

NO GENERATION, NO JUDGE: one graph load and one scout pass per question, scored
against coverage numbers already on disk. Retrieval is deterministic given the
seed, so re-running it here reproduces what the eval injected.

CORRELATIONAL. A question where the graph happens to hold divergent subgroups may
differ in other ways too (contested topics are not a random sample). The causal
versions are the selection arms: merge_v2_divrand (random pair) and
merge_v2_jsdiv (rank by JS directly).

    python scripts/analysis/divergence_payoff.py --scores overton_scores_flat.csv \\
        --embeddings embeddings_opinionqa.pt --text_feat feats_opinionqa.pt \\
        --condition merge_v2
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.intrinsic.structure_metrics import _pearson              # noqa: E402
from scripts.analysis.delta_regressor import load_coverage               # noqa: E402


def question_divergence(question: str, graph, h_all, text_feat, manifold, cfg,
                        q_emb=None) -> dict:
    """Selected-vs-available divergence for one question, re-running the scout.

    Mirrors retrieval.scout.scout(): same anchors, same candidate pool, same rank
    key. Returns NaNs when nothing resolves, which is the same state in which the
    eval fell back to the baseline prompt.
    """
    from retrieval.scout import (embed_question, fork_js, lexical_anchors,
                                 node_relevance, rank_key, score_anchor)

    if q_emb is None:
        q_emb = embed_question(question)
    rel = node_relevance(q_emb, text_feat)
    anchors = lexical_anchors(question, graph, rel, cfg.max_anchors)
    pool = []
    for a in anchors:
        pool.extend(score_anchor(a, graph, h_all, rel, manifold, cfg))
    nan = float("nan")
    if not pool:
        return {"n_pool": 0, "n_sel": 0, "js_sel": nan, "js_pool": nan,
                "w_sel": nan, "rel_sel": nan}
    selected = sorted(pool, key=rank_key(question, cfg, graph),
                      reverse=True)[: cfg.top_k]

    def _max_js(forks):
        vals = [j for j in (fork_js(graph, f, cfg.max_nodes) for f in forks)
                if j is not None]
        return max(vals) if vals else nan

    return {"n_pool": len(pool), "n_sel": len(selected),
            "js_sel": _max_js(selected), "js_pool": _max_js(pool),
            "w_sel": max((f.w for f in selected), default=nan),
            "rel_sel": max((f.relevance for f in selected), default=nan)}


def _avg_ranks(vals: list[float]) -> list[float]:
    """Ranks with TIES AVERAGED.

    structure_metrics._spearman ranks by argsort, which assigns tied values
    distinct ranks in input order -- on a constant vector that manufactures a
    perfect correlation. Per-question deltas tie at exactly 0 often enough for
    this to matter, so ranking is done here instead.
    """
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        shared = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _spearman_ties(xs: list[float], ys: list[float]) -> float:
    """Pearson over tie-averaged ranks; NaN when either side is constant."""
    import torch

    rx, ry = _avg_ranks(xs), _avg_ranks(ys)
    X = torch.tensor(rx, dtype=torch.float)
    Y = torch.tensor(ry, dtype=torch.float)
    if float(X.std()) == 0.0 or float(Y.std()) == 0.0:
        return float("nan")
    return float(_pearson(X, Y))


def corr_with_ci(xs: list[float], ys: list[float], n_boot: int = 2000,
                 seed: int = 0) -> tuple[float, float, float]:
    """Spearman plus a percentile bootstrap CI over questions."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    if len(pairs) < 4:
        return float("nan"), float("nan"), float("nan")
    rho = _spearman_ties([p[0] for p in pairs], [p[1] for p in pairs])
    rng = random.Random(seed)
    boots = []
    idx = range(len(pairs))
    for _ in range(n_boot):
        pick = [rng.choice(idx) for _ in idx]
        r = _spearman_ties([pairs[i][0] for i in pick], [pairs[i][1] for i in pick])
        if r == r:
            boots.append(r)
    if not boots:
        return rho, float("nan"), float("nan")
    boots.sort()
    return rho, boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]


def tertile_table(rows: list[dict], key: str, label: str) -> None:
    """Mean delta by tertile of ``key`` -- the shape a correlation can hide."""
    usable = [r for r in rows if r[key] == r[key] and r["delta"] == r["delta"]]
    if len(usable) < 6:
        print(f"  [{label}] too few questions ({len(usable)})")
        return
    usable.sort(key=lambda r: r[key])
    k = len(usable) // 3
    parts = [("low", usable[:k]), ("mid", usable[k:2 * k]), ("high", usable[2 * k:])]
    print(f"  {label} tertile      n   mean {key}   mean delta")
    for name, part in parts:
        print(f"    {name:<14}{len(part):>4}{st.mean(r[key] for r in part):>12.4f}"
              f"{st.mean(r['delta'] for r in part):>13.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Does disagreement magnitude predict the gain?")
    ap.add_argument("--scores", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--text_feat", default=None)
    ap.add_argument("--dataset", default="opinionqa",
                    choices=["opinionqa", "globalopinionqa", "issp"])
    ap.add_argument("--condition", default="merge_v2")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--split", default="full", help="OvertonBench split for question text")
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42,
                    help="graph split seed; MUST match the run being analysed")
    ap.add_argument("--out", default="docs/divergence_payoff.csv")
    args = ap.parse_args()

    import torch

    from evaluation.overton.eval_overtonbench import load_questions
    from pluraltree.manifolds.poincare import PoincareBall
    from retrieval.answer import CONDITIONS
    from retrieval.scout import embed_question, load_or_compute_text_feat
    from data.loaders.graphs import load_graph

    cfg = CONDITIONS.get(args.condition)
    if cfg is None:
        ap.error(f"{args.condition} is not a retrieved condition; its forks are "
                 f"what this measures")
    cov = load_coverage(args.scores)
    questions = dict(load_questions(args.split))

    graph = load_graph(args.dataset, split_seed=args.seed, leakage_safe=True)
    h_all = torch.load(args.embeddings, map_location="cpu")
    if not isinstance(h_all, torch.Tensor):
        h_all = h_all["h_all"]
    if h_all.shape[0] != len(graph.id_to_entity):
        raise SystemExit(f"{args.embeddings} has {h_all.shape[0]} rows, graph has "
                         f"{len(graph.id_to_entity)} nodes -- wrong seed or dataset")
    manifold = PoincareBall(c=args.curvature)
    text_feat = load_or_compute_text_feat(graph, args.dataset, args.text_feat)

    rows = []
    for qid, per_cond in sorted(cov.items()):
        if args.condition not in per_cond or args.baseline not in per_cond:
            continue
        q = questions.get(qid)
        if not q:
            continue
        r = question_divergence(q, graph, h_all, text_feat, manifold, cfg,
                                q_emb=embed_question(q))
        r.update({"question_id": qid,
                  "delta": per_cond[args.condition] - per_cond[args.baseline],
                  "coverage": per_cond[args.condition]})
        rows.append(r)
        if len(rows) % 20 == 0:
            print(f"  {len(rows)} questions ...")

    if len(rows) < 4:
        print(f"only {len(rows)} questions had both conditions -- nothing to correlate")
        return 1

    resolved = [r for r in rows if r["n_sel"] > 0]
    js = [r["js_sel"] for r in resolved if r["js_sel"] == r["js_sel"]]
    print(f"\n=== selected-fork disagreement ({len(resolved)}/{len(rows)} questions "
          f"resolved) ===")
    if js:
        js_sorted = sorted(js)
        print(f"  max JS of the SELECTED forks: p25={js_sorted[len(js) // 4]:.4f}  "
              f"p50={js_sorted[len(js) // 2]:.4f}  "
              f"p75={js_sorted[3 * len(js) // 4]:.4f}  max={js_sorted[-1]:.4f}")
        print(f"  mean over same-axis pairs for reference: 0.0154 bits "
              f"(docs/findings_2026_09_15.md)")

    print(f"\n=== does the gain track disagreement? ({args.condition} - "
          f"{args.baseline}) ===")
    for key, label in (("js_sel", "selected JS"), ("js_pool", "best available JS"),
                       ("w_sel", "selected W (learned)"),
                       ("rel_sel", "selected relevance")):
        xs = [r[key] for r in resolved]
        ys = [r["delta"] for r in resolved]
        rho, lo, hi = corr_with_ci(xs, ys)
        flag = "" if lo != lo or (lo <= 0 <= hi) else "   <- interval excludes 0"
        print(f"  spearman({label:<22}, delta) = {rho:+.3f}  95% CI "
              f"[{lo:+.3f}, {hi:+.3f}]{flag}")
    print("  W is the quantity the scout maximises; JS is the disagreement it is "
          "meant to stand for.")

    print()
    tertile_table(resolved, "js_sel", "selected JS")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = ["question_id", "delta", "coverage", "n_pool", "n_sel", "js_sel",
            "js_pool", "w_sel", "rel_sel"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{k: r.get(k) for k in cols} for r in rows])
    print(f"\nwrote {args.out}")
    print("  Correlational: questions whose subgroups disagree are not a random "
          "sample. The causal tests are merge_v2_divrand and merge_v2_jsdiv.")
    return 0


def _selftest() -> None:
    """The aggregation logic, on synthetic questions with a planted relationship."""
    nan = float("nan")
    planted = [{"js_sel": i / 20.0, "delta": i / 20.0 + (0.01 if i % 2 else -0.01),
                "n_sel": 1} for i in range(21)]
    rho, lo, hi = corr_with_ci([r["js_sel"] for r in planted],
                               [r["delta"] for r in planted])
    assert rho > 0.9 and lo > 0.5, (rho, lo, hi)

    flat = [{"js_sel": i / 20.0, "delta": 0.05, "n_sel": 1} for i in range(21)]
    rho_f, lo_f, hi_f = corr_with_ci([r["js_sel"] for r in flat],
                                     [r["delta"] for r in flat])
    assert rho_f != rho_f or abs(rho_f) < 1e-6, rho_f    # constant y -> no signal

    assert corr_with_ci([1.0, 2.0], [1.0, 2.0])[0] != corr_with_ci([1.0, 2.0], [1.0, 2.0])[0], \
        "fewer than 4 usable pairs must give NaN"
    mixed = [{"js_sel": nan, "delta": 0.1, "n_sel": 0}] + planted
    rho_m, _, _ = corr_with_ci([r["js_sel"] for r in mixed],
                               [r["delta"] for r in mixed])
    assert abs(rho_m - rho) < 1e-9, "NaN rows must be dropped, not counted"
    tertile_table(planted, "js_sel", "selected JS")
    print("divergence_payoff self-test OK (correlation, CI, NaN handling, tertiles)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sys.exit(main())
