"""How to weight JS against EMD in the disagreement target D. No model, no LLM.

    D_lam = lam * d_JS + (1 - lam) * EMD

over same-axis subgroup pairs (two groups of one attribute answering one item),
with d_JS the base-2 Jensen-Shannon DISTANCE and EMD over the release's
option_ordinal positions (semantic cost for nominal items). Three checks, all
from the survey data alone, so the choice cannot leak coverage results:

  agreement    Spearman(d_JS, EMD), overall and by scale type, and Kendall tau
               between the pair orderings at each lam. The disagreement loss is
               rank-based, so if the orderings barely move, lam does not matter.
  reliability  split-half reliability per lam: for every group pair that recurs
               across items (e.g. very liberal vs very conservative), mean D on a
               random half of the items vs on the other half, correlated across
               group pairs. The most reliable lam is the least noisy measure.
  spread       sd of each component, for the variance-equalised fallback
               D = 0.5 d_JS / sd_JS + 0.5 EMD / sd_EMD.

Uses every subgroup leaf (grouped by node name, not by the leakage-safe tree),
since this only measures the survey data.

    python scripts/analysis/dis_weight_check.py --out dis_weight_check.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _ranks(x: list[float]) -> list[float]:
    """Average ranks (ties share their mean rank)."""
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def _pearson(a: list[float], b: list[float]) -> float:
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den > 0 else float("nan")


def spearman(a: list[float], b: list[float]) -> float:
    return _pearson(_ranks(a), _ranks(b))


def kendall(a: list[float], b: list[float], n_sample: int, rng) -> float:
    """Kendall tau-b on a random subsample of pairs-of-pairs (exact is O(n^2))."""
    try:
        from scipy.stats import kendalltau
        idx = rng.sample(range(len(a)), min(len(a), n_sample))
        return float(kendalltau([a[i] for i in idx], [b[i] for i in idx])[0])
    except ImportError:
        conc = disc = 0
        n = len(a)
        for _ in range(n_sample):
            i, j = rng.randrange(n), rng.randrange(n)
            s = (a[i] - a[j]) * (b[i] - b[j])
            conc += s > 0
            disc += s < 0
        return (conc - disc) / max(1, conc + disc)


def main() -> None:
    ap = argparse.ArgumentParser(description="JS vs EMD weight for the disagreement target")
    ap.add_argument("--dataset", default="opinionqa")
    ap.add_argument("--seed", type=int, default=42, help="graph split seed (node ids)")
    ap.add_argument("--embed_model", default="all-MiniLM-L6-v2",
                    help="encoder for the nominal-option ground cost")
    ap.add_argument("--n_splits", type=int, default=20)
    ap.add_argument("--min_items", type=int, default=10,
                    help="items a group pair needs in EACH half to enter reliability")
    ap.add_argument("--kendall_sample", type=int, default=20000)
    ap.add_argument("--out", default="dis_weight_check.csv")
    args = ap.parse_args()

    from data.loaders.graphs import load_graph
    from retrieval.scout import _emd_norm, _js_bits, _option_cost

    rng = random.Random(0)
    graph = load_graph(args.dataset, split_seed=args.seed, leakage_safe=True)
    dists = graph.opinion_dist

    # (qkey, attr) -> [(group, leaf)] from node names op:{qkey}:{attr}:{group}
    axes: dict[tuple[str, str], list[tuple[str, int]]] = collections.defaultdict(list)
    for leaf in dists:
        name = graph.id_to_entity[leaf]
        if not name.startswith("op:"):
            continue
        qkey, attr, group = name[3:].rsplit(":", 2)
        axes[(qkey, attr)].append((group, leaf))

    pairs = []          # (qkey, attr, group_a, group_b, scale, d_js, emd)
    for (qkey, attr), members in axes.items():
        if len(members) < 2:
            continue
        cost, pos = _option_cost(graph, members[0][1], args.embed_model)
        scale = "ordinal" if cost is None else "nominal"
        members = sorted(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                (ga, a), (gb, b) = members[i], members[j]
                if len(dists[a]) != len(dists[b]):
                    continue
                pairs.append((qkey, attr, ga, gb, scale,
                              math.sqrt(_js_bits(dists[a], dists[b])),
                              _emd_norm(dists[a], dists[b], cost, pos)))
    js = [p[5] for p in pairs]
    emd = [p[6] for p in pairs]
    n_nom = sum(1 for p in pairs if p[4] == "nominal")
    print(f"{len(pairs)} same-axis pairs over {len(axes)} axes "
          f"({n_nom} on nominal items)")

    # 1. agreement
    print("\nAGREEMENT (rank-based loss: only the ordering matters)")
    print(f"  Spearman(d_JS, EMD) overall  {spearman(js, emd):.3f}")
    for s in ("ordinal", "nominal"):
        sub = [(p[5], p[6]) for p in pairs if p[4] == s]
        if len(sub) > 2:
            print(f"  Spearman(d_JS, EMD) {s:8} {spearman(*map(list, zip(*sub))):.3f}  (n={len(sub)})")
    d = {lam: [lam * x + (1 - lam) * y for x, y in zip(js, emd)] for lam in LAMS}
    print("  Kendall tau of the pair ordering vs lam=0.5:")
    for lam in LAMS:
        print(f"    lam={lam:<4} {kendall(d[lam], d[0.5], args.kendall_sample, rng):.3f}")

    # 2. spread
    sd_js, sd_emd = st.pstdev(js), st.pstdev(emd)
    print(f"\nSPREAD  sd(d_JS)={sd_js:.4f}  sd(EMD)={sd_emd:.4f}  "
          f"-> variance-equalised weight on d_JS = {sd_emd / (sd_js + sd_emd):.3f} "
          f"(on the raw scale)")

    # 3. split-half reliability over recurring group pairs
    by_gp: dict[tuple[str, str, str], list[int]] = collections.defaultdict(list)
    for k, p in enumerate(pairs):
        by_gp[(p[1], p[2], p[3])].append(k)
    qkeys = sorted({p[0] for p in pairs})
    rel = {lam: [] for lam in LAMS}
    n_gp_used = []
    for _ in range(args.n_splits):
        half = set(rng.sample(qkeys, len(qkeys) // 2))
        xs = {lam: [] for lam in LAMS}
        ys = {lam: [] for lam in LAMS}
        used = 0
        for ks in by_gp.values():
            h1 = [k for k in ks if pairs[k][0] in half]
            h2 = [k for k in ks if pairs[k][0] not in half]
            if len(h1) < args.min_items or len(h2) < args.min_items:
                continue
            used += 1
            for lam in LAMS:
                xs[lam].append(st.mean(d[lam][k] for k in h1))
                ys[lam].append(st.mean(d[lam][k] for k in h2))
        n_gp_used.append(used)
        for lam in LAMS:
            if used > 2:
                rel[lam].append(_pearson(xs[lam], ys[lam]))
    print(f"\nRELIABILITY  split-half over {st.mean(n_gp_used):.0f} recurring group "
          f"pairs (>= {args.min_items} items per half), {args.n_splits} splits")
    rows = []
    for lam in LAMS:
        r = rel[lam]
        m, s = (st.mean(r), st.pstdev(r)) if r else (float("nan"), float("nan"))
        rows.append({"lam": lam, "reliability_mean": round(m, 4),
                     "reliability_sd": round(s, 4)})
        print(f"  lam={lam:<4} r = {m:.3f} +/- {s:.3f}")
    best = max(rows, key=lambda r: r["reliability_mean"])
    print(f"  most reliable: lam={best['lam']}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lam", "reliability_mean", "reliability_sd"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nDone -> {args.out}")


if __name__ == "__main__":
    main()
