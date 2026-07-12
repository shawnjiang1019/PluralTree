"""Intra-pool self-similarity of INFINITY-CHAT generations (Artificial Hivemind).

Reproduces the paper's repetition metric (Jiang et al., NeurIPS 2025, Fig 4): for
each query's pool of N samples, compute the mean pairwise sentence-embedding
cosine similarity; a pool that collapses to one mode has mean sim -> 1. We report
this per condition so scout/div_only injection can be tested against baseline:
lower mean self-similarity = less mode collapse.

Metric note: the paper embeds with OpenAI text-embedding-3-small (headline: 79%
of query pools exceed 0.8). We use the same MiniLM as the scout (offline, no
extra deps), so absolute values differ — the baseline-vs-injected *delta* is the
valid comparison. Both mean-sim and %pools>0.8 are printed.

Usage:
    python -m evaluation.hivemind.diversity_metrics hivemind_gen.jsonl \
        --out hivemind_diversity.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _pool_similarity(embs) -> tuple[float, float, float]:
    """(mean, %>0.8, %>0.7) over the upper-triangle pairwise cosines of a pool."""
    import numpy as np

    e = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    sim = e @ e.T
    iu = np.triu_indices(len(e), k=1)
    if iu[0].size == 0:                              # pool of 1 -> undefined
        return float("nan"), float("nan"), float("nan")
    pair = sim[iu]
    return float(pair.mean()), float((pair > 0.8).mean()), float((pair > 0.7).mean())


def main():
    ap = argparse.ArgumentParser(description="INFINITY-CHAT diversity metrics")
    ap.add_argument("infile", help="JSONL from generate_hivemind.py")
    ap.add_argument("--out", default="hivemind_diversity.csv")
    ap.add_argument("--min_samples", type=int, default=2,
                    help="skip (query, condition) pools smaller than this")
    args = ap.parse_args()

    import numpy as np
    from sentence_transformers import SentenceTransformer

    from retrieval.scout import MINILM

    # pool[(condition, query_id)] = (category, [responses...])
    pool: dict[tuple[str, int], list[str]] = defaultdict(list)
    cat_of: dict[tuple[str, int], str] = {}
    with open(args.infile, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("sample_idx", 0) < 0:           # dry_run prompt row
                continue
            key = (r["condition"], r["query_id"])
            pool[key].append(r["response"])
            cat_of[key] = r.get("category", "uncategorized")

    enc = SentenceTransformer(MINILM, device="cpu")

    # per-pool metrics
    rows = []
    for (cond, qid), responses in pool.items():
        if len(responses) < args.min_samples:
            continue
        embs = enc.encode(responses, convert_to_numpy=True, batch_size=64)
        mean, p80, p70 = _pool_similarity(embs)
        rows.append({"condition": cond, "query_id": qid,
                     "category": cat_of[(cond, qid)], "n": len(responses),
                     "mean_sim": mean, "frac_pairs_gt80": p80,
                     "frac_pairs_gt70": p70})

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "query_id", "category",
                                          "n", "mean_sim", "frac_pairs_gt80",
                                          "frac_pairs_gt70"])
        w.writeheader()
        w.writerows(rows)

    # aggregate per condition: mean over query pools + %pools whose mean sim>0.8
    print(f"\n{len(rows)} pools -> {args.out}\n")
    print(f"{'condition':<12}{'pools':>7}{'mean_sim':>11}"
          f"{'%pools>.8':>11}{'mean%pairs>.8':>15}")
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    for cond in sorted(by_cond):
        rs = by_cond[cond]
        ms = np.array([r["mean_sim"] for r in rs])
        pools_collapsed = float((ms > 0.8).mean())
        mean_pairs80 = float(np.nanmean([r["frac_pairs_gt80"] for r in rs]))
        print(f"{cond:<12}{len(rs):>7}{np.nanmean(ms):>11.3f}"
              f"{pools_collapsed:>11.3f}{mean_pairs80:>15.3f}")

    # per-category x condition mean self-similarity (collapse hotspots)
    cats = sorted({r["category"] for r in rows})
    if len(cats) > 1:
        conds = sorted(by_cond)
        print("\nmean_sim by category x condition:")
        print(f"{'category':<24}" + "".join(f"{c:>12}" for c in conds))
        for cat in cats:
            cells = []
            for c in conds:
                vals = [r["mean_sim"] for r in rows
                        if r["category"] == cat and r["condition"] == c]
                cells.append(f"{np.nanmean(vals):>12.3f}" if vals else f"{'-':>12}")
            print(f"{cat[:24]:<24}" + "".join(cells))


if __name__ == "__main__":
    main()
