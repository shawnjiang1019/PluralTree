"""Mean coverage delta vs baseline, with a bootstrap 95% CI. One row per condition.

The point of the picture is the INTERVAL, not the dot: +0.0389 reads as a result
and [-0.0035, +0.0835] reads as what it is. Whether the bar crosses zero is the
whole finding, so zero is the only reference line drawn heavily.

    python scripts/analysis/plot_delta_ci.py --scores overton_scores_v10.csv \
        --conditions merge_v2,merge --out docs/delta_ci_v10.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.analysis.delta_regressor import load_coverage

BLUE, INK, INK2, MUTED, AXIS = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
SURFACE = "#fcfcfb"


def boot_ci(xs, n_boot=10000, seed=0):
    """Percentile bootstrap on the mean, plus a two-sided p against 0."""
    rng = random.Random(seed)
    idx = range(len(xs))
    bs = sorted(st.mean(xs[rng.choice(idx)] for _ in idx) for _ in range(n_boot))
    n_le0 = sum(1 for b in bs if b <= 0.0)
    p = 2.0 * min(n_le0, n_boot - n_le0) / n_boot
    return st.mean(xs), bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot)], p


def main():
    ap = argparse.ArgumentParser(description="Plot mean delta with a bootstrap CI")
    ap.add_argument("--scores", required=True)
    ap.add_argument("--conditions", default="merge_v2")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--tol", type=float, default=0.027, help="measured noise floor")
    ap.add_argument("--out", default=None)
    ap.add_argument("--responses", default=None,
                    help="responses JSONL; with it, each condition is also split "
                         "into questions where it retrieved forks (n_forks>0 in "
                         "any rollout) and questions where it did not. On a graph "
                         "that resolves only part of the benchmark (ISSP: 32/60) "
                         "the pooled delta is diluted by arms that ran as baseline.")
    args = ap.parse_args()

    cov = load_coverage(args.scores)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]

    resolved = None
    if args.responses:
        resolved = defaultdict(set)      # condition -> qids with forks
        with open(args.responses, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if int(r.get("n_forks", 0)) > 0:
                        resolved[r["condition"]].add(int(r["question_id"]))

    rows = []
    for c in conds:
        paired = {qid: q[c] - q[args.baseline] for qid, q in cov.items()
                  if args.baseline in q and c in q}
        subsets = [(c, list(paired.values()))]
        if resolved is not None:
            subsets += [(f"{c} resolved", [d for qid, d in paired.items() if qid in resolved[c]]),
                        (f"{c} unresolved", [d for qid, d in paired.items() if qid not in resolved[c]])]
        for label, d in subsets:
            if len(d) < 4:
                print(f"  [skip] {label}: only {len(d)} questions")
                continue
            m, lo, hi, p = boot_ci(d)
            rows.append((label, m, lo, hi, p, len(d)))
            print(f"  {label:<22} n={len(d)}  mean={m:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  p={p:.4f}")
    if resolved is not None:
        print("  unresolved rows ran without forks, so their delta is baseline-vs-"
              "baseline drift: it should sit inside the noise floor. If it does not, "
              "the arm differs from baseline for a reason other than retrieval.")
    if not rows:
        ap.error("nothing to plot")

    fig, ax = plt.subplots(figsize=(8.4, 1.05 * len(rows) + 2.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.20, right=0.96, top=0.76, bottom=0.30)

    ax.axvspan(-args.tol, args.tol, color=MUTED, alpha=0.13, lw=0, zorder=0)
    ax.axvline(0, color=INK2, lw=1.2, zorder=2)

    for i, (c, m, lo, hi, p, n) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=BLUE, lw=2.5, solid_capstyle="round",
                zorder=3, alpha=0.55)
        ax.plot([m], [i], "o", ms=11, color=BLUE, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=4)
        # The label sits over the zero line and the noise band; give it an
        # opaque backing so neither shows through the digits.
        ax.text(hi, i + 0.28, f"{m:+.4f}  [{lo:+.3f}, {hi:+.3f}]   p={p:.3f}",
                fontsize=8.5, color=INK2, va="bottom", ha="right", zorder=5,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.6))

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{c}\n(n={n})" for c, _, _, _, _, n in rows],
                       fontsize=10, color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.15)
    ax.set_xlabel(f"mean coverage delta vs {args.baseline}   "
                  f"(dot = mean, bar = bootstrap 95% CI)",
                  fontsize=9, color=INK2)
    ax.tick_params(axis="x", labelsize=8, colors=INK2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=AXIS, lw=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)

    fig.suptitle("Does injection beat the baseline?", x=0.02, y=0.95,
                 ha="left", fontsize=14, color=INK, weight="bold")
    fig.text(0.02, 0.855, f"{os.path.basename(args.scores)}  |  "
             f"a bar crossing zero is not a result", ha="left",
             fontsize=9.5, color=INK2)
    fig.text(0.02, 0.06, f"Shaded: +/-{args.tol}, the measured per-question noise "
             f"floor. 10,000 bootstrap resamples.", ha="left",
             fontsize=8.5, color=MUTED)

    out = args.out or f"docs/delta_ci_{os.path.basename(args.scores).replace('.csv', '')}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
