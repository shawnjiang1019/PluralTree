"""Mean OvertonScore per condition, with bootstrap CIs and a baseline reference.

The companion to plot_delta_ci.py: that one shows the DIFFERENCE from baseline,
this one shows the levels. Use it when the absolute score matters (a talk, a
table caption); use the delta plot when the question is whether the difference is
real, because a difference-of-means CI is much tighter than two overlapping
level CIs and eyeballing the overlap understates significance.

    python scripts/analysis/plot_condition_bars.py --scores overton_scores_v11.csv \\
        --conditions baseline,merge_v2,merge_v2_rand
"""

from __future__ import annotations

import argparse
import os
import random
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.analysis.delta_regressor import load_coverage

BLUE, INK, INK2, MUTED, AXIS = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
SURFACE = "#fcfcfb"


def boot_ci(xs, n_boot=10000, seed=0):
    rng = random.Random(seed)
    idx = range(len(xs))
    bs = sorted(st.mean(xs[rng.choice(idx)] for _ in idx) for _ in range(n_boot))
    return st.mean(xs), bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot)]


def main():
    ap = argparse.ArgumentParser(description="Bar chart of mean score per condition")
    ap.add_argument("--scores", required=True)
    ap.add_argument("--conditions", default="baseline,merge_v2,merge_v2_rand")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--title", default="Coverage by condition")
    ap.add_argument("--ymin", type=float, default=0.0,
                    help="lower y limit. ANY value > 0 switches bars to dots: "
                         "bar LENGTH encodes magnitude from zero, so a "
                         "truncated bar chart makes a 0.05 gap look like 3x. "
                         "Dots encode by POSITION, where zooming is honest.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cov = load_coverage(args.scores)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    rows = []
    for c in conds:
        v = [q[c] for q in cov.values() if c in q]
        if len(v) < 4:
            print(f"  [skip] {c}: only {len(v)} questions")
            continue
        m, lo, hi = boot_ci(v)
        rows.append((c, m, lo, hi, len(v)))
        print(f"  {c:<16} n={len(v)}  mean={m:.4f}  CI[{lo:.4f}, {hi:.4f}]")
    if not rows:
        ap.error("nothing to plot")

    fig, ax = plt.subplots(figsize=(1.7 * len(rows) + 3.0, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.80,
                        bottom=0.20 if args.ymin > 0 else 0.16)

    base = next((m for c, m, *_ in rows if c == args.baseline), None)
    if base is not None:
        ax.axhline(base, color=MUTED, ls=(0, (4, 3)), lw=1, zorder=1)
        # Axes fraction, not data coords: at data x=len(rows)-0.42 the label
        # landed outside the visible axis and never rendered.
        ax.text(0.995, base, f" {args.baseline} ", fontsize=8, color=MUTED,
                va="bottom", ha="right", zorder=4,
                transform=ax.get_yaxis_transform(),
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))

    x = range(len(rows))
    zoomed = args.ymin > 0
    # One measure across categories, so ONE colour: shading by rank would encode
    # the reader's conclusion rather than the data.
    if not zoomed:
        ax.bar(x, [m for _, m, *_ in rows], width=0.58, color=BLUE, zorder=2)
    for i, (_, m, lo, hi, _n) in enumerate(rows):
        if zoomed:
            # Dot + full interval: nothing encodes length, so the axis may start
            # anywhere.
            ax.plot([i, i], [lo, hi], color=BLUE, lw=2.2, alpha=0.55,
                    zorder=3, solid_capstyle="round")
            ax.plot([i], [m], "o", ms=11, color=BLUE, zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        else:
            # A white spine inside the bar, dark above it: a single colour loses
            # the lower half against the fill and the interval reads one-sided.
            ax.plot([i, i], [lo, min(hi, m)], color=SURFACE, lw=1.8, zorder=3,
                    solid_capstyle="round")
            ax.plot([i, i], [max(lo, m), hi], color=INK, lw=1.8, zorder=3,
                    solid_capstyle="round")
            for y in (lo, hi):
                ax.plot([i - 0.10, i + 0.10], [y, y],
                        color=SURFACE if y < m else INK, lw=1.8, zorder=3)
        ax.text(i, hi + 0.008, f"{m:.3f}", ha="center", va="bottom",
                fontsize=10, color=INK)

    ax.set_xlim(-0.62, len(rows) - 0.38)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{c}\n(n={n})" for c, _, _, _, n in rows], fontsize=9.5,
                       color=INK)
    ax.set_ylabel("mean OvertonScore", fontsize=9.5, color=INK2)
    top = max(hi for *_, hi, _ in rows)
    if zoomed:
        ax.set_ylim(args.ymin, top + (top - args.ymin) * 0.16)
    else:
        ax.set_ylim(0, top * 1.18)
    ax.tick_params(axis="y", labelsize=8, colors=INK2)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=AXIS, lw=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)

    fig.suptitle(args.title, x=0.018, y=0.955, ha="left", fontsize=14,
                 color=INK, weight="bold")
    mark = "dots are means, whiskers" if zoomed else "bars"
    fig.text(0.018, 0.865, f"{os.path.basename(args.scores)}  |  "
             f"{mark} are bootstrap 95% CIs on the mean", ha="left", fontsize=9,
             color=INK2)
    note = ("Overlapping level CIs do NOT mean the difference is insignificant"
            " — the paired difference is the test (plot_delta_ci.py).")
    if zoomed:
        note = (f"Axis starts at {args.ymin:g}, so marks are dots, not bars: "
                f"bar length must encode from zero.\n") + note
    fig.text(0.018, 0.030, note, ha="left", va="bottom", fontsize=8,
             color=MUTED, linespacing=1.5)

    out = args.out or f"docs/bars_{os.path.basename(args.scores).replace('.csv','')}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
