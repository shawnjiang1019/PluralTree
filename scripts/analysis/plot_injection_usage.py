"""How much of the injection reaches the answer: the whole distribution, not three cuts.

`used@t` is the fraction of injected positions whose best-matching answer unit
clears cosine `t`. That is a SURVIVAL FUNCTION, so the three thresholds in the
text report are three points on one curve. Plotting the curve removes the "why
0.35?" question and shows the gap between arms at every threshold at once.

Panel A  survival: fraction of injected positions matched at >= x, per condition
Panel B  where in the answer the match landed (0 = first unit, 1 = last)

Reads the per-position csv from `injection_usage.py --dump`.

    python scripts/analysis/injection_usage.py --responses overton_responses_v11.jsonl \\
        --conditions merge_v2,merge_v2_rand --dump docs/usage_positions.csv
    python scripts/analysis/plot_injection_usage.py --dump docs/usage_positions.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# dataviz reference palette, categorical slots 1-2 (validated all-pairs, light).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, AXIS = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
SURFACE = "#fcfcfb"


def survival(xs, grid):
    """Fraction of xs >= each point of grid. This IS used@t, swept."""
    s = sorted(xs)
    n = len(s)
    out, i = [], 0
    for g in grid:
        while i < n and s[i] < g:
            i += 1
        out.append((n - i) / n if n else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser(description="Plot injection usage distributions")
    ap.add_argument("--dump", required=True, help="csv from injection_usage --dump")
    ap.add_argument("--thresholds", default="0.35,0.45,0.55",
                    help="marked for cross-reference with the text report")
    ap.add_argument("--out", default="docs/injection_usage.png")
    args = ap.parse_args()

    by_cond = collections.defaultdict(lambda: {"cos": [], "loc": []})
    with open(args.dump, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c = row["condition"]
            by_cond[c]["cos"].append(float(row["best_cos"]))
            by_cond[c]["loc"].append(float(row["loc"]))
    if not by_cond:
        ap.error(f"no rows in {args.dump}")
    conds = list(by_cond)
    thrs = [float(t) for t in args.thresholds.split(",")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.9), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.76, bottom=0.19, wspace=0.24)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(color=AXIS, lw=0.6, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(labelsize=8, colors=INK2)

    # --- A: survival over cosine -------------------------------------------
    grid = [i / 200 for i in range(201)]
    for t in thrs:
        ax1.axvline(t, color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax1.text(t, 1.02, f"{t:g}", fontsize=7.5, color=MUTED, ha="center",
                 va="bottom")
    for k, c in enumerate(conds):
        y = survival(by_cond[c]["cos"], grid)
        col = SERIES[k % len(SERIES)]
        ax1.plot(grid, y, color=col, lw=2, zorder=3)
        # Direct label at the curve's own height, so no legend lookup is needed.
        yi = y[int(0.30 * len(grid))]
        ax1.text(0.305, yi + 0.02, c, color=col, fontsize=9, ha="left",
                 va="bottom", zorder=4,
                 bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlabel("cosine of the best-matching answer unit", fontsize=9, color=INK2)
    ax1.set_ylabel("fraction of injected positions matched at >= x", fontsize=9,
                   color=INK2)
    ax1.set_title("A · How much of the injection lands", fontsize=11,
                  color=INK, loc="left", pad=14)

    # --- B: where in the answer --------------------------------------------
    bins = [i / 10 for i in range(11)]
    for k, c in enumerate(conds):
        ax2.hist(by_cond[c]["loc"], bins=bins, density=True, histtype="step",
                 lw=2, color=SERIES[k % len(SERIES)], zorder=3, label=c)
    # Panel A direct-labels its curves; B is steps that cannot carry a label on
    # the line, so it needs the legend or its colours are undecodable alone.
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK2)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("position in the answer  (0 = first unit, 1 = last)",
                   fontsize=9, color=INK2)
    ax2.set_ylabel("density of matches", fontsize=9, color=INK2)
    ax2.set_title("B · Where it lands", fontsize=11, color=INK, loc="left",
                  pad=14)

    fig.suptitle("Does the injected block reach the answer?", x=0.012, y=0.955,
                 ha="left", fontsize=14, color=INK, weight="bold")
    fig.text(0.012, 0.865,
             f"{os.path.basename(args.dump)}  |  one point per injected position; "
             f"panel A swept, so every used@t is readable off the curve",
             ha="left", fontsize=9, color=INK2)
    fig.text(0.012, 0.030,
             "Cosine against an unrelated position is not zero — read the GAP "
             "between curves, never the level.\nA flat panel B means influence is "
             "spread through the answer; a spike near 0 would mean the model "
             "acknowledges the forks and then reverts.",
             ha="left", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}")
    for c in conds:
        xs = by_cond[c]["cos"]
        print(f"  {c:<16} n={len(xs):5}  " +
              "  ".join(f"used@{t:g}={survival(xs, [t])[0]:.3f}" for t in thrs))


if __name__ == "__main__":
    main()
