"""What each condition ADDS to baseline, versus what it scores alone.

The v12 run inverted the ranking, and a bar chart of the headline metric hides it:

  condition        alone (coverage@K)   union with baseline   gain
  merge_v2                     0.6460                0.6848  +0.0389
  merge_v2_rand                0.6401                0.7174  +0.0773
  persona_merge                0.6423                0.7340  +0.0917

The three score almost identically ALONE (0.640-0.646) and differ enormously in
what they add. merge_v2 wins the headline by covering clusters baseline already
covers; persona_merge finds different ones and then loses them in the merge.
That is a statement about the MERGE, not about retrieval, and it is invisible on
any plot of the single-answer score.

Union needs per-question covered-cluster SETS, which the scores csv does not
carry -- only judge_overtonbench computes them. So this parses the union table
out of the judge's stdout instead of asking anyone to retype it.

    python scripts/analysis/plot_union_gain.py --log logs/overton_eval_2186455.out
"""

from __future__ import annotations

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, INK2, MUTED, AXIS = ("#2a78d6", "#eb6834", "#0b0b0b",
                                        "#52514e", "#898781", "#c3c2b7")
SURFACE = "#fcfcfb"

# "  baseline+merge_v2   60   0.6460   0.6821   0.6848 +0.0389"
_ROW = re.compile(
    r"^\s*([A-Za-z0-9_+]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)\s*$")


def parse_union_table(path: str, baseline: str):
    """Two-condition rows of the judge's cross-condition union table."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _ROW.match(line)
            if not m:
                continue
            combo, n, best, oracle, union, gain = m.groups()
            parts = combo.split("+")
            if len(parts) != 2 or baseline not in parts:
                continue                      # pairs only; the 4-way row is a ceiling
            other = parts[0] if parts[1] == baseline else parts[1]
            out.append({"condition": other, "n": int(n), "best": float(best),
                        "oracle": float(oracle), "union": float(union),
                        "gain": float(gain)})
    return out


def main():
    ap = argparse.ArgumentParser(description="Alone vs union-with-baseline")
    ap.add_argument("--log", required=True, help="judge stdout with the union table")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--title", default="What each condition adds to baseline")
    ap.add_argument("--out", default="docs/union_gain.png")
    args = ap.parse_args()

    rows = parse_union_table(args.log, args.baseline)
    if not rows:
        ap.error(f"no two-condition union rows found in {args.log}")
    rows.sort(key=lambda r: r["gain"])          # smallest gain at the bottom
    for r in rows:
        print(f"  {r['condition']:<16} alone {r['best']:.4f}  union "
              f"{r['union']:.4f}  gain {r['gain']:+.4f}")

    fig, ax = plt.subplots(figsize=(9.2, 1.05 * len(rows) + 3.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.20, right=0.955, top=0.76, bottom=0.22)

    for i, r in enumerate(rows):
        # The connector IS the gain; the dots are the two levels it spans.
        ax.plot([r["best"], r["union"]], [i, i], color=AXIS, lw=2.5, zorder=2,
                solid_capstyle="round")
        ax.plot([r["best"]], [i], "o", ms=11, color=BLUE, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.plot([r["union"]], [i], "o", ms=11, color=ORANGE, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(r["union"] + 0.004, i, f"  +{r['gain']:.4f}", fontsize=9.5,
                color=INK, va="center", ha="left")

    lo = min(r["best"] for r in rows)
    hi = max(r["union"] for r in rows)
    pad = (hi - lo) * 0.30
    ax.set_xlim(lo - pad * 0.6, hi + pad)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["condition"] for r in rows], fontsize=10, color=INK)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(f"coverage@K   (blue = the condition alone, "
                  f"orange = union with {args.baseline})", fontsize=9, color=INK2)
    ax.tick_params(axis="x", labelsize=8, colors=INK2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=AXIS, lw=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)

    fig.suptitle(args.title, x=0.018, y=0.955, ha="left", fontsize=14, color=INK,
                 weight="bold")
    fig.text(0.018, 0.865, f"{os.path.basename(args.log)}  |  n={rows[0]['n']}  "
             f"|  the bar length IS the gain", ha="left", fontsize=9, color=INK2)
    fig.text(0.018, 0.030,
             "Blue dots sit almost on top of each other: the arms score the same "
             "ALONE. The orange dots do not.\nA long bar means the condition "
             "covers clusters baseline misses — content its own merged answer "
             "failed to keep.",
             ha="left", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
