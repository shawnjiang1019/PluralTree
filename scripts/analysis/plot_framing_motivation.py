"""Why injection hurts coverage -- the MEASURED mechanism (not a schematic).

Every panel is real data from the v5 traces, via scripts/analysis/measure_pole_collapse.py:
  A  P1: on-pole similarity per question, baseline -> scout. The answer moves onto
     the injected poles on 72% of questions.
  B  P2: the more an answer moves onto the poles, the more coverage it loses (-0.31).
  C  the outcome: scout-baseline Dcoverage by how broad baseline already was.

NOT used: a PCA of the answer units. PC1 there is dominated by text FORMAT (poles
are survey-distribution strings, answers are prose), so both conditions sit far
from the poles and the real, relative shift is invisible. Panel A plots the
measured quantity directly instead.

Run measure_pole_collapse.py first (writes docs/pole_collapse.csv), then:
    python scripts/analysis/plot_framing_motivation.py   -> docs/framing_hurts.png

Palette: dataviz diverging pair blue<->red encodes the sign of the coverage
outcome; ink for measured means. No dual axes; each panel is one measure.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, RED = "#2a78d6", "#e34948"                       # diverging pair (polarity)
INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, SURF = "#e1e0d9", "#fcfcfb"

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(csv_path):
    rows = []
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        rows.append({k: (float(v) if v not in ("", None) else None)
                     for k, v in r.items()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default=os.path.join(ROOT, "docs/pole_collapse.csv"))
    args = ap.parse_args()

    rows = load(args.stats)
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14.4, 4.5),
                                        gridspec_kw={"width_ratios": [1.05, 1.0, 1.0]})
    for ax in (axA, axB, axC):
        ax.set_facecolor(SURF)

    # ---------- A: P1 -- does the answer move onto the injected poles? ----------
    # A PCA of the units does NOT work here: PC1 is dominated by text FORMAT (the
    # poles are survey-distribution strings, the answers are prose), so both
    # conditions land far from the poles and the real, relative shift is invisible.
    # Plot the measured quantity itself instead: on-pole similarity, baseline->scout.
    up = down = 0
    for r in rows:
        rose = r["delta"] > 0
        up += rose
        down += not rose
        axA.plot([0, 1], [r["pole_baseline"], r["pole_scout"]],
                 color=(BLUE if rose else RED), alpha=0.35, lw=1.1, zorder=2,
                 solid_capstyle="round")
    for xi, key, mk in ((0, "pole_baseline", "o"), (1, "pole_scout", "s")):
        m = st.mean([r[key] for r in rows])
        axA.scatter([xi], [m], s=150, marker=mk, color=INK, zorder=5,
                    edgecolor=SURF, linewidth=1.5)
        axA.annotate(f"mean {m:.3f}", (xi, m), xytext=(0, 14),
                     textcoords="offset points", ha="center", fontsize=9,
                     color=INK, fontweight="bold")
    axA.plot([0, 1], [st.mean([r["pole_baseline"] for r in rows]),
                      st.mean([r["pole_scout"] for r in rows])],
             color=INK, lw=2.5, zorder=4)
    att_mean = st.mean([r["attraction"] for r in rows])
    frac_up = sum(r["attraction"] > 0 for r in rows) / len(rows)
    axA.set_title(f"A · the answer moves ONTO the injected poles\n"
                  f"+{att_mean:.3f} mean, on {frac_up:.0%} of questions (n={len(rows)})",
                  fontsize=10, color=INK)
    axA.set_xticks([0, 1], ["baseline", "scout"], fontsize=10)
    axA.set_xlim(-0.32, 1.32)
    axA.set_ylabel("on-pole similarity  (answer → injected poles)",
                   fontsize=9, color=SEC)
    axA.tick_params(colors=MUT, labelsize=8)
    axA.grid(axis="y", lw=0.5, color=GRID, alpha=0.9)
    axA.set_axisbelow(True)
    from matplotlib.lines import Line2D
    axA.legend(handles=[Line2D([], [], color=RED, lw=2, label=f"lost coverage ({down})"),
                        Line2D([], [], color=BLUE, lw=2, label=f"gained ({up})")],
               loc="upper left", fontsize=8, frameon=False)

    # ---------- B: attraction vs delta (all questions) ----------
    att = np.array([r["attraction"] for r in rows])
    dl = np.array([r["delta"] for r in rows])
    axB.scatter(att, dl, s=42, color=BLUE, alpha=0.75, edgecolor=SURF, linewidth=0.8)
    m, b = np.polyfit(att, dl, 1)
    xs = np.linspace(att.min(), att.max(), 50)
    axB.plot(xs, m * xs + b, color=INK, lw=2, zorder=4)
    axB.axhline(0, color=GRID, lw=1, zorder=0)
    axB.axvline(0, color=GRID, lw=1, zorder=0)
    r_ad = np.corrcoef(att, dl)[0, 1]
    axB.set_title(f"B · moving onto the poles predicts the loss\n"
                  f"r = {r_ad:+.2f}  (n={len(rows)})", fontsize=10, color=INK)
    axB.set_xlabel("pole attraction  (scout − baseline on-pole similarity)",
                   fontsize=9, color=SEC)
    axB.set_ylabel("Δ coverage  (scout − baseline)", fontsize=9, color=SEC)
    axB.tick_params(colors=MUT, labelsize=8)
    axB.annotate("answer drifts onto\nthe injected poles →", (0.62, 0.90),
                 xycoords="axes fraction", fontsize=8, color=MUT, ha="left", va="top")

    # ---------- C: outcome by baseline breadth ----------
    bins = [("baseline\n< 0.3", lambda x: x < 0.3),
            ("baseline\n0.3 – 0.8", lambda x: 0.3 <= x < 0.8),
            ("baseline\n≥ 0.8", lambda x: x >= 0.8)]
    labels, vals = [], []
    for name, pred in bins:
        g = [r["delta"] for r in rows if pred(r["base_cov"])]
        labels.append(f"{name}\n(n={len(g)})"); vals.append(st.mean(g))
    x = np.arange(3)
    bars = axC.bar(x, vals, 0.6, color=[BLUE if v > 0 else RED for v in vals])
    for r_, v in zip(bars, vals):
        axC.annotate(f"{v:+.2f}", (r_.get_x() + r_.get_width() / 2,
                     v + (0.015 if v > 0 else -0.015)), ha="center",
                     va="bottom" if v > 0 else "top", fontsize=10, color=INK)
    axC.axhline(0, color="#c3c2b7", lw=1)
    axC.set_ylim(min(vals) - 0.09, max(vals) + 0.07)      # headroom for the labels
    axC.set_xticks(x, labels, fontsize=9)
    axC.set_ylabel("Δ coverage  (scout − baseline)", fontsize=9, color=SEC)
    axC.set_title("C · the loss is worst where the model\nwas already broad",
                  fontsize=10, color=INK)
    axC.tick_params(colors=MUT, labelsize=8)
    axC.grid(axis="y", lw=0.5, color=GRID, alpha=0.9)
    axC.set_axisbelow(True)

    for ax in (axA, axB, axC):
        ax.spines[["top", "right"]].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")

    fig.suptitle("Injection pulls the answer onto the two injected poles — "
                 "and that drift is what costs coverage",
                 fontsize=12, color=INK, y=1.02)
    plt.tight_layout()
    out = os.path.join(ROOT, "docs", "framing_hurts.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
