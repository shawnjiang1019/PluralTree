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

import csv
from collections import defaultdict

# Which column of the judge's scores CSV to plot. Both are per question:
#   coverage        mean over rollouts of ONE answer's coverage (OvertonScore)
#   union_coverage  a cluster counts if ANY rollout covers it (union@K)
METRICS = {"coverage": "mean per-answer coverage (OvertonScore)",
           "union_coverage": "mean union@K coverage (any of K answers)"}


def load_coverage(path: str, metric: str = "coverage") -> dict:
    """qid -> condition -> score from the judge's scores CSV."""
    cov = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cov[int(r["question_id"])][r["condition"]] = float(r[metric])
    return cov

BLUE, INK, INK2, MUTED, AXIS = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
SURFACE = "#fcfcfb"


def boot_ci(xs, n_boot=10000, seed=0):
    rng = random.Random(seed)
    idx = range(len(xs))
    bs = sorted(st.mean(xs[rng.choice(idx)] for _ in idx) for _ in range(n_boot))
    return st.mean(xs), bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot)]


def plot_panels(panels, conds, baseline, metric, title, out):
    """One panel per scores file (e.g. per judge), side by side, shared y-axis.

    Panels are deliberately NOT interleaved bars: judges differ in generosity,
    so old-vs-new bars on the same arm mostly show that offset. The valid
    comparison is the ordering of arms WITHIN one judge, which is what each
    panel and its own baseline line show. Arms a judge never scored keep their
    slot with a "not judged" mark, so the x positions line up across panels.
    """
    stats = []
    for ptitle, path in panels:
        cov = load_coverage(path, metric)
        rows = []
        for c in conds:
            v = [q[c] for q in cov.values() if c in q]
            rows.append((c, *boot_ci(v), len(v)) if len(v) >= 4 else (c, None, None, None, 0))
        stats.append((ptitle, path, rows))
        print(f"[{ptitle}]  {os.path.basename(path)}")
        for c, m, lo, hi, n in rows:
            print(f"  {c:<16} " + (f"n={n}  mean={m:.4f}  CI[{lo:.4f}, {hi:.4f}]"
                                   if m is not None else "not judged"))
    draw_panels(stats, conds, baseline, metric, title, out)


def draw_panels(stats, conds, baseline, metric, title, out, *,
                colors=None, zoom=False, marks="dots"):
    """Draw from precomputed rows (cond, mean, lo, hi, n) per panel.

    lo/hi may be None (summary numbers only): marks are drawn without an
    interval and the subtitle says so, rather than implying a CI that was
    never computed.

    colors: {condition: hex}. Colour follows the CONDITION, so an arm keeps its
    colour in every panel. Default: one colour for all.

    zoom: each panel's y-axis is fitted to its own values, so small differences
    are visible -- and the marks switch from bars to DOTS. A bar encodes its
    value by LENGTH from zero; truncating the axis under a bar makes a 0.03 gap
    read as a several-fold one. A dot encodes by POSITION, where zooming is
    honest. Panels then no longer share a y-axis, which is fine: levels are not
    comparable across judges anyway.
    """
    colors = colors or {}
    has_ci = all(lo is not None for _, _, rows in stats
                 for _, m, lo, hi, _n in rows if m is not None)
    hi_of = lambda m, hi: hi if hi is not None else m
    lo_of = lambda m, lo: lo if lo is not None else m
    top = max(hi_of(m, hi) for _, _, rows in stats
              for _, m, lo, hi, _n in rows if m is not None)
    fig, axes = plt.subplots(1, len(stats), sharey=not zoom, facecolor=SURFACE,
                             figsize=(1.15 * len(conds) * len(stats) + 1.6, 4.6))
    axes = [axes] if len(stats) == 1 else list(axes)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.22,
                        wspace=0.16 if zoom else 0.08)
    for ax, (ptitle, path, rows) in zip(axes, stats):
        ax.set_facecolor(SURFACE)
        vals = [m for _, m, *_ in rows if m is not None]
        p_lo = min(lo_of(m, lo) for _, m, lo, hi, _n in rows if m is not None)
        p_hi = max(hi_of(m, hi) for _, m, lo, hi, _n in rows if m is not None)
        span = max(p_hi - p_lo, 0.02)
        y0, y1 = (p_lo - 0.30 * span, p_hi + 0.30 * span) if zoom else (0, top * 1.16)
        pad = (y1 - y0) * 0.02

        base = next((m for c, m, *_ in rows if c == baseline and m is not None), None)
        if base is not None:
            ax.axhline(base, color=MUTED, ls=(0, (4, 3)), lw=1, zorder=1)
        for i, (c, m, lo, hi, n) in enumerate(rows):
            col = colors.get(c, BLUE)
            if m is None:
                ax.text(i, y0 + (y1 - y0) * 0.04, "not\njudged", ha="center",
                        va="bottom", fontsize=8, color=MUTED, linespacing=1.1)
                continue
            if zoom and marks == "bars":
                # Truncated bars, on request. Length no longer encodes value
                # from zero, so the axis break, the subtitle and the printed
                # value carry the honesty the bar length cannot.
                ax.bar(i, m - y0, bottom=y0, width=0.62, color=col, zorder=2)
                if lo is not None:
                    ax.plot([i, i], [lo, hi], color=INK, lw=1.6, zorder=3)
                ax.text(i, hi_of(m, hi) + pad, f"{m:.3f}", ha="center",
                        va="bottom", fontsize=8.5, color=INK)
            elif zoom:
                if lo is not None:
                    ax.plot([i, i], [lo, hi], color=col, lw=2.2, alpha=0.55,
                            zorder=3, solid_capstyle="round")
                ax.plot([i], [m], "o", ms=12, color=col, zorder=4,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
                ax.text(i, hi_of(m, hi) + pad * 2.2, f"{m:.3f}", ha="center",
                        va="bottom", fontsize=8.5, color=INK)
            else:
                ax.bar(i, m, width=0.62, color=col, zorder=2)
                if lo is not None:
                    ax.plot([i, i], [lo, min(hi, m)], color=SURFACE, lw=1.6,
                            zorder=3, solid_capstyle="round")
                    ax.plot([i, i], [max(lo, m), hi], color=INK, lw=1.6,
                            zorder=3, solid_capstyle="round")
                    for y in (lo, hi):
                        ax.plot([i - 0.1, i + 0.1], [y, y],
                                color=SURFACE if y < m else INK, lw=1.6, zorder=3)
                ax.text(i, hi_of(m, hi) + pad, f"{m:.3f}", ha="center",
                        va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c.replace("_", "_\n", 1) if len(c) > 9 else c
                            for c in conds], fontsize=8.5, color=INK)
        ax.set_xlim(-0.6, len(conds) - 0.4)
        ax.set_ylim(y0, y1)
        if zoom and marks == "bars":
            # axis-break mark at the foot of the y-axis: this axis is cut
            kw = dict(transform=ax.transAxes, color=INK2, clip_on=False, lw=1.0)
            for dy in (0.0, 0.03):
                ax.plot([-0.018, 0.018], [0.02 + dy, 0.045 + dy], **kw)
        ax.set_title(ptitle, loc="left", fontsize=10.5, color=INK, pad=6)
        ax.tick_params(axis="y", labelsize=8, colors=INK2)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=AXIS, lw=0.6, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
    axes[0].set_ylabel(METRICS[metric], fontsize=9, color=INK2)
    fig.suptitle(title, x=0.012, y=0.965, ha="left", fontsize=13, color=INK,
                 weight="bold")
    if zoom and marks == "bars":
        lead = ("bars are means, lines are bootstrap 95% CIs" if has_ci
                else "bars are means (no intervals computed)")
        lead += "; y-axes do NOT start at 0 and are zoomed per panel"
    elif zoom:
        lead = ("dots are means, lines are bootstrap 95% CIs" if has_ci
                else "dots are means (no intervals computed)")
        lead += "; each panel's axis is zoomed to its own range"
    else:
        lead = ("whiskers are bootstrap 95% CIs on the mean" if has_ci
                else "means only (no intervals computed)")
    fig.text(0.012, 0.875, f"{lead}; dashed line = {baseline} in that panel; "
             "compare arms WITHIN a panel", ha="left", fontsize=8.5, color=INK2)
    tail = ("Overlapping CIs do not mean a difference is insignificant"
            if has_ci else "Mark positions alone do not establish a difference")
    fig.text(0.012, 0.025, "Judges differ in generosity, so levels are not "
             f"comparable across panels. {tail}: the paired difference is the test.",
             ha="left", fontsize=7.5, color=MUTED)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Bar chart of mean score per condition")
    ap.add_argument("--scores", default=None)
    ap.add_argument("--panel", action="append", default=[],
                    help="'Title=scores.csv', repeatable: one panel per file "
                         "(e.g. one per judge), shared y-axis. Overrides --scores.")
    ap.add_argument("--conditions", default="baseline,merge_v2,merge_v2_rand")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--metric", default="coverage", choices=sorted(METRICS),
                    help="coverage = per answer (the benchmark's score); "
                         "union_coverage = union@K")
    ap.add_argument("--title", default="Coverage by condition")
    ap.add_argument("--ymin", type=float, default=0.0,
                    help="lower y limit. ANY value > 0 switches bars to dots: "
                         "bar LENGTH encodes magnitude from zero, so a "
                         "truncated bar chart makes a 0.05 gap look like 3x. "
                         "Dots encode by POSITION, where zooming is honest.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.panel:
        specs = []
        for item in args.panel:
            t, sep, p = item.partition("=")
            if not sep:
                ap.error(f"--panel {item!r} must be 'Title=path'")
            specs.append((t.strip(), p.strip()))
        conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
        plot_panels(specs, conds, args.baseline, args.metric, args.title,
                    args.out or "condition_bars_panels.png")
        return
    if not args.scores:
        ap.error("--scores or --panel is required")

    cov = load_coverage(args.scores, args.metric)
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
    ax.set_ylabel(METRICS[args.metric], fontsize=9.5, color=INK2)
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
