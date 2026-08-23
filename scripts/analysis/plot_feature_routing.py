"""Which features could route injection? One plot, two panels, honest error bars.

injection_triage.py prints the same numbers as text. This exists because the
text table invites two mistakes that a plot can make structurally hard:

  1. RANKING BY RAW DIFFERENCE. The text table's `diff` column puts r_chars
     (+54.7 CHARACTERS) above g_z_level (+1.676 STANDARD DEVIATIONS) purely
     because characters are a bigger unit. Panel B standardizes to Cohen's d so
     the comparison means something.

  2. READING A POINT ESTIMATE AS A RESULT. At n=60 the 95% CI on a correlation
     is about +/-0.25 wide, so almost every feature's interval covers zero.
     Both panels draw the CI and shade the region that is indistinguishable
     from zero, so "nothing here is significant" is visible rather than
     something you have to work out from the numbers.

COST is the color, because it decides what a feature can be used FOR:
  pre       computable BEFORE generating       -> can route
  baseline  needs the plain answer first       -> reranker only
  post      needs the injected answer          -> analysis only
A `post` feature with a beautiful correlation is still useless for routing, and
colouring by cost keeps that from being forgotten halfway down a ranked list.

    python scripts/analysis/plot_feature_routing.py \
        --scores overton_scores_v9.csv --condition merge_v2 \
        --embeddings embeddings_opinionqa.pt --out docs/feature_routing_v9.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")                      # headless compute node
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.analysis.delta_regressor import (FEATURES, attach_features, build_rows,
                                              derive_responses_path, features_in)
from scripts.analysis.injection_triage import _corr, _ok

# dataviz reference palette, categorical slots 1-3 (validated all-pairs, light).
# Marker shape repeats the encoding so cost survives greyscale and CVD.
COST_STYLE = {
    "pre":      ("#2a78d6", "o", "pre - can route"),
    "baseline": ("#eb6834", "s", "baseline - reranker only"),
    "post":     ("#1baf7a", "^", "post - analysis only"),
}
INK, INK2, MUTED, AXIS = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
SURFACE = "#fcfcfb"


def _t_crit(df: int) -> float:
    """Two-sided 5% t critical value. Table for small df, 1.96 asymptote."""
    tab = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
           8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086,
           25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}
    if df <= 0:
        return float("nan")
    for k in sorted(tab):
        if df <= k:
            return tab[k]
    return 1.96


def _fisher_ci(r: float, n: int, z: float = 1.96):
    """95% CI on a correlation, via the Fisher z transform."""
    if n < 4 or not -1 < r < 1:
        return float("nan"), float("nan")
    zr = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    return tuple(math.tanh(zr + s * z * se) for s in (-1, 1))


def _r_crit(n: int, k: int = 1) -> float:
    """|r| needed for p<0.05 at n, Bonferroni-corrected over k features.

    k>1 widens the bar via a normal approximation on the Fisher scale -- exact
    enough to draw a line that says "the best of 13 needs to clear THIS", which
    is the only thing the line is for.
    """
    if n < 4:
        return float("nan")
    if k <= 1:
        t = _t_crit(n - 2)
        return t / math.sqrt(t * t + n - 2)
    p = 0.05 / (2 * k)
    z = math.sqrt(2 * math.log(1 / p)) - 0.5   # adequate for p in [1e-4, 0.05]
    return math.tanh(z / math.sqrt(n - 3))


def _cohens_d(a, b):
    """Standardized mean difference with a pooled SD, plus its 95% CI."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return (float("nan"),) * 3
    va, vb = st.variance(a), st.variance(b)
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if sp <= 0:
        return (float("nan"),) * 3
    d = (st.mean(a) - st.mean(b)) / sp
    se = math.sqrt((na + nb) / (na * nb) + d * d / (2 * (na + nb - 2)))
    t = _t_crit(na + nb - 2)
    return d, d - t * se, d + t * se


def collect(rows, names, tol: float):
    """Per-feature: correlation with delta, Cohen's d wins-vs-losses, and n.

    Each feature is scored on its OWN non-NaN rows -- no complete-matrix
    requirement, because nothing here fits a joint model.
    """
    wins = [r for r in rows if r.inj - r.base > tol]
    losses = [r for r in rows if r.inj - r.base < -tol]
    out = []
    for name in names:
        pairs = [(r.feats[name], r.inj - r.base) for r in rows if _ok(r.feats.get(name))]
        if len(pairs) < 4:
            continue
        xs = [a for a, _ in pairs]
        if max(xs) - min(xs) < 1e-12:            # constant: nothing to measure
            continue
        w = [r.feats[name] for r in wins if _ok(r.feats.get(name))]
        l = [r.feats[name] for r in losses if _ok(r.feats.get(name))]
        r_val = _corr(xs, [b for _, b in pairs])
        lo, hi = _fisher_ci(r_val, len(pairs))
        d, d_lo, d_hi = _cohens_d(w, l)
        out.append({"name": name, "cost": FEATURES[name]["cost"],
                    "r": r_val, "r_lo": lo, "r_hi": hi, "n": len(pairs),
                    "d": d, "d_lo": d_lo, "d_hi": d_hi,
                    "n_win": len(w), "n_loss": len(l),
                    "doc": FEATURES[name]["doc"]})
    out.sort(key=lambda s: abs(s["r"]) if s["r"] == s["r"] else -1)
    return out


def _panel(ax, stats, key, lo_key, hi_key, crit, crit_k, title, xlabel, n_note):
    y = range(len(stats))
    # Null band FIRST so marks sit on top of it.
    if crit == crit:
        ax.axvspan(-crit, crit, color=MUTED, alpha=0.13, lw=0, zorder=0)
    if crit_k == crit_k:
        for s in (-1, 1):
            ax.axvline(s * crit_k, color=MUTED, ls=(0, (4, 3)), lw=1, zorder=1)
    ax.axvline(0, color=AXIS, lw=1, zorder=2)

    for i, s in enumerate(stats):
        color, marker, _ = COST_STYLE.get(s["cost"], (MUTED, "o", ""))
        v, lo, hi = s[key], s[lo_key], s[hi_key]
        if v != v:
            continue
        if lo == lo and hi == hi:
            ax.plot([lo, hi], [i, i], color=color, lw=2, solid_capstyle="round",
                    zorder=3, alpha=0.55)
        ax.plot([v], [i], marker=marker, ms=9, color=color, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.5)

    ax.set_yticks(list(y))
    ax.set_yticklabels([s["name"] for s in stats], fontsize=9, color=INK)
    ax.set_ylim(-0.7, len(stats) - 0.3)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.tick_params(axis="x", labelsize=8, colors=INK2)
    ax.grid(axis="x", color=AXIS, lw=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.text(0.0, -0.10, n_note, transform=ax.transAxes, fontsize=8,
            color=MUTED, ha="left", va="top")


def plot(stats, out: str, subtitle: str, tol: float) -> None:
    n_corr = max((s["n"] for s in stats), default=0)
    n_w = max((s["n_win"] for s in stats), default=0)
    n_l = max((s["n_loss"] for s in stats), default=0)
    k = len(stats)
    r_c, r_bonf = _r_crit(n_corr), _r_crit(n_corr, k)
    t = _t_crit(n_w + n_l - 2)
    d_c = t * math.sqrt(1 / n_w + 1 / n_l) if n_w > 1 and n_l > 1 else float("nan")

    h = max(3.6, 0.34 * len(stats) + 3.0)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, h), sharey=True,
                             facecolor=SURFACE)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.78, bottom=0.235,
                        wspace=0.10)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    _panel(axes[0], stats, "r", "r_lo", "r_hi", r_c, r_bonf,
           "A - Correlation with coverage delta",
           "Pearson r   (bars = 95% CI)",
           f"shaded: |r| < {r_c:.2f}, not distinguishable from 0 at n={n_corr}\n"
           f"dashed: {r_bonf:.2f}, Bonferroni bar for picking the best of {k}")
    _panel(axes[1], stats, "d", "d_lo", "d_hi", d_c, float("nan"),
           "B - Separation, wins vs losses",
           "Cohen's d   (bars = 95% CI)",
           f"shaded: |d| < {d_c:.2f}, not distinguishable from 0 "
           f"at {n_w} wins / {n_l} losses")
    axes[1].tick_params(axis="y", length=0)

    fig.suptitle("Can any feature decide when to inject?", x=0.0135, y=0.955,
                 ha="left", fontsize=15, color=INK, weight="bold")
    fig.text(0.0135, 0.902, subtitle, ha="left", fontsize=9.5, color=INK2)
    fig.text(0.0135, 0.028,
             "Sorted by |r|. A routing feature must be BLUE (computable before "
             "generating) and clear the shaded band in both panels.\n"
             f"Win/loss split at |delta| > {tol}, the measured coverage noise floor.",
             ha="left", fontsize=8.5, color=MUTED)

    handles = [Line2D([], [], marker=m, ls="", ms=9, color=c,
                      markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
               for c, m, lab in COST_STYLE.values()]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.968),
               frameon=False, fontsize=9, ncol=3, handletextpad=0.4,
               columnspacing=1.4, labelcolor=INK2)

    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Plot feature usefulness for routing")
    ap.add_argument("--scores", help="required unless --selftest")
    ap.add_argument("--responses", default=None)
    ap.add_argument("--condition", default="merge_v2")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--groups", default="graph,response,trace")
    ap.add_argument("--tol", type=float, default=0.027)
    ap.add_argument("--out", default=None,
                    help="default: docs/feature_routing_<run>_<cond>.png")
    ap.add_argument("--contestedness", default=None)
    ap.add_argument("--embeddings", default=None,
                    help="required for g_z_level / g_driver_sim")
    ap.add_argument("--text_feat", default=None)
    ap.add_argument("--dataset", choices=["globalopinionqa", "opinionqa"],
                    default="opinionqa")
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--n_null", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if not args.scores:
        ap.error("--scores required (or --selftest)")

    resp = args.responses or derive_responses_path(args.scores)
    rows, _ = build_rows([args.scores], [resp], args.baseline, [args.condition])
    rows = [r for r in rows if r.condition == args.condition]
    if not rows:
        ap.error(f"no rows for condition {args.condition!r} in {args.scores}")

    names = features_in(args.groups.split(","))
    attach_features(rows, names, args)          # prints its own drops
    stats = collect(rows, names, args.tol)
    if not stats:
        ap.error("no feature had enough non-NaN, non-constant values to score")

    tag = os.path.basename(args.scores).replace("overton_scores_", "").replace(".csv", "")
    out = args.out or f"docs/feature_routing_{tag}_{args.condition}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    sub = (f"{args.scores}  |  condition={args.condition}  |  "
           f"{len(rows)} questions  |  {len(stats)} features scored")
    plot(stats, out, sub, args.tol)

    # The contrast WARN on the aqua slot obliges a table view; it is also just
    # easier to quote a number from a csv than to read one off a dot.
    csv_path = os.path.splitext(out)[0] + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(stats[0].keys()))
        wr.writeheader()
        wr.writerows(stats)
    print(f"wrote {csv_path}")
    print(f"\n  {'feature':<18}{'cost':>10}{'r':>8}{'95% CI':>16}{'d':>8}{'n':>5}")
    for s in reversed(stats):
        ci = f"[{s['r_lo']:+.2f},{s['r_hi']:+.2f}]"
        print(f"  {s['name']:<18}{s['cost']:>10}{s['r']:>+8.3f}{ci:>16}"
              f"{s['d']:>+8.2f}{s['n']:>5}")


def _selftest() -> None:
    """Plant one real feature among noise; it must be the only one that clears.

    Uses the SAME Row shape the real path builds, so the plotting code is
    exercised end to end rather than against a hand-made stats list.
    """
    import random
    from scripts.analysis.delta_regressor import Row
    rng = random.Random(0)
    rows = []
    for q in range(60):
        signal = rng.gauss(0, 1)
        delta = 0.45 * signal + rng.gauss(0, 0.35)   # strong, but not perfect
        base = 0.5
        r = Row(run="vT", qid=q, condition="c", base=base, inj=base + delta)
        r.feats = {"g_z_level": signal, "g_w_raw": rng.gauss(0, 1),
                   "r_chars": rng.gauss(400, 90) * 1.0, "t_words": 7.0}
        rows.append(r)

    stats = collect(rows, ["g_z_level", "g_w_raw", "r_chars", "t_words"], 0.027)
    got = [s["name"] for s in stats]
    assert "t_words" not in got, "a constant feature must be dropped, not scored"
    assert stats[-1]["name"] == "g_z_level", f"planted feature must rank first: {got}"
    assert stats[-1]["r_lo"] > 0, "planted feature's CI must exclude zero"
    assert any(s["r_lo"] < 0 < s["r_hi"] for s in stats if s["name"] == "g_w_raw"), \
        "pure noise must NOT clear zero"

    out = os.path.join(os.environ.get("TMPDIR", "."), "_routing_selftest.png")
    plot(stats, out, "selftest | planted signal in g_z_level", 0.027)
    assert os.path.getsize(out) > 10000, "figure looks empty"
    print("plot_feature_routing selftest OK "
          f"({len(stats)} scored, planted feature ranked first)")


if __name__ == "__main__":
    main()
