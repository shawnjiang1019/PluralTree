"""Bar chart of the GlobalOpinionQA steerable + distributional results.

Two panels, one per mode. Colour follows the CONDITION across panels: label and
label_mix share a colour, as do grounded and grounded_mix; the uniform guess is
a grey reference. Brackets carry the paired difference and p-value, because the
run reports intervals for differences, not for each bar.

Axes are zoomed per panel (bars do NOT start at 0): an axis-break mark and the
subtitle say so, and every bar prints its value, so length is never the only
reading of a number.

    # from the run's own rows (recomputes everything)
    python scripts/analysis/plot_goqa.py --jsonl goqa_steer.jsonl
    # from the numbers logged by job 6176604 (no data file needed)
    python scripts/analysis/plot_goqa.py --logged
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

# dataviz palette slots 1-3, validated all-pairs (light); grey = reference
C_NONE, C_LABEL, C_GROUND, C_REF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, MUTED, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#ffffff"
OURS = {"grounded", "grounded_mix"}          # tick labels set in bold

# Logged by evaluation/overton/eval_goqa.py, job 6176604
# (Qwen2.5-72B-Instruct-AWQ, World Values Survey, 150 questions x <=8 countries).
LOGGED = {
    "steer": {"none": 0.4003, "label": 0.4283, "grounded": 0.4520},
    "steer_d": [("grounded", "label", 0.0237, 0.009),
                ("label", "none", 0.0280, 0.004)],
    "dist": {"uniform": 0.5109, "none": 0.4043, "label_mix": 0.4738,
             "grounded_mix": 0.5638},
    "dist_d": [("grounded_mix", "label_mix", 0.0900, 0.0005),
               ("grounded_mix", "uniform", None, None)],
    "n_q": 150, "model": "Qwen2.5-72B-Instruct-AWQ",
}


def stats_from_jsonl(path: str, boot: int = 10000) -> dict:
    """Recompute exactly what eval_goqa.summarise prints, from the run's rows."""
    from cluster_overlap import paired_ci
    from evaluation.overton.eval_goqa import js_similarity
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        per[r["condition"]][r["qid"]].append(r["sim"])
    s = {c: {q: st.mean(v) for q, v in d.items()} for c, d in per.items()}

    byq = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        byq[r["qid"]][r["condition"]].append(r)
    mix = collections.defaultdict(dict)
    for q, conds in byq.items():
        target = [st.mean(c) for c in zip(*[r["true"] for r in conds["label"]])]
        k = len(target)
        mix["uniform"][q] = js_similarity([1 / k] * k, target)
        mix["none"][q] = js_similarity(conds["none"][0]["dist"], target)
        for c in ("label", "grounded"):
            m = [st.mean(col) for col in zip(*[r["dist"] for r in conds[c]])]
            mix[c + "_mix"][q] = js_similarity(m, target)

    def d(a, b, src):
        diff, lo, hi, p, n = paired_ci(src[a], src[b], boot)
        return (a, b, diff, p)
    return {"steer": {c: st.mean(s[c].values()) for c in ("none", "label", "grounded")},
            "steer_d": [d("grounded", "label", s), d("label", "none", s)],
            "dist": {c: st.mean(mix[c].values())
                     for c in ("uniform", "none", "label_mix", "grounded_mix")},
            "dist_d": [d("grounded_mix", "label_mix", mix),
                       d("grounded_mix", "uniform", mix)],
            "n_q": len(s["label"]), "model": os.path.basename(path)}


def _p(p) -> str:
    return "p<.001" if p < 0.001 else f"p={p:.3f}".replace("0.", ".", 1)


def plot(stats: dict, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("Steerable\none country's real answers",
         [("none", "question\nonly", C_NONE), ("label", "country\nlabel", C_LABEL),
          ("grounded", "grounded\n(ours)", C_GROUND)],
         stats["steer"], stats["steer_d"]),
        ("Distributional\nthe population's answers",
         [("uniform", "uniform\nguess", C_REF), ("none", "question\nonly", C_NONE),
          ("label_mix", "label\nmix", C_LABEL),
          ("grounded_mix", "grounded mix\n(ours)", C_GROUND)],
         stats["dist"], stats["dist_d"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [3, 4]})
    fig.subplots_adjust(left=0.08, right=0.99, top=0.72, bottom=0.18, wspace=0.22)

    for ax, (title, bars, vals, diffs) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        v = [vals[k] for k, *_ in bars]
        span = max(v) - min(v)
        y0 = min(v) - 0.45 * span
        y1 = max(v) + 0.95 * span                       # room for brackets
        pad = (y1 - y0) * 0.015
        pos = {k: i for i, (k, *_) in enumerate(bars)}
        for i, (k, lab, col) in enumerate(bars):
            ax.bar(i, vals[k] - y0, bottom=y0, width=0.62, color=col, zorder=2)
            ax.text(i, vals[k] + pad, f"{vals[k]:.3f}", ha="center", va="bottom",
                    fontsize=9, color=INK)
        # significance brackets, stacked so they never overlap
        level = max(v) + 0.30 * span                    # clear of value labels
        for a, b, diff, p in diffs:
            if diff is None:
                diff = vals[a] - vals[b]
            ia, ib = sorted((pos[a], pos[b]))
            ax.plot([ia, ia, ib, ib], [level - pad, level, level, level - pad],
                    color=INK2, lw=1.0, zorder=3)
            txt = f"{diff:+.3f}" + (f"  {_p(p)}" if p is not None else "")
            ax.text((ia + ib) / 2, level + pad, txt, ha="center", va="bottom",
                    fontsize=8.5, color=INK)
            level += 0.20 * span
        ax.set_ylim(y0, y1)
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels([lab for _, lab, _ in bars], fontsize=8.5, color=INK)
        for tl, (k, *_r) in zip(ax.get_xticklabels(), bars):
            if k in OURS:
                tl.set_fontweight("bold")
        ax.set_xlim(-0.6, len(bars) - 0.4)
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6,
                     linespacing=1.25)
        ax.tick_params(axis="y", labelsize=8, colors=INK2)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=AXIS, lw=0.6, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        kw = dict(transform=ax.transAxes, color=INK2, clip_on=False, lw=1.0)
        for dy in (0.0, 0.03):                          # axis-break mark
            ax.plot([-0.02, 0.02], [0.02 + dy, 0.045 + dy], **kw)
    axes[0].set_ylabel("1 − Jensen–Shannon distance", fontsize=9, color=INK2)

    fig.suptitle("GlobalOpinionQA: grounding personas in real survey data",
                 x=0.012, y=0.975, ha="left", fontsize=12.5, color=INK,
                 weight="bold")
    fig.text(0.012, 0.855,
             f"{stats['model']}, World Values Survey, {stats['n_q']} questions. "
             "Higher = closer to real answers.\ny-axes do NOT start at 0 and are "
             "zoomed per panel; brackets are paired differences.",
             ha="left", fontsize=8, color=INK2)
    fig.text(0.012, 0.025,
             "grounded = the persona also sees its group's real answers to the most "
             "similar OTHER question (near-duplicates, cosine ≥ 0.9, excluded).",
             ha="left", fontsize=7.5, color=MUTED)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(os.path.splitext(out)[0] + ext, dpi=200, facecolor=SURFACE)
    print(f"wrote {os.path.splitext(out)[0]}.png/.pdf")


def main() -> int:
    ap = argparse.ArgumentParser(description="GlobalOpinionQA result chart")
    ap.add_argument("--jsonl", default=None, help="eval_goqa.py output")
    ap.add_argument("--logged", action="store_true",
                    help="use the numbers logged by job 6176604")
    ap.add_argument("--out", default="docs/figures/goqa/goqa_results.png")
    args = ap.parse_args()
    if args.jsonl:
        stats = stats_from_jsonl(args.jsonl)
    elif args.logged:
        stats = LOGGED
    else:
        ap.error("pass --jsonl or --logged")
    plot(stats, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
