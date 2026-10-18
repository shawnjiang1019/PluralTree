"""Main-result forest plot: each arm's paired coverage delta against a reference.

One row per comparison. The line is the 95% paired-bootstrap interval over
questions, the marker is the point estimate -- FILLED when the interval excludes
zero, HOLLOW when it does not, so significance is never carried by colour alone.
A Delta [CI] column sits on the right, and the same numbers are written to a CSV
beside the figure so a table view always exists.

Numbers come from cluster_overlap.py's own functions (load / weighted_scores /
paired_ci), so the plot shows exactly what that script prints. Everything must
come from ONE cluster dump: participant subsampling makes scores from separate
judging passes incomparable, so this script refuses to mix files.

    # current data: every arm against the plain answer, calibrated judge
    python scripts/analysis/plot_main_result.py \
        --clusters overton_scores_v11_q38.csv --reference baseline \
        --conditions "merge_v2=merge_v2,persona_merge=persona_merge" \
        --out docs/figures/main_result.pdf

    # the paper's design, once plain_merge has been generated and judged with
    # the others: drafting vs baseline, then each method vs plain merge
    python scripts/analysis/plot_main_result.py \
        --clusters overton_scores_lad_q38.csv \
        --contrasts "plain_merge-baseline=Plain merge vs baseline,\
merge_v2-plain_merge=merge_v2 vs plain merge,\
persona_merge-plain_merge=persona_merge vs plain merge" \
        --out docs/figures/main_result.pdf

    python scripts/analysis/plot_main_result.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_overlap import load, paired_ci, weighted_scores  # noqa: E402

# Validated with the dataviz palette checker (light mode, #fcfcfb surface):
# lightness band, chroma floor and >=3:1 contrast all pass. One series, so no
# legend -- the axis label and title name it.
SERIES = "#2a78d6"
SURFACE = "#ffffff"          # print figure: page white
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
ZERO = "#8a8984"


def parse_pairs(spec: str) -> list[tuple[str, str]]:
    """'key=Label,key2=Label 2' -> [(key, label), ...]; a bare key is its own label."""
    out = []
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        key, _, label = item.partition("=")
        out.append((key.strip(), (label or key).strip()))
    return out


def _is_cluster_dump(path: str) -> bool:
    with open(path, newline="", encoding="utf-8") as f:
        return "cluster" in (csv.DictReader(f).fieldnames or [])


def score_source(path: str, metric: str, weight: str):
    """-> (present_conditions, fn(cond) -> {question_id: score}).

    WHICH NUMBER: the judge's scores CSV carries, per question, both
      coverage        mean over rollouts of ONE answer's coverage (OvertonScore,
                      the benchmark's metric; metric="within")
      union_coverage  a cluster counts if ANY rollout covers it (metric="union")
    The --dump_clusters file stores only the UNION set per (question, arm) --
    judge_overtonbench.py builds `covered` as set().union(*per_resp) -- so it can
    give union@K but never within-answer coverage. Plotting a cluster dump as if
    it were the benchmark score overstated both levels and gaps (merge_v2 +7.8
    union@3 against the +5-6 within-answer headline). Hence the guard below.
    """
    if _is_cluster_dump(path):
        if metric == "within":
            raise SystemExit(
                f"{path} is a cluster dump: it holds union@K sets only, not "
                f"within-answer coverage. Pass the judge's scores CSV (the file "
                f"without _clusters), or --metric union to plot union@K on purpose.")
        covered, all_clusters, size = load(path)
        qs = sorted(all_clusters)
        return ({c for _, c in covered},
                lambda c: weighted_scores(covered, all_clusters, size, qs, c, weight))
    if weight != "uniform":
        raise SystemExit("--weight minority/invprev needs per-cluster sets, which "
                         "only the cluster dump has (and those are union@K).")
    col = "coverage" if metric == "within" else "union_coverage"
    by: dict[str, dict[int, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by.setdefault(r["condition"], {})[int(r["question_id"])] = float(r[col])
    return set(by), lambda c: by.get(c, {})


def compute_rows(clusters_csv: str, contrasts: list[tuple[str, str, str]],
                 weight: str = "uniform", boot: int = 10000,
                 metric: str = "within") -> list[dict]:
    """contrasts: [(arm, reference, label)] -> one row per contrast, paired on question."""
    present, scores = score_source(clusters_csv, metric, weight)
    rows = []
    for arm, ref, label in contrasts:
        for c in (arm, ref):
            if c not in present:
                raise SystemExit(f"condition {c!r} not in {clusters_csv} "
                                 f"(has: {', '.join(sorted(present))})")
        a = scores(arm)
        b = scores(ref)
        d, lo, hi, p, n = paired_ci(a, b, boot)
        rows.append({"label": label, "arm": arm, "reference": ref,
                     "weight": weight, "delta": d, "lo": lo, "hi": hi,
                     "p": p, "n": n})
    return rows


def compute_scores(clusters_csv: str, arms: list[tuple[str, str]],
                   weight: str = "uniform", boot: int = 10000,
                   seed: int = 0, metric: str = "within") -> list[dict]:
    """Absolute coverage per arm: mean over questions, with a bootstrap interval.

    The interval resamples questions for ONE arm, so it is NOT paired. Two arms'
    intervals can overlap while their paired difference is significant -- the
    paired comparison is the delta plot, and this figure is only the anchor that
    says what the deltas are deltas OF. Rows reuse the delta keys (delta = the
    score) so plot_panels draws both kinds.
    """
    import random
    present, scores = score_source(clusters_csv, metric, weight)
    rng = random.Random(seed)
    rows = []
    for arm, label in arms:
        if arm not in present:
            raise SystemExit(f"condition {arm!r} not in {clusters_csv} "
                             f"(has: {', '.join(sorted(present))})")
        vals = list(scores(arm).values())
        n = len(vals)
        mean = sum(vals) / n
        means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n
                       for _ in range(boot))
        rows.append({"label": label, "arm": arm, "reference": "", "weight": weight,
                     "delta": mean, "lo": means[int(0.025 * boot)],
                     "hi": means[int(0.975 * boot) - 1], "p": float("nan"),
                     "n": n})
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["panel", "label", "arm", "reference",
                                          "weight", "delta", "lo", "hi", "p", "n"],
                           restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})


def _fmt_p(p: float) -> str:
    return "p<.001" if p < 0.001 else f"p={p:.3f}".replace("0.", ".", 1)


def plot_panels(panels: list[tuple[str, list[dict]]], out: str, xlabel: str,
                width_in: float = 3.3, show_p: bool = False,
                kind: str = "delta", ref_arm: str | None = None) -> None:
    """Stacked forest panels on ONE shared x-axis; total width ``width_in``.

    Each panel is (title, rows). Panels come from separate judging passes (e.g.
    one per judge), so each panel's deltas are computed within its own file --
    only the axis is shared, which is what makes a sign change readable at a
    glance. Labels and the Delta column are measured after drawing and the plot
    area is fitted between them, so nothing spills past the column width.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    row_h = 0.40           # inches per comparison row
    title_h = 0.24         # inches above a panel for its title
    gap_h = 0.10           # inches between panels
    bottom_in = 0.46       # ticks + x label under the last panel
    top_in = 0.04
    heights = [row_h * len(rows) for _, rows in panels]
    has_title = [bool(t) for t, _ in panels]
    height_in = (bottom_in + top_in + sum(heights)
                 + sum(title_h for h in has_title if h) + gap_h * (len(panels) - 1))
    fig = plt.figure(figsize=(width_in, height_in))
    fig.patch.set_facecolor(SURFACE)

    # One x-range for every panel, so the same delta sits at the same place.
    all_rows = [r for _, rows in panels for r in rows]
    lo_all = min(r["lo"] for r in all_rows)
    hi_all = max(r["hi"] for r in all_rows)
    span = max(hi_all, 0) - min(lo_all, 0)
    pad = 0.08 * span if span > 0 else 0.05
    xmin, xmax = min(lo_all, 0) - pad, max(hi_all, 0) + pad
    if kind == "score":                    # magnitudes: the axis starts at 0
        xmin, xmax = 0.0, min(1.0, hi_all + 0.08 * hi_all)

    axes, notes = [], []
    y_top = height_in - top_in
    for i, ((title, rows), h) in enumerate(zip(panels, heights)):
        if title:
            y_top -= title_h
        ax = fig.add_axes([0.3, (y_top - h) / height_in, 0.4, h / height_in])
        y_top -= h + gap_h
        ax.set_facecolor(SURFACE)
        last = i == len(panels) - 1

        n = len(rows)
        ys = list(range(n))[::-1]                   # first contrast on top
        if kind == "score":
            # Dotted line at the reference arm's score: the deltas in the
            # companion figure are distances from this line.
            ref = [r for r in rows if r["arm"] == ref_arm]
            if ref:
                ax.axvline(ref[0]["delta"], color=ZERO, lw=0.8,
                           ls=(0, (1, 1.5)), zorder=1)
        else:
            ax.axvline(0, color=ZERO, lw=0.8, ls=(0, (3, 2)), zorder=1)
        for y, r in zip(ys, rows):
            ax.plot([r["lo"], r["hi"]], [y, y], color=SERIES, lw=2,
                    solid_capstyle="round", zorder=2)
            # Filled/hollow encodes "interval excludes zero", which only means
            # something for a delta. Scores are always filled.
            sig = kind == "score" or r["lo"] > 0 or r["hi"] < 0
            ax.plot(r["delta"], y, marker="o", ms=6.5, mew=1.6, zorder=3,
                    mfc=SERIES if sig else SURFACE, mec=SERIES)

        ax.set_yticks(ys)
        # Wrap long labels at " vs " so they fit a single column.
        ax.set_yticklabels([r["label"].replace(" vs ", "\nvs ") for r in rows],
                           color=TEXT_PRIMARY, linespacing=1.1)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(xmin, xmax)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.tick_params(axis="x", colors=TEXT_SECONDARY, length=2.5 if last else 0,
                       width=0.6, labelbottom=last)
        ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.spines["bottom"].set_visible(last)
        if kind == "score":
            from matplotlib.ticker import FuncFormatter
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        if last:
            ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
        if title:
            ax.set_title(title, loc="left", fontsize=8.5, color=TEXT_PRIMARY,
                         pad=5)

        # Delta [CI] column, in text ink -- never the series colour. p lives in
        # the CSV by default; the filled/hollow marker carries significance.
        for y, r in zip(ys, rows):
            if kind == "score":
                txt = (f"{r['delta']:.1%}\n"
                       f"[{r['lo'] * 100:.1f}, {r['hi'] * 100:.1f}]")
            else:
                txt = f"{r['delta']:+.3f}\n[{r['lo']:+.3f}, {r['hi']:+.3f}]"
                txt = txt.replace("-", "−")      # true minus, as on the axis
                if show_p:
                    txt += f"\n{_fmt_p(r['p'])}"
            notes.append(ax.annotate(
                txt, xy=(1.0, y), xycoords=("axes fraction", "data"),
                xytext=(5, 0), textcoords="offset points", va="center",
                ha="left", fontsize=6.5, color=TEXT_SECONDARY, linespacing=1.15))
        axes.append(ax)

    # Measure every label block and Delta column, then fit ALL panels to the
    # same left/right edges so their x-axes line up exactly.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    dpi = fig.dpi
    lab_w = max(t.get_window_extent(rend).width
                for ax in axes for t in ax.get_yticklabels()) / dpi
    col_w = max(t.get_window_extent(rend).width for t in notes) / dpi
    left = (lab_w + 0.10) / width_in
    right = 1.0 - (col_w + 5 / 72 + 0.04) / width_in
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([left, pos.y0, max(right - left, 0.15), pos.height])

    fig.savefig(out, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def plot_forest(rows: list[dict], out: str, xlabel: str, title: str = "",
                width_in: float = 3.3, show_p: bool = False) -> None:
    """Single-panel forest plot (see plot_panels)."""
    plot_panels([(title, rows)], out, xlabel, width_in, show_p)


def _selftest(outdir: str) -> None:
    """Synthetic dump with known effects: the plot must place each arm correctly."""
    import random
    rng = random.Random(0)
    path = os.path.join(outdir, "_selftest_clusters.csv")
    sizes = [40, 30, 12, 6, 3]
    # per-arm probability of covering each cluster index
    probs = {"baseline":      [0.8, 0.6, 0.2, 0.1, 0.1],
             "plain_merge":   [0.8, 0.6, 0.3, 0.2, 0.1],
             "merge_v2":      [0.6, 0.4, 0.2, 0.1, 0.1],
             "persona_merge": [0.8, 0.7, 0.5, 0.4, 0.3]}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "condition", "cluster", "cluster_size", "covered"])
        for q in range(60):
            for cond, pr in probs.items():
                for i, s in enumerate(sizes):
                    w.writerow([q, cond, i, s, int(rng.random() < pr[i])])
    contrasts = [("plain_merge", "baseline", "Plain merge vs baseline"),
                 ("merge_v2", "plain_merge", "merge_v2 vs plain merge"),
                 ("persona_merge", "plain_merge", "persona_merge vs plain merge")]
    rows = compute_rows(path, contrasts, boot=2000, metric="union")
    by = {r["arm"]: r for r in rows}
    assert by["persona_merge"]["lo"] > 0, by["persona_merge"]      # clearly positive
    assert by["merge_v2"]["hi"] < 0, by["merge_v2"]                # clearly negative
    out = os.path.join(outdir, "_selftest_main_result.png")
    plot_forest(rows, out, "Δ coverage (paired, 95% CI)",
                title="Synthetic fixture")
    write_csv(rows, out.replace(".png", ".csv"))
    print(f"selftest OK -> {out}")
    for r in rows:
        print(f"  {r['label']:32} {r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] "
              f"{_fmt_p(r['p'])}")

    # Two panels, mimicking a judge swap: merge_v2 flips sign, persona does not.
    # Each panel is its own file (its own judging pass); only the axis is shared.
    judge_probs = {
        "old": {"baseline":      [0.8, 0.6, 0.2, 0.1, 0.1],
                "merge_v2":      [0.9, 0.7, 0.4, 0.2, 0.1],
                "persona_merge": [0.9, 0.7, 0.4, 0.2, 0.2]},
        "new": {"baseline":      [0.7, 0.5, 0.2, 0.1, 0.1],
                "merge_v2":      [0.5, 0.3, 0.1, 0.1, 0.1],
                "persona_merge": [0.8, 0.6, 0.3, 0.2, 0.1]},
    }
    panels = []
    for name, pr_by in judge_probs.items():
        pth = os.path.join(outdir, f"_selftest_clusters_{name}.csv")
        with open(pth, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["question_id", "condition", "cluster", "cluster_size",
                        "covered"])
            for q in range(60):
                for cond, pr in pr_by.items():
                    for i, s in enumerate(sizes):
                        w.writerow([q, cond, i, s, int(rng.random() < pr[i])])
        jrows = compute_rows(pth, [("merge_v2", "baseline", "merge_v2 vs baseline"),
                                   ("persona_merge", "baseline",
                                    "persona_merge vs baseline")], boot=2000,
                             metric="union")
        panels.append((f"{name.capitalize()} judge", jrows))
    old = {r["arm"]: r for r in panels[0][1]}
    new = {r["arm"]: r for r in panels[1][1]}
    assert old["merge_v2"]["delta"] > 0 > new["merge_v2"]["delta"], "no sign flip"
    assert new["persona_merge"]["delta"] > 0, new["persona_merge"]
    out2 = os.path.join(outdir, "_selftest_by_judge.png")
    plot_panels(panels, out2, "Δ coverage vs baseline (paired, 95% CI)")
    print(f"selftest two-panel OK -> {out2}")

    # Absolute scores from the same two files: a score panel's differences must
    # reproduce the delta panel exactly, since both are means over one question set.
    arms = [("baseline", "baseline"), ("merge_v2", "merge_v2"),
            ("persona_merge", "persona_merge")]
    spanels = []
    for name in judge_probs:
        pth = os.path.join(outdir, f"_selftest_clusters_{name}.csv")
        spanels.append((f"{name.capitalize()} judge",
                        compute_scores(pth, arms, boot=2000, metric="union")))
    for (_, srows), (_, drows) in zip(spanels, panels):
        sc = {r["arm"]: r["delta"] for r in srows}
        for d in drows:
            got = sc[d["arm"]] - sc["baseline"]
            assert abs(got - d["delta"]) < 1e-9, (d["arm"], got, d["delta"])
    out3 = os.path.join(outdir, "_selftest_scores_by_judge.png")
    plot_panels(spanels, out3, "Clusters covered (95% CI)", kind="score",
                ref_arm="baseline")
    print(f"selftest scores OK (differences reproduce the deltas) -> {out3}")

    # within-answer: the benchmark's metric, read from a scores CSV.
    spath = os.path.join(outdir, "_selftest_scores.csv")
    with open(spath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "question_id", "coverage", "union_coverage",
                    "positions_per_answer", "n_rollouts", "n_clusters"])
        for q in range(60):
            base = 0.4 + 0.2 * rng.random()
            w.writerow(["baseline", q, base, base + 0.1, 2, 3, 5])
            w.writerow(["persona_merge", q, base + 0.05, base + 0.2, 2, 3, 5])
    wrow = compute_rows(spath, [("persona_merge", "baseline", "p")], boot=2000)[0]
    assert abs(wrow["delta"] - 0.05) < 1e-9, wrow          # reads `coverage`
    urow = compute_rows(spath, [("persona_merge", "baseline", "p")], boot=2000,
                        metric="union")[0]
    assert abs(urow["delta"] - 0.10) < 1e-9, urow          # reads `union_coverage`
    # A cluster dump holds union sets only; asking it for within must refuse.
    try:
        compute_rows(os.path.join(outdir, "_selftest_clusters_old.csv"),
                     [("merge_v2", "baseline", "m")], boot=10)
    except SystemExit:
        pass
    else:
        raise AssertionError("cluster dump accepted for within-answer coverage")
    print("selftest metric OK: within reads `coverage`, union reads "
          "`union_coverage`, a cluster dump is refused for within")


def main() -> int:
    ap = argparse.ArgumentParser(description="Main-result forest plot")
    ap.add_argument("--clusters", help="ONE cluster dump (judge --dump_clusters)")
    ap.add_argument("--panel", action="append", default=[],
                    help="'Title=path', repeatable: one stacked panel per cluster "
                         "dump (e.g. one per judge), same contrasts, shared x-axis. "
                         "Overrides --clusters.")
    ap.add_argument("--reference", default="baseline",
                    help="reference arm for --conditions")
    ap.add_argument("--conditions", default="",
                    help="'arm=Label,...' each compared against --reference")
    ap.add_argument("--contrasts", default="",
                    help="'arm-ref=Label,...' explicit pairs; overrides --conditions")
    ap.add_argument("--metric", default="within", choices=["within", "union"],
                    help="within = the benchmark's OvertonScore (mean over "
                         "rollouts of one answer's coverage), read from the "
                         "judge's scores CSV. union = a cluster counts if ANY "
                         "rollout covers it (union@K); a cluster dump gives only this.")
    ap.add_argument("--weight", default="uniform",
                    choices=["uniform", "minority", "invprev"],
                    help="uniform is the benchmark's metric and the one to lead with")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--xlabel", default="Δ coverage vs reference (paired, 95% CI)")
    ap.add_argument("--title", default="")
    ap.add_argument("--width", type=float, default=3.3,
                    help="figure width in inches (3.3 = one ACL column)")
    ap.add_argument("--show_p", action="store_true",
                    help="print p under each interval (always in the CSV)")
    ap.add_argument("--absolute", action="store_true",
                    help="plot each arm's own coverage (%%) instead of paired "
                         "deltas. --conditions lists the arms to show, baseline "
                         "included; --reference gets a dotted line. Intervals are "
                         "per-arm, NOT paired: use the delta plot for tests.")
    ap.add_argument("--out", default="docs/figures/main_result.pdf")
    ap.add_argument("--selftest", nargs="?", const=".", metavar="OUTDIR")
    args = ap.parse_args()

    if args.selftest:
        _selftest(args.selftest)
        return 0
    if not args.clusters and not args.panel:
        ap.error("--clusters or --panel is required (or pass --selftest)")

    if args.absolute:
        arms = parse_pairs(args.conditions)
        if not arms:
            ap.error("--absolute needs --conditions listing the arms to show")
        contrasts = []
    elif args.contrasts:
        contrasts = []
        for pair, label in parse_pairs(args.contrasts):
            arm, sep, ref = pair.partition("-")
            if not sep:
                ap.error(f"contrast {pair!r} must be 'arm-ref'")
            contrasts.append((arm.strip(), ref.strip(), label))
    else:
        conds = parse_pairs(args.conditions)
        if not conds:
            ap.error("pass --conditions or --contrasts")
        contrasts = [(arm, args.reference, label) for arm, label in conds]

    if args.panel:
        specs = []
        for item in args.panel:
            title, sep, path = item.partition("=")
            if not sep:
                ap.error(f"--panel {item!r} must be 'Title=path'")
            specs.append((title.strip(), path.strip()))
    else:
        specs = [(args.title, args.clusters)]

    panels, flat = [], []
    for title, path in specs:
        rows = (compute_scores(path, arms, args.weight, args.boot,
                               metric=args.metric) if args.absolute
                else compute_rows(path, contrasts, args.weight, args.boot,
                                  metric=args.metric))
        for r in rows:
            r["panel"] = title
        panels.append((title, rows))
        flat += rows

    # Defaults always NAME the metric: a union@K figure must never read as the
    # benchmark's within-answer score.
    xlabel = args.xlabel
    if xlabel == ap.get_default("xlabel"):
        if args.metric == "union":
            xlabel = ("Clusters covered by any rollout (union@K)" if args.absolute
                      else "Δ union@K coverage vs reference (paired 95% CI)")
        else:
            xlabel = ("Clusters covered per answer (95% CI)" if args.absolute
                      else "Δ coverage per answer vs reference (paired 95% CI)")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_panels(panels, args.out, xlabel, args.width, args.show_p,
                kind="score" if args.absolute else "delta",
                ref_arm=args.reference)
    csv_out = os.path.splitext(args.out)[0] + ".csv"
    write_csv(flat, csv_out)
    for title, rows in panels:
        if title:
            print(f"[{title}]")
        for r in rows:
            if args.absolute:
                print(f"  {r['label']:32} {r['delta']:.1%} "
                      f"[{r['lo']:.1%}, {r['hi']:.1%}]  n={r['n']}")
            else:
                print(f"  {r['label']:32} {r['delta']:+.3f} [{r['lo']:+.3f}, "
                      f"{r['hi']:+.3f}] {_fmt_p(r['p'])}  n={r['n']}")
    print(f"-> {args.out}\n-> {csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
