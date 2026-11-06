"""Training curves from embedding logs: every logged value at every step.

Parses the .out files train.py writes (no --metrics_csv needed, so it works on
runs already finished) and, for one or more runs, produces:

  <out>.png         small multiples, one panel per quantity, one line per run
  <out>_eval.csv    per-eval rows: MRR, hits, ball occupancy, val Spearman(g, D)
  <out>_loss.csv    per-log-step rows: every loss term and the gate bias
  a printed table   final values and the step the selected checkpoint came from

One axis per panel on purpose: these quantities have unrelated scales. Loss lines
are smoothed for reading (--smooth, in logged points); the CSVs stay raw.

    python scripts/analysis/plot_train_log.py logs/embed_opinionqa_8471547.out logs/embed_opinionqa_8471546.out logs/embed_opinionqa_8471545.out --out train_curves
"""

from __future__ import annotations

import argparse
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"
# Validated categorical order, assigned by run position and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#6250d6", "#e34948"]

_NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
RE_LABEL = re.compile(r"\bEMB=(\S+)")
RE_STEP = re.compile(r"^\s*step\s+(\d+) \| (.*)$")
RE_EVAL = re.compile(rf"\[eval\] step (\d+) \| MRR {_NUM} \| H@1 {_NUM} \| H@10 {_NUM}")
RE_BALL = re.compile(rf"\[ball\]\s+rho mean {_NUM} \| p50 {_NUM} \| "
                     rf"frac>0\.9 {_NUM} \| pair-dist sd {_NUM}")
RE_DIS = re.compile(rf"\[dis\]\s+val Spearman\(geodesic, D\) {_NUM}")
RE_TEST = re.compile(rf"\bDIS \|.*test_dis={_NUM}")

EVAL_PANELS = [("val_dis", "Val Spearman(geodesic, D)"), ("mrr", "Val link MRR"),
               ("h10", "Val Hits@10"), ("rho_p50", "Radius, median"),
               ("rho_mean", "Radius, mean"), ("frac_rim", "Share of nodes, radius > 0.9"),
               ("pair_sd", "Pairwise distance sd")]
LOSS_PANELS = [("loss", "Total loss"), ("lp", "Link-prediction loss"),
               ("struct", "Hierarchy loss"), ("div", "Sibling-floor loss"),
               ("dis", "Disagreement loss")]


def parse_log(path: str) -> dict:
    """{label, eval: [row per eval], loss: [row per logged step], test_dis}."""
    run = {"path": path, "label": os.path.splitext(os.path.basename(path))[0],
           "eval": [], "loss": [], "test_dis": None}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if (m := RE_LABEL.search(line)) and not run["eval"] and not run["loss"]:
                run["label"] = os.path.splitext(m.group(1))[0].replace("embeddings_", "")
            elif m := RE_STEP.match(line):
                row = {"step": int(m.group(1))}
                for part in m.group(2).split("|"):
                    kv = part.split()
                    if len(kv) == 2:
                        try:
                            row[kv[0]] = float(kv[1])
                        except ValueError:
                            pass
                run["loss"].append(row)
            elif m := RE_EVAL.search(line):
                run["eval"].append({"step": int(m.group(1)), "mrr": float(m.group(2)),
                                    "h1": float(m.group(3)), "h10": float(m.group(4))})
            elif (m := RE_BALL.search(line)) and run["eval"]:
                run["eval"][-1].update(rho_mean=float(m.group(1)), rho_p50=float(m.group(2)),
                                       frac_rim=float(m.group(3)), pair_sd=float(m.group(4)))
            elif (m := RE_DIS.search(line)) and run["eval"]:
                run["eval"][-1]["val_dis"] = float(m.group(1))
            elif m := RE_TEST.search(line):
                run["test_dis"] = float(m.group(1))
    return run


def _smooth(ys: list[float], k: int) -> list[float]:
    if k <= 1:
        return ys
    out, acc = [], 0.0
    for i, y in enumerate(ys):
        acc += y
        if i >= k:
            acc -= ys[i - k]
        out.append(acc / min(i + 1, k))
    return out


def _write_csv(path: str, runs: list[dict], key: str) -> None:
    cols = ["run"] + sorted({c for r in runs for row in r[key] for c in row},
                            key=lambda c: (c != "step", c))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in runs:
            for row in r[key]:
                w.writerow({"run": r["label"], **row})


def plot(runs: list[dict], out: str, smooth: int) -> None:
    panels = ([("eval", k, t) for k, t in EVAL_PANELS]
              + [("loss", k, t) for k, t in LOSS_PANELS])
    panels = [p for p in panels
              if any(k_in(r[p[0]], p[1]) for r in runs)]
    ncol = 3
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.7 * nrow + 0.5),
                             facecolor=SURFACE, squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (src, key, title) in zip(axes.flat, panels):
        for i, r in enumerate(runs):
            pts = [(row["step"], row[key]) for row in r[src] if key in row]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ys = _smooth(list(ys), smooth) if src == "loss" else list(ys)
            ax.plot(xs, ys, color=SERIES[i], lw=2, solid_capstyle="round",
                    label=r["label"], zorder=3)
            if key == "val_dis":
                j = max(range(len(ys)), key=ys.__getitem__)
                ax.plot([xs[j]], [ys[j]], "o", ms=8, color=SERIES[i], zorder=4,
                        markeredgecolor=SURFACE, markeredgewidth=2)
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_xlabel("step", fontsize=8, color=MUTED)
    if len(runs) >= 2:
        h, lab = axes.flat[0].get_legend_handles_labels()
        leg = fig.legend(h, lab, loc="upper left", ncol=min(len(runs), 4),
                         frameon=False, fontsize=9, bbox_to_anchor=(0.01, 1.0))
        for t in leg.get_texts():
            t.set_color(INK)
    else:
        fig.suptitle(runs[0]["label"], x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.32 / (2.7 * nrow + 0.5)))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def k_in(rows: list[dict], key: str) -> bool:
    return any(key in r for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot every logged training value")
    ap.add_argument("logs", nargs="+", help="embedding job .out files")
    ap.add_argument("--out", default="train_curves", help="output prefix")
    ap.add_argument("--smooth", type=int, default=10, help="loss smoothing, logged points")
    args = ap.parse_args()
    if len(args.logs) > len(SERIES):
        ap.error(f"at most {len(SERIES)} runs per figure; a further hue would be "
                 "indistinguishable -- split them")

    runs = [parse_log(p) for p in args.logs]
    for r in runs:
        if not r["eval"]:
            print(f"warning: no [eval] lines in {r['path']} (still loading, or failed?)")
    runs = [r for r in runs if r["eval"] or r["loss"]]
    if not runs:
        raise SystemExit("nothing to plot")

    _write_csv(f"{args.out}_eval.csv", runs, "eval")
    _write_csv(f"{args.out}_loss.csv", runs, "loss")
    plot(runs, f"{args.out}.png", args.smooth)

    print(f"{'run':28} {'evals':>5} {'best val dis':>12} {'at step':>8} {'test dis':>9} "
          f"{'final MRR':>9} {'rho p50':>8} {'rho mean':>8} {'>0.9':>6} {'dist sd':>8}")
    for r in runs:
        ev = r["eval"]
        last = ev[-1] if ev else {}
        scored = [e for e in ev if "val_dis" in e]
        best = max(scored, key=lambda e: e["val_dis"]) if scored else None
        fmt = lambda v, s="{:.3f}": s.format(v) if v is not None else "--"
        print(f"{r['label']:28} {len(ev):5} "
              f"{fmt(best and best['val_dis'], '{:+.3f}'):>12} "
              f"{fmt(best and best['step'], '{}'):>8} "
              f"{fmt(r['test_dis'], '{:+.3f}'):>9} {fmt(last.get('mrr')):>9} "
              f"{fmt(last.get('rho_p50'), '{:.4f}'):>8} {fmt(last.get('rho_mean')):>8} "
              f"{fmt(last.get('frac_rim')):>6} {fmt(last.get('pair_sd')):>8}")
    print(f"-> {args.out}.png  {args.out}_eval.csv  {args.out}_loss.csv")


if __name__ == "__main__":
    main()
