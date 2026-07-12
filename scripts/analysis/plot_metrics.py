"""Plot validation learning curves from one or more metrics CSVs.

train.py --metrics_csv writes rows: step, epoch, val_mrr, hits@1, hits@3, hits@10.
This script overlays a chosen column across runs and saves a PNG (headless, no
display needed — works on the cluster).

Usage:
    # single run
    python scripts/plot_metrics.py metrics_up.csv

    # overlay the sweep, plot MRR
    python scripts/plot_metrics.py metrics_up.csv metrics_lat.csv metrics_updown.csv \
        metrics_both.csv --metric val_mrr --out mrr_curve.png

    # plot Hits@10 instead
    python scripts/plot_metrics.py metrics_*.csv --metric hits@10
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")          # headless backend — no display required
import matplotlib.pyplot as plt


def read_csv(path: str):
    steps, vals_by_col = [], {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = [c for c in reader.fieldnames if c not in ("step", "epoch")]
        for c in cols:
            vals_by_col[c] = []
        for row in reader:
            steps.append(int(row["step"]))
            for c in cols:
                vals_by_col[c].append(float(row[c]))
    return steps, vals_by_col


def main():
    p = argparse.ArgumentParser(description="Plot validation curves from metrics CSVs.")
    p.add_argument("csvs", nargs="+", help="One or more metrics CSV files.")
    p.add_argument("--metric", default="val_mrr",
                   help="Column to plot (val_mrr, hits@1, hits@3, hits@10).")
    p.add_argument("--out", default=None, help="Output PNG (default: <metric>.png).")
    p.add_argument("--xaxis", default="step", choices=["step"],
                   help="X axis (step).")
    args = p.parse_args()

    out = args.out or f"{args.metric.replace('@', '')}.png"

    plt.figure(figsize=(8, 5))
    for path in args.csvs:
        steps, cols = read_csv(path)
        if args.metric not in cols:
            print(f"  [skip] {path}: no column '{args.metric}' (has {list(cols)})")
            continue
        label = os.path.splitext(os.path.basename(path))[0]
        plt.plot(steps, cols[args.metric], marker="", label=label)

    plt.xlabel("training step")
    plt.ylabel(args.metric)
    plt.title(f"Validation {args.metric} over training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
