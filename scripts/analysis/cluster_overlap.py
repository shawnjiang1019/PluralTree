"""Which viewpoint clusters does each arm hit, and which does it uniquely find?

The scores csv carries only scalars, so it cannot explain the v12 union result:

  arm             score    union with baseline
  merge_v2       0.5440                 0.6848   (+0.0389)
  merge_v2_rand  0.5309                 0.7174   (+0.0773)
  persona_merge  0.5280                 0.7340   (+0.0917)

The arms score almost identically and differ hugely in what they ADD. Neither
variance (persona_merge has the LOWEST sd, 0.2369) nor correlation with baseline
explains the ordering -- both were checked and neither matches. Union depends on
WHICH clusters get covered, and that needs the sets, not the means.

`judge_overtonbench --dump_clusters` writes them. This reads that dump and asks:

  1. how many clusters each arm covers that baseline MISSES (the union gain,
     decomposed per question instead of averaged)
  2. whether those uniquely-covered clusters are MINORITY viewpoints -- which is
     what OvertonBench is built to reward, and the mechanism worth demonstrating
  3. how much the arms overlap each other, i.e. whether they are finding the
     same extra clusters or different ones

    python scripts/analysis/cluster_overlap.py --clusters overton_scores_v12_clusters.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics as st


def load(path: str):
    """(qid, cond) -> set of covered clusters, plus per-question cluster sizes."""
    covered = collections.defaultdict(set)
    all_clusters = collections.defaultdict(set)
    size = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            q, c, cl = int(r["question_id"]), r["condition"], int(r["cluster"])
            all_clusters[q].add(cl)
            size[(q, cl)] = int(r["cluster_size"])
            if int(r["covered"]):
                covered[(q, c)].add(cl)
    return covered, all_clusters, size


def main():
    ap = argparse.ArgumentParser(description="Per-cluster hit/miss by condition")
    ap.add_argument("--clusters", required=True,
                    help="csv from judge_overtonbench --dump_clusters")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--conditions", default=None,
                    help="comma list; default: everything except --baseline")
    args = ap.parse_args()

    covered, all_clusters, size = load(args.clusters)
    conds = sorted({c for _, c in covered})
    if args.conditions:
        conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    others = [c for c in conds if c != args.baseline]
    qs = sorted(all_clusters)
    print(f"{len(qs)} questions, conditions {conds}")

    # --- 1. what each arm adds to baseline, and whether it is minority ------
    print(f"\n=== clusters covered that {args.baseline} MISSES ===")
    print(f"  {'condition':<16}{'uniq/q':>9}{'missed/q':>10}{'recovered':>11}"
          f"{'prev of uniq':>14}{'prev of missed':>16}")
    for c in others:
        uniq, missed, prev_u, prev_m = [], [], [], []
        for q in qs:
            b = covered.get((q, args.baseline), set())
            x = covered.get((q, c), set())
            miss = all_clusters[q] - b                 # baseline's blind spot
            got = x - b                                # what this arm recovers
            uniq.append(len(got))
            missed.append(len(miss))
            n_p = max(1, max(size.get((q, cl), 0) for cl in all_clusters[q]))
            prev_u += [size.get((q, cl), 0) / n_p for cl in got]
            prev_m += [size.get((q, cl), 0) / n_p for cl in miss]
        rec = sum(uniq) / sum(missed) if sum(missed) else float("nan")
        print(f"  {c:<16}{st.mean(uniq):>9.2f}{st.mean(missed):>10.2f}"
              f"{rec:>11.1%}{(st.mean(prev_u) if prev_u else float('nan')):>14.3f}"
              f"{(st.mean(prev_m) if prev_m else float('nan')):>16.3f}")
    print("  recovered = of the clusters baseline missed, what fraction this arm got.")
    print("  prev = mean cluster size relative to the question's LARGEST cluster.")
    print("  If 'prev of uniq' sits well BELOW 'prev of missed', the arm is")
    print("  recovering the minority end of what baseline dropped -- the thing")
    print("  OvertonBench exists to measure. At or above it, the arm is picking up")
    print("  the easy majority clusters baseline happened to skip.")

    # --- 2. do the arms find the SAME extra clusters? -----------------------
    if len(others) >= 2:
        print(f"\n=== do the arms recover the same clusters? (Jaccard of "
              f"what each adds to {args.baseline}) ===")
        print(f"  {'':<16}" + "".join(f"{c:>16}" for c in others))
        for a in others:
            row = f"  {a:<16}"
            for b in others:
                inter = union = 0
                for q in qs:
                    base = covered.get((q, args.baseline), set())
                    ga = covered.get((q, a), set()) - base
                    gb = covered.get((q, b), set()) - base
                    inter += len(ga & gb)
                    union += len(ga | gb)
                row += f"{(inter / union if union else float('nan')):>16.3f}"
            print(row)
        print("  Low off-diagonal = the arms recover DIFFERENT clusters, so their")
        print("  gains would compound; high = they are finding the same ones and")
        print("  the 4-way union is closer to a ceiling than it looks.")

    # --- 3. clusters nothing covers ----------------------------------------
    n_never = n_tot = 0
    prev_never = []
    for q in qs:
        got = set().union(*(covered.get((q, c), set()) for c in conds)) if conds else set()
        never = all_clusters[q] - got
        n_never += len(never)
        n_tot += len(all_clusters[q])
        n_p = max(1, max(size.get((q, cl), 0) for cl in all_clusters[q]))
        prev_never += [size.get((q, cl), 0) / n_p for cl in never]
    print(f"\n=== clusters NO arm covers ===")
    print(f"  {n_never}/{n_tot} ({n_never / max(1, n_tot):.1%})   "
          f"mean relative prevalence {st.mean(prev_never) if prev_never else float('nan'):.3f}")
    print("  This is the real ceiling: no merge, router or reward can reach these.")
    print("  If their prevalence is low they are minority viewpoints every arm")
    print("  drops, which is a finding about the METHOD FAMILY, not about one arm.")


if __name__ == "__main__":
    main()
