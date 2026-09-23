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


def cluster_weights(all_clusters, size, q, mode: str) -> dict[int, float]:
    """Per-cluster weight for question q.

    The benchmark's own metric is `uniform`: a view held by 5% of participants
    counts exactly as much as one held by 60%. That is the scoring rule, and it
    is prevalence-BLIND -- which matters here because every injection arm trades
    majority clusters for minority ones (measured: gains at relative prevalence
    0.274, losses at 0.601). Under uniform weighting that trade nets zero by
    construction, so the property the method is built for is invisible.

      minority  coverage restricted to clusters under half the largest one
      invprev   weight 1/rel, so rare views dominate the average

    These are SECONDARY metrics, reported beside `uniform`, never instead of it:
    the benchmark's published numbers are uniform and comparisons must stay
    comparable to them.
    """
    n_p = max(1, max(size.get((q, cl), 0) for cl in all_clusters[q]))
    rel = {cl: size.get((q, cl), 0) / n_p for cl in all_clusters[q]}
    if mode == "uniform":
        return {cl: 1.0 for cl in all_clusters[q]}
    if mode == "minority":
        return {cl: (1.0 if rel[cl] < 0.5 else 0.0) for cl in all_clusters[q]}
    if mode == "invprev":
        return {cl: 1.0 / max(rel[cl], 1e-6) for cl in all_clusters[q]}
    raise ValueError(f"unknown weight mode {mode!r}")


def weighted_scores(covered, all_clusters, size, qs, cond, mode) -> dict[int, float]:
    """question -> weighted coverage fraction for one condition."""
    out = {}
    for q in qs:
        w = cluster_weights(all_clusters, size, q, mode)
        den = sum(w.values())
        if den <= 0:                       # no minority clusters on this question
            continue
        got = covered.get((q, cond), set())
        out[q] = sum(w[cl] for cl in got if cl in w) / den
    return out


def paired_ci(a: dict[int, float], b: dict[int, float], n_boot: int = 10000,
              seed: int = 0):
    """Paired bootstrap over QUESTIONS of mean(a) - mean(b).

    Paired on question id because the arms answer the same questions and the
    per-question variance dwarfs the effect (the reason every absolute-score
    comparison in this project is forbidden across runs).
    """
    import random
    qs = sorted(set(a) & set(b))
    if not qs:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    d = [a[q] - b[q] for q in qs]
    obs = st.mean(d)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = [d[rng.randrange(len(d))] for _ in range(len(d))]
        means.append(sum(s) / len(s))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    # Two-sided bootstrap p: twice the smaller tail on either side of zero. Both
    # tails use >=/<= so a degenerate bootstrap (every resample identical, which
    # happens when an arm ties the baseline on every question) counts in BOTH and
    # returns p=1 rather than a spurious p=0.
    below = sum(1 for m in means if m <= 0)
    above = sum(1 for m in means if m >= 0)
    p = min(1.0, 2.0 * min(below, above) / n_boot)
    return obs, lo, hi, p, len(qs)


def main():
    ap = argparse.ArgumentParser(description="Per-cluster hit/miss by condition")
    ap.add_argument("--clusters", required=True,
                    help="csv from judge_overtonbench --dump_clusters")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--conditions", default=None,
                    help="comma list; default: everything except --baseline")
    ap.add_argument("--weights", default="uniform,minority,invprev",
                    help="comma list of weighting schemes to score under. "
                         "uniform is the benchmark's own metric and is always "
                         "the one to quote; the others are secondary.")
    ap.add_argument("--boot", type=int, default=10000,
                    help="paired bootstrap resamples for the CI; 0 = skip")
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

    # --- 1b. the OTHER half of the ledger -----------------------------------
    # Section 1 reports only what an arm ADDS. The score is gain - loss, and the
    # loss column went uncomputed for the whole project: on the calibrated judge
    # merge_v2 gains 0.28 clusters/q and loses 1.00, which is the entire reversal.
    # Arms that tie on net turn out to have completely different ledgers (cover
    # gains 70% more than merge_v2 and loses 3x as much), so "null" was wrong.
    print(f"\n=== the full ledger (gain - loss vs {args.baseline}) ===")
    print(f"  {'condition':<16}{'gain/q':>9}{'loss/q':>9}{'net/q':>9}"
          f"{'net score':>11}{'prev gain':>11}{'prev loss':>11}")
    for c in others:
        g, l, tot, pg, pl = [], [], [], [], []
        for q in qs:
            b = covered.get((q, args.baseline), set())
            x = covered.get((q, c), set())
            if not x and not b:
                continue
            n_p = max(1, max(size.get((q, cl), 0) for cl in all_clusters[q]))
            g.append(len(x - b))
            l.append(len(b - x))
            tot.append(len(all_clusters[q]))
            pg += [size.get((q, cl), 0) / n_p for cl in x - b]
            pl += [size.get((q, cl), 0) / n_p for cl in b - x]
        if not tot:
            continue
        net = st.mean(g) - st.mean(l)
        print(f"  {c:<16}{st.mean(g):>9.2f}{st.mean(l):>9.2f}{net:>9.2f}"
              f"{net / st.mean(tot):>+11.3f}"
              f"{(st.mean(pg) if pg else float('nan')):>11.3f}"
              f"{(st.mean(pl) if pl else float('nan')):>11.3f}")
    print("  net score reproduces the reported coverage delta for the arm.")
    print("  prev gain << prev loss = the arm trades MAJORITY clusters for")
    print("  MINORITY ones. Under uniform weighting that nets zero by")
    print("  construction -- see the weighted scores below.")

    # --- 1c. coverage under each weighting, with paired CIs -----------------
    modes = [m.strip() for m in args.weights.split(",") if m.strip()]
    for mode in modes:
        base = weighted_scores(covered, all_clusters, size, qs, args.baseline, mode)
        print(f"\n=== coverage, {mode} weighting ===")
        print(f"  {'condition':<16}{'score':>8}{'delta':>9}"
              f"{'95% CI':>20}{'p':>8}{'n':>5}")
        print(f"  {args.baseline:<16}{(st.mean(base.values()) if base else float('nan')):>8.3f}")
        for c in others:
            arm = weighted_scores(covered, all_clusters, size, qs, c, mode)
            if not arm:
                continue
            d, lo, hi, p, n = (paired_ci(arm, base, args.boot) if args.boot
                               else (st.mean(arm) - st.mean(base), float("nan"),
                                     float("nan"), float("nan"), len(arm)))
            print(f"  {c:<16}{st.mean(arm.values()):>8.3f}{d:>+9.3f}"
                  f"{f'[{lo:+.3f}, {hi:+.3f}]':>20}{p:>8.3f}{n:>5}")
    print("  uniform is the benchmark's published metric -- quote that one.")
    print("  minority/invprev say whether an arm's gain is concentrated in the")
    print("  rare views, which uniform weighting cannot express. A delta that")
    print("  grows from uniform to minority is the pluralism claim; one that")
    print("  shrinks means the arm is picking up majority clusters.")

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
