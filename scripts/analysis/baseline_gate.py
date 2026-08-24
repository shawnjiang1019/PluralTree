"""Gate injection on the BASELINE's own coverage, and score that honestly.

Every PRE-generation routing signal failed: 13 graph features flip sign between
the merge and merge_v2 arms, the CtrlA-style probe reads contestedness at AUC
0.692 but predicts the delta at only +0.194 (permutation p=0.130), and question
surface form separates wins from losses at p=0.222. The signal that does track
the outcome is not a property of the question at all -- it is how well the plain
answer already covered it:

    v10, independent baseline vs delta:  r = -0.433, p < 0.001
    buckets:  base 0 -> +0.203 (5W/0L)   0-0.5 -> +0.070   0.5-1 -> -0.000   1.0 -> -0.079

So: generate the baseline, look at it, THEN decide. Two passes, not a router.
merge_v2 already produces drafts, so the marginal cost is small.

TWO GATING SOURCES, AND THEY ANSWER DIFFERENT QUESTIONS.
  same-run    the baseline coverage measured in THIS run. That is what a
              deployed system would actually have -- but the same measurement
              noise appears in `delta = inj - base`, so a question that scored
              low by luck both trips the gate AND shows a large delta. Scoring
              the rule this way is optimistic.
  independent the baseline from a DIFFERENT run of the same questions. Its noise
              is uncorrelated with this run's delta, so the gain it reports is
              unbiased -- but it is not available at deployment time.
Both are printed. The gap is the selection bias, exactly as in injection_triage.

AND THE THRESHOLD IS CHOSEN, WHICH ALSO COSTS. `best fixed` is the maximum over
the sweep and is therefore an overfit ceiling. `LOO` re-picks the threshold on
the other 59 questions for each question scored, which is the number to report.
A permutation null redoes the whole search on shuffled baselines, so "the gate
helped" is tested against "any gate on any signal would have helped this much".

    python scripts/analysis/baseline_gate.py \
        --scores overton_scores_v10.csv --condition merge_v2 \
        --baseline-from overton_scores_v9.csv
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.analysis.delta_regressor import load_coverage


def _sign_p(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return float("nan")
    k = min(w, l)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)


def gate_score(qs, gate_on, thr: float) -> float:
    """Mean coverage under 'inject iff gate_on[q] < thr'."""
    return st.mean(qs[q]["inj"] if gate_on[q] < thr else qs[q]["base"] for q in qs)


def best_threshold(qs, gate_on, thrs, exclude=None) -> tuple[float, float]:
    """Threshold maximising mean coverage. ``exclude`` holds out one question so
    the caller can score it out-of-sample."""
    keys = [q for q in qs if q != exclude]
    best_t, best_s = thrs[0], -1.0
    for t in thrs:
        s = st.mean(qs[q]["inj"] if gate_on[q] < t else qs[q]["base"] for q in keys)
        if s > best_s:
            best_t, best_s = t, s
    return best_t, best_s


def loo_score(qs, gate_on, thrs) -> tuple[float, list]:
    """Leave-one-out: pick the threshold on the other n-1, score the held-out one.

    This is the number to report. `best fixed` sees every question it is scored
    on, so on 60 questions it will always look better than it is.
    """
    out, picks = [], []
    for q in qs:
        t, _ = best_threshold(qs, gate_on, thrs, exclude=q)
        picks.append(t)
        out.append(qs[q]["inj"] if gate_on[q] < t else qs[q]["base"])
    return st.mean(out), picks


def permutation_p(qs, gate_on, thrs, observed: float, n_perm: int, seed: int) -> float:
    """Reattach the gating values to the WRONG questions and redo the search.

    Tests the search, not a fixed threshold: if shuffled signals routinely reach
    `observed`, the gate found nothing that any arbitrary signal would not.
    """
    keys = list(qs)
    vals = [gate_on[q] for q in keys]
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(vals)
        shuffled = dict(zip(keys, vals))
        _, s = best_threshold(qs, shuffled, thrs)
        if s >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def report(qs, gate_on, label: str, thrs, n_perm: int, seed: int, always_inj: float):
    print(f"\n=== gating on {label} baseline ===")
    print(f"  {'thr':>6}{'score':>10}{'n_inject':>10}{'vs always-inject':>18}")
    for t in thrs:
        s = gate_score(qs, gate_on, t)
        n_inj = sum(1 for q in qs if gate_on[q] < t)
        print(f"  {t:>6.2f}{s:>10.4f}{n_inj:>10}{s - always_inj:>+18.4f}")

    bt, bs = best_threshold(qs, gate_on, thrs)
    lo, picks = loo_score(qs, gate_on, thrs)
    pv = permutation_p(qs, gate_on, thrs, bs, n_perm, seed)
    mode = max(set(picks), key=picks.count)
    print(f"  best fixed thr={bt:.2f} -> {bs:.4f}   (OVERFIT CEILING: chosen on "
          f"the same {len(qs)} questions it scores)")
    print(f"  LOO             -> {lo:.4f}   <- REPORT THIS   "
          f"(modal thr {mode:.2f}, {picks.count(mode)}/{len(picks)} folds)")
    print(f"  permutation p   = {pv:.4f} over {n_perm} shuffles "
          f"({'beats' if pv < 0.05 else 'does NOT beat'} an arbitrary signal)")

    # Paired: does the gate beat simply always injecting, question by question?
    w = l = 0
    for q in qs:
        g = qs[q]["inj"] if gate_on[q] < bt else qs[q]["base"]
        if g > qs[q]["inj"] + 1e-12:
            w += 1
        elif g < qs[q]["inj"] - 1e-12:
            l += 1
    print(f"  vs always-inject, paired: {w}W/{l}L/{len(qs) - w - l}T   "
          f"sign p={_sign_p(w, l):.4f}")
    return lo


def main():
    ap = argparse.ArgumentParser(description="Score a baseline-coverage gate")
    ap.add_argument("--scores", required=True)
    ap.add_argument("--condition", default="merge_v2")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--baseline-from", default=None,
                    help="another run's scores; enables the UNBIASED estimate")
    ap.add_argument("--thrs", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cov = load_coverage(args.scores)
    qs = {int(q): {"base": c[args.baseline], "inj": c[args.condition]}
          for q, c in cov.items()
          if args.baseline in c and args.condition in c}
    if not qs:
        ap.error(f"no questions with both {args.baseline!r} and {args.condition!r}")

    thrs = [float(t) for t in args.thrs.split(",")]
    always_base = st.mean(v["base"] for v in qs.values())
    always_inj = st.mean(v["inj"] for v in qs.values())
    oracle = st.mean(max(v["base"], v["inj"]) for v in qs.values())

    print(f"{args.scores}  |  {args.condition} vs {args.baseline}  |  "
          f"{len(qs)} questions")
    print(f"\n  always-baseline   {always_base:.4f}")
    print(f"  always-inject     {always_inj:.4f}   ({always_inj - always_base:+.4f})")
    print(f"  ORACLE            {oracle:.4f}   ({oracle - always_inj:+.4f} "
          f"headroom over always-inject)")

    same = {q: v["base"] for q, v in qs.items()}
    report(qs, same, "SAME-RUN (deployable, optimistic)", thrs,
           args.n_perm, args.seed, always_inj)

    if args.baseline_from:
        icov = load_coverage(args.baseline_from)
        indep = {int(q): c[args.baseline] for q, c in icov.items()
                 if args.baseline in c and int(q) in qs}
        missing = set(qs) - set(indep)
        if missing:
            print(f"\n  [warn] {len(missing)} questions absent from "
                  f"{args.baseline_from}; skipping them in the independent arm")
            sub = {q: v for q, v in qs.items() if q in indep}
        else:
            sub = qs
        ai = st.mean(v["inj"] for v in sub.values())
        report(sub, indep, "INDEPENDENT (unbiased, not deployable)", thrs,
               args.n_perm, args.seed, ai)
        print("\n  The same-run arm is what you could ship; the independent arm is")
        print("  what the rule is really worth. If same-run is much higher, the")
        print("  difference is selection on this run's noise, not signal.")


if __name__ == "__main__":
    main()
