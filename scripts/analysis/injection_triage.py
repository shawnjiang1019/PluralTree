"""Which questions should we have injected on -- and where did merge_v2 lose?

delta_regressor.py PREDICTS the delta; this EXPLAINS it. Same labels, opposite
purpose: a per-question win/loss table plus what separates the losses from the
wins, so the routing feature set gets chosen from evidence rather than intuition.

THE REGRESSION-TO-THE-MEAN TRAP, and why this needs two runs.
The obvious hypothesis is "injection loses where baseline was already good" --
the v4/v5 consensus dilution, where forced enumeration turned committed answers
scoring 1.0 into hedged lists scoring 0.0. Testing that with the SAME run's
baseline is invalid: coverage is a noisy fraction over a handful of clusters, so
a question whose baseline scored high did so partly by luck and will regress
downward on any re-measurement, manufacturing a negative delta with no causal
content.

So the headline test uses an INDEPENDENT baseline -- v8's baseline coverage
predicting v9's delta. Both runs cover the same 60 questions and their noise is
independent, so a surviving correlation is real. The same-run version prints
beside it purely as the contrast, and the gap between them IS the artifact.

    python scripts/analysis/injection_triage.py \
        --scores overton_scores_v9.csv --condition merge_v2 \
        --baseline-from overton_scores_v8.csv
"""

from __future__ import annotations

import argparse
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.analysis.delta_regressor import (Row, attach_features, build_rows,
                                              derive_responses_path, features_in,
                                              load_coverage)


def _ok(v) -> bool:
    """Feature values are NaN wherever a source was unavailable (no graph, no
    --embeddings). NaN must be DROPPED, not treated as a measured 0."""
    return isinstance(v, (int, float)) and v == v


def _corr(xs, ys) -> float:
    if len(xs) < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


def _sign_p(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return float("nan")
    k = min(w, l)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)


def _fmt(v, nd=3):
    return "  n/a" if v != v else f"{v:+.{nd}f}"


def triage(rows, indep, top: int, tol: float) -> None:
    """Win/loss table + what separates them. ``indep`` maps qid -> baseline
    coverage from a DIFFERENT run; None disables the uncontaminated test."""
    rows = sorted(rows, key=lambda r: r.inj - r.base)
    deltas = [r.inj - r.base for r in rows]
    wins = [r for r in rows if r.inj - r.base > tol]
    losses = [r for r in rows if r.inj - r.base < -tol]
    ties = [r for r in rows if abs(r.inj - r.base) <= tol]

    print(f"\n=== {len(rows)} questions | tol={tol} (the measured noise floor) ===")
    print(f"  win {len(wins)}  loss {len(losses)}  tie {len(ties)}   "
          f"sign p={_sign_p(len(wins), len(losses)):.4f}   "
          f"mean delta={st.mean(deltas):+.4f}")
    print(f"  ties are {len(ties) / len(rows):.0%} of questions: coverage is a "
          f"fraction over few clusters, so it is coarse.")
    print("  That is a POWER limit, not a null result -- more rollouts, not more questions.")

    print(f"\n=== worst {top}: injection hurt most ===")
    print(f"  {'qid':>6}{'base':>8}{'inj':>8}{'delta':>9}")
    for r in rows[:top]:
        print(f"  {r.qid:>6}{r.base:>8.3f}{r.inj:>8.3f}{r.inj - r.base:>+9.3f}")

    print(f"\n=== best {top}: injection helped most ===")
    print(f"  {'qid':>6}{'base':>8}{'inj':>8}{'delta':>9}")
    for r in rows[-top:][::-1]:
        print(f"  {r.qid:>6}{r.base:>8.3f}{r.inj:>8.3f}{r.inj - r.base:>+9.3f}")

    # --- the headline test ---------------------------------------------------
    print("\n=== does injection lose where baseline was ALREADY GOOD? ===")
    same = _corr([r.base for r in rows], deltas)
    print(f"  same-run baseline    vs delta : {_fmt(same)}")
    print("    ^ CONTAMINATED: a high-by-luck baseline regresses down regardless.")
    if indep:
        pairs = [(indep[r.qid], r.inj - r.base) for r in rows if r.qid in indep]
        if len(pairs) >= 3:
            ind = _corr([a for a, _ in pairs], [b for _, b in pairs])
            print(f"  INDEPENDENT baseline vs delta : {_fmt(ind)}   over "
                  f"{len(pairs)} questions   <- THE REAL NUMBER")
            print(f"  artifact size = {_fmt(same - ind)}   (same-run minus independent)")
            print("  Clearly negative on the independent test => consensus dilution is")
            print("  real, and 'baseline already covers it' is a usable routing feature.")
            print("  Near zero => the losses are NOT explained by baseline quality;")
            print("  look to the feature table below instead.")
    else:
        print("  (pass --baseline-from <other run> for the uncontaminated test)")

    # --- what separates wins from losses -------------------------------------
    if not (wins and losses):
        return
    print(f"\n=== features: wins (n={len(wins)}) vs losses (n={len(losses)}) ===")
    print(f"  {'feature':<20}{'win mean':>11}{'loss mean':>11}{'diff':>9}"
          f"{'corr w/delta':>14}")
    scored = []
    for name in sorted({k for r in rows for k in r.feats if not k.startswith("_")}):
        w = [r.feats[name] for r in wins if _ok(r.feats.get(name))]
        l = [r.feats[name] for r in losses if _ok(r.feats.get(name))]
        if len(w) < 2 or len(l) < 2:
            continue
        pairs = [(r.feats[name], r.inj - r.base) for r in rows
                 if _ok(r.feats.get(name))]
        scored.append((abs(st.mean(w) - st.mean(l)), name, st.mean(w),
                       st.mean(l), _corr([a for a, _ in pairs],
                                         [b for _, b in pairs])))
    for _, name, wm, lm, c in sorted(scored, reverse=True):
        print(f"  {name:<20}{wm:>11.3f}{lm:>11.3f}{wm - lm:>+9.3f}{_fmt(c):>14}")
    print("  Ranked by separation. A feature that separates AND correlates is a")
    print("  routing candidate. One that separates WITHOUT correlating is probably")
    print("  tracking the win/loss split itself, not the quantity underneath it.")


def _selftest() -> None:
    """Reproduce the artifact honestly.

    The delta is CAUSED by the true baseline, but base and inj are each
    MEASURED with independent noise, so the observed delta = obs_inj - obs_base
    carries -e_base. That shared term is what inflates the same-run
    correlation. The real run has exactly this structure, and a fixture without
    it proves nothing -- the first version of this test built inj from obs_base
    and therefore could not reproduce the artifact at all.
    """
    import random
    rng = random.Random(0)
    rows, indep = [], {}
    for q in range(200):
        true_base = rng.uniform(0.1, 0.9)
        true_inj = true_base + 0.6 * (0.5 - true_base)     # the causal effect
        e_b, e_i = rng.gauss(0, 0.15), rng.gauss(0, 0.15)  # independent measurement
        indep[q] = max(0.0, min(1.0, true_base + rng.gauss(0, 0.05)))
        rows.append(Row(run="vT", qid=q, condition="c",
                        base=true_base + e_b, inj=true_inj + e_i,
                        feats={"true_base": true_base, "noise": rng.gauss(0, 1)}))

    same = _corr([r.base for r in rows], [r.inj - r.base for r in rows])
    ind = _corr([indep[r.qid] for r in rows], [r.inj - r.base for r in rows])
    print(f"\nselftest  same-run {same:+.3f}   independent {ind:+.3f}   "
          f"artifact {same - ind:+.3f}")
    assert same < ind, "same-run must be MORE negative; that gap IS the artifact"
    assert ind < -0.3, "the planted causal effect must survive the independent test"

    # No causal effect at all, only shared measurement noise. The same-run test
    # must STILL fire -- that is precisely the false positive it invites.
    flat = []
    for q in range(200):
        tb = rng.uniform(0.1, 0.9)
        flat.append(Row(run="vF", qid=q, condition="c",
                        base=tb + rng.gauss(0, 0.15),
                        inj=tb + rng.gauss(0, 0.15), feats={}))
    same_f = _corr([r.base for r in flat], [r.inj - r.base for r in flat])
    print(f"          NO-EFFECT control: same-run {same_f:+.3f} -- pure artifact, "
          f"zero causal effect")
    assert same_f < -0.3, "the artifact must appear even with no causal effect"

    triage(rows, indep, top=3, tol=0.027)
    print("\ninjection_triage selftest OK (artifact reproduced, separated, "
          "and shown to fire on a no-effect control)")


def main():
    ap = argparse.ArgumentParser(description="Which questions should we inject on?")
    ap.add_argument("--scores", help="scored run, e.g. overton_scores_v9.csv")
    ap.add_argument("--responses", default=None, help="default: derived from --scores")
    ap.add_argument("--condition", default="merge_v2")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--baseline-from", default=None,
                    help="ANOTHER run's scores CSV; its baseline column becomes the "
                         "uncontaminated predictor. Without it the headline test is "
                         "confounded by regression to the mean.")
    ap.add_argument("--groups", default="graph,response,trace")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--tol", type=float, default=0.027,
                    help="OvertonBench noise floor from two baseline draws")
    ap.add_argument("--contestedness", default=None,
                    help="external contestedness labels csv")
    ap.add_argument("--embeddings", default=None,
                    help="enables the live graph features (z_level, driver_sim)")
    ap.add_argument("--text_feat", default=None)
    ap.add_argument("--dataset", choices=["globalopinionqa", "opinionqa"],
                    default="opinionqa")
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--n_null", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42, help="graph split seed")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if not args.scores:
        ap.error("--scores required (or --selftest)")

    resp = args.responses or derive_responses_path(args.scores)
    rows, meta = build_rows([args.scores], [resp], args.baseline, [args.condition])
    rows = [r for r in rows if r.condition == args.condition]
    if not rows:
        ap.error(f"no rows for condition {args.condition!r} in {args.scores}")
    for tag, m in meta.items():
        print(f"{tag}: {m['n_q']} questions, conditions={m['conditions']}, "
              f"responses={m['responses'] or 'MISSING (text features -> NaN)'}")

    indep = None
    if args.baseline_from:
        cov = load_coverage(args.baseline_from)
        indep = {q: c[args.baseline] for q, c in cov.items() if args.baseline in c}
        print(f"independent baseline: {len(indep)} questions from {args.baseline_from}")

    names = features_in(args.groups.split(","))
    usable = attach_features(rows, names, args)
    print(f"features: {len(usable)}/{len(names)} usable -> {', '.join(sorted(usable))}")
    triage(rows, indep, args.top, args.tol)


if __name__ == "__main__":
    main()
