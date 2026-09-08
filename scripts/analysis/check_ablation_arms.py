"""Did the ablation arms actually ablate? Exit 2 when they did not.

A job exiting 0 does not mean the experiment ran. merge_v2_flat returning zero
forks, or merge_v2_divrand drawing the same pairs as merge_v2, both produce a
complete responses file, a clean judge pass and a plausible score table -- in
which case the arm is a baseline clone and a tie between arms means nothing.
That is the failure merge_v2_rand actually had for three runs: 1.00 forks per
row against merge_v2's 4.85, plus a dispatch gap that sent it to merge v1.

So this asserts the manipulation, and is meant to gate the real runs:

    sbatch --dependency=afterok:<smoke id> ...

Reads the `pair_select` trace dict written by retrieval.answer for every
retrieved condition -- mode, n_forks, mean_w, mean_relevance, anchor_pool.

    python scripts/analysis/check_ablation_arms.py \\
        --responses overton_responses_sel.jsonl --strict

Exit 0 = arms differ as designed. Exit 2 = they do not (a real result about the
run, not a crash). Exit 1 = the file is unreadable or a condition is missing.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys


def load(path: str):
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  warning: {bad} unparseable lines")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Verify the ablation arms differ")
    ap.add_argument("--responses", required=True)
    ap.add_argument("--baseline", default="merge_v2",
                    help="the arm the ablations are measured against")
    ap.add_argument("--strict", action="store_true",
                    help="also fail when an ablation is INERT (equal mean_w). "
                         "Off for a 3-question smoke, where a tie is possible "
                         "by chance; ON for the real run, where it is fatal.")
    args = ap.parse_args()

    rows = load(args.responses)
    if not rows:
        print(f"FAIL: {args.responses} has no rows"); return 1

    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("condition", "?")].append(r)
    print(f"{len(rows)} rows, {len(by)} conditions: {sorted(by)}")

    problems: list[str] = []

    # --- every condition produced real answers ------------------------------
    print("\n=== responses are non-empty ===")
    for cond, rs in sorted(by.items()):
        empty = sum(1 for r in rs if not (r.get("response") or "").strip())
        ok = empty == 0
        print(f"  [{'ok' if ok else 'FAIL'}] {cond:20} {len(rs):4} rows, "
              f"{empty} empty")
        if not ok:
            problems.append(f"{cond}: {empty} empty responses")

    # --- the manipulation itself -------------------------------------------
    print("\n=== pair_select: did retrieval actually differ? ===")
    stats = {}
    for cond, rs in sorted(by.items()):
        ps = [r["pair_select"] for r in rs if r.get("pair_select")]
        if not ps:
            if cond != "baseline":
                print(f"  [--]   {cond:20} no pair_select "
                      f"(fine for a non-retrieved arm)")
            continue
        nf = [p["n_forks"] for p in ps]
        w = [p["mean_w"] for p in ps if p.get("mean_w") is not None]
        rel = [p["mean_relevance"] for p in ps
               if p.get("mean_relevance") is not None]
        stats[cond] = {"mode": ps[0].get("mode"), "rows": len(ps),
                       "n_forks": st.mean(nf) if nf else 0.0,
                       "zero_fork_rows": sum(1 for n in nf if n == 0),
                       "mean_w": st.mean(w) if w else None,
                       "rel": st.mean(rel) if rel else None}
        s = stats[cond]
        print(f"  {cond:20} mode={str(s['mode']):8} n_forks={s['n_forks']:5.2f} "
              f"mean_w={s['mean_w'] if s['mean_w'] is None else round(s['mean_w'], 4)} "
              f"rel={s['rel'] if s['rel'] is None else round(s['rel'], 4)} "
              f"({s['zero_fork_rows']} rows with 0 forks)")

    base = stats.get(args.baseline)
    if base is None:
        print(f"\nFAIL: no {args.baseline} rows to compare against"); return 1

    # An arm that retrieved nothing is a baseline clone wearing another name.
    print("\n=== every retrieved arm actually retrieved ===")
    for cond, s in sorted(stats.items()):
        ok = s["n_forks"] > 0
        print(f"  [{'ok' if ok else 'FAIL'}] {cond:20} mean n_forks {s['n_forks']:.2f}")
        if not ok:
            problems.append(f"{cond}: retrieved 0 forks -- it is a baseline clone")

    # Volume must match or the arm confounds content with quantity.
    print(f"\n=== fork VOLUME matches {args.baseline} ===")
    for cond, s in sorted(stats.items()):
        if cond == args.baseline:
            continue
        d = abs(s["n_forks"] - base["n_forks"])
        ok = d <= 0.25
        print(f"  [{'ok' if ok else 'FAIL'}] {cond:20} {s['n_forks']:.2f} vs "
              f"{base['n_forks']:.2f}  (delta {d:.2f})")
        if not ok:
            problems.append(f"{cond}: {s['n_forks']:.2f} forks vs "
                            f"{args.baseline}'s {base['n_forks']:.2f} -- volume "
                            f"moves with the ablation, exactly merge_v2_rand's bug")

    # divrand specifically: maxw takes the argmax, so a uniform draw over the
    # same pool MUST sit below it. Equal means the draw picked the same pairs.
    print("\n=== divergence selection was really randomised ===")
    dr = stats.get("merge_v2_divrand")
    if dr is None:
        print("  [--]   merge_v2_divrand not in this run")
    elif dr["mean_w"] is None or base["mean_w"] is None:
        problems.append("merge_v2_divrand: no mean_w to compare")
        print("  [FAIL] no mean_w recorded")
    else:
        lower = dr["mean_w"] < base["mean_w"]
        equal = abs(dr["mean_w"] - base["mean_w"]) < 1e-9
        print(f"  divrand mean_w {dr['mean_w']:.4f} vs {args.baseline} "
              f"{base['mean_w']:.4f}")
        if dr["mean_w"] > base["mean_w"] + 1e-9:
            problems.append("merge_v2_divrand: mean_w ABOVE maxw -- impossible "
                            "if both rank the same pool; the selection is wired "
                            "backwards")
            print("  [FAIL] random selection beat the argmax -- wiring is wrong")
        elif equal:
            msg = ("merge_v2_divrand: mean_w identical to maxw -- the random "
                   "draw picked the same pairs, so the ablation is INERT and a "
                   "tie between the arms would mean nothing")
            print(f"  [{'FAIL' if args.strict else 'warn'}] inert")
            (problems if args.strict else []).append(msg) if args.strict else \
                print(f"         {msg}")
        else:
            print("  [ok]   below the argmax, as it must be")
        if dr["rel"] is not None and base["rel"] is not None:
            drift = abs(dr["rel"] - base["rel"])
            # Same anchors, so relevance should barely move. This is what
            # separates divrand from merge_v2_rand, whose whole point is to be
            # LESS relevant -- check_random_fork.py would assert the opposite.
            print(f"  [{'ok' if drift <= 0.05 else 'warn'}] relevance drift "
                  f"{drift:.4f} (same anchors -> expect ~0)")

    print("\n" + "=" * 64)
    if problems:
        print(f"{len(problems)} PROBLEM(S) -- the run did not test what it claims:")
        for p in problems:
            print(f"  - {p}")
        return 2
    print("ablation arms verified: all retrieved, volume matched, "
          "selection really randomised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
