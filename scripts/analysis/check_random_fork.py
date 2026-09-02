"""Manipulation check for the merge_v2_rand control (docs/random_fork_control.md).

The control claims to hold everything fixed except fork RELEVANCE. Two ways that
claim fails silently, and both are checkable from the response rows alone:

  relevance  If the sampled forks are nearly as relevant as the retrieved ones,
             there is no manipulation and the arm is a noisy copy of merge_v2.
             `random_forks` draws from unrelated subtrees, but a graph whose
             topics overlap heavily could defeat that.
  length     If the random block is much shorter or longer than the real one,
             the arm varies prompt length as well as relevance, and any
             difference is uninterpretable.
  count      Same for the number of forks. The first version of the sampler
             returned ONE random fork and compared it against the top real fork,
             reporting "ratio 0.97" while the arms injected 4.85 vs 1.00 forks
             (4256 vs 859 chars). Both are now checked on the WHOLE block.

Exits non-zero when either fails, so a chained analysis job stops instead of
reporting deltas nobody should read.

    python scripts/analysis/check_random_fork.py --responses overton_responses_v11.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the random-fork control")
    ap.add_argument("--responses", required=True)
    ap.add_argument("--min_rel_gap", type=float, default=0.10,
                    help="required mean drop in fork relevance")
    ap.add_argument("--max_len_ratio", type=float, default=1.50,
                    help="max mean rendered-length ratio, either direction")
    args = ap.parse_args()

    stats, skipped = [], 0
    with open(args.responses, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rf = row.get("random_fork")
            if not rf:
                continue
            if rf.get("randomized"):
                stats.append(rf)
            else:
                skipped += 1

    if not stats:
        print("no randomized rows found -- was merge_v2_rand in --conditions?")
        return 1

    rel_real = st.mean(s["rel_real"] for s in stats)
    rel_rand = st.mean(s["rel_rand"] for s in stats)
    len_real = st.mean(s["len_real"] for s in stats)
    len_rand = st.mean(s["len_rand"] for s in stats)
    n_cand = st.mean(s["n_candidates"] for s in stats)
    ratio = len_rand / len_real if len_real else float("inf")
    n_real = st.mean(s.get("n_real", 0) for s in stats)
    n_rand = st.mean(s.get("n_rand", 0) for s in stats)

    print(f"  randomized rows      {len(stats)}   (skipped, no match: {skipped})")
    print(f"  candidates/question  {n_cand:.1f}")
    print(f"  fork relevance       real {rel_real:.3f}  ->  random {rel_rand:.3f}"
          f"   gap {rel_real - rel_rand:+.3f}")
    print(f"  forks per question   real {n_real:.2f}  ->  random {n_rand:.2f}")
    print(f"  rendered length      real {len_real:.0f}  ->  random {len_rand:.0f}"
          f"   ratio {ratio:.2f}")

    ok = True
    if rel_real - rel_rand < args.min_rel_gap:
        print(f"  FAIL relevance gap {rel_real - rel_rand:+.3f} < "
              f"{args.min_rel_gap} -- the 'irrelevant' forks are not irrelevant, "
              f"so this arm is not a control")
        ok = False
    if not (1 / args.max_len_ratio) <= ratio <= args.max_len_ratio:
        print(f"  FAIL length ratio {ratio:.2f} outside "
              f"[{1 / args.max_len_ratio:.2f}, {args.max_len_ratio:.2f}] -- "
              f"prompt length is confounded with relevance")
        ok = False
    if n_real and abs(n_rand - n_real) > 0.5:
        print(f"  FAIL fork count {n_real:.2f} vs {n_rand:.2f} -- the arms inject "
              f"different VOLUMES, so relevance is confounded with how much "
              f"context there is")
        ok = False
    if skipped > len(stats) * 0.25:
        print(f"  WARN {skipped} questions had no comparable unrelated anchor; "
              f"the arms are paired on a biased subset")

    print(f"  {'PASS' if ok else 'FAIL'} -- "
          f"{'deltas are interpretable' if ok else 'do NOT interpret the deltas'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
