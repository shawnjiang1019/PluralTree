"""Is the "middle" real? Do a fork's non-pole subgroups lie BETWEEN the two poles?

The scout injects only the two most-divergent subgroups (branch_a/branch_b) of an
anchor axis. The pole-collapse study (docs/framing_hurts.png) showed the answer
then collapses onto those two extremes. The content-fix -- inject the FULL subgroup
spectrum -- only makes sense if the other subgroups actually form a graded middle
rather than piling at the two poles (a true binary). This script measures that,
graph-only (no embeddings, no LLM).

For each axis with >=3 unique subgroup leaves:
  poles      the max-total-variation subgroup pair (approximates scout's max-W pair)
  for each other subgroup: project its option-distribution onto the pole->pole line
    t     position along A->B (0=A, 1=B); "middle" iff 0.2<t<0.8
    off   off-axis residual (0 = exactly on the A-B line); "on the spectrum" iff <0.5
Reports the fraction of non-pole subgroups in the middle, and the share of axes
that are bimodal (nothing in the middle). Options are categorical, so total
variation (0.5*L1) is the ground metric -- no OT/POT needed.

Run (needs the ATP data; set OPINIONQA_DIR):
    OPINIONQA_DIR=.../human_resp python -m evaluation.intrinsic.subtree_middle
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser(description="Subgroup-middle diagnostic")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_pole_tv", type=float, default=0.15,
                    help="axis counts as contested iff its pole pair TV >= this")
    ap.add_argument("--t_lo", type=float, default=0.2)
    ap.add_argument("--t_hi", type=float, default=0.8)
    ap.add_argument("--max_off", type=float, default=0.5)
    args = ap.parse_args()

    import numpy as np
    from data.loaders.opinionqa import load_opinionqa

    g = load_opinionqa(split_seed=args.seed, leakage_safe=True)
    axes = [i for i, n in enumerate(g.id_to_entity) if n.startswith("ax:")]

    def vec(nid):
        d = g.opinion_dist.get(nid)
        return np.array(d, float) if d else None

    con, n_notcon = [], 0
    for a in axes:
        leaves = list(dict.fromkeys(k for k in g.children_indices[a] if vec(k) is not None))
        if len(leaves) < 3:
            continue
        P = {k: vec(k) for k in leaves}
        if len({len(v) for v in P.values()}) != 1:      # ragged options
            continue
        A, B = max(combinations(leaves, 2),
                   key=lambda p: 0.5 * np.abs(P[p[0]] - P[p[1]]).sum())
        pole_tv = 0.5 * float(np.abs(P[A] - P[B]).sum())
        if pole_tv < args.min_pole_tv:
            n_notcon += 1
            continue
        av = P[B] - P[A]
        den = float(av @ av)
        mids = []
        for k in leaves:
            if k in (A, B):
                continue
            d = P[k] - P[A]
            t = float(d @ av / den)
            off = float(np.linalg.norm(d - t * av) / np.sqrt(den))
            mids.append((t, off))
        if not mids:
            continue
        inm = sum(1 for t, off in mids if args.t_lo < t < args.t_hi and off < args.max_off)
        con.append({"n_sub": len(leaves), "pole_tv": pole_tv,
                    "mid_frac": inm / len(mids),
                    "mean_off": float(np.mean([o for _, o in mids]))})

    mf = np.array([r["mid_frac"] for r in con])
    print(f"axes (>=3 unique subgroups): contested {len(con)}  not-contested {n_notcon}")
    print(f"among contested ({len(con)}):")
    print(f"  mean subgroups/axis   {np.mean([r['n_sub'] for r in con]):.2f}")
    print(f"  mean pole TV          {np.mean([r['pole_tv'] for r in con]):.3f}")
    print(f"  mean mid_frac (non-pole subgroups between the poles)  {mf.mean():.3f}")
    print(f"  mean off-axis residual (0 = on the pole-pole line)    "
          f"{np.mean([r['mean_off'] for r in con]):.3f}")
    print(f"  >=50% of non-poles in middle:  {(mf >= 0.5).mean():.1%}")
    print(f"  bimodal (0 in middle):         {(mf == 0).mean():.1%}")
    print(f"  mid_frac  p10 {np.percentile(mf, 10):.2f}  median "
          f"{np.median(mf):.2f}  p90 {np.percentile(mf, 90):.2f}")
    verdict = "REAL" if mf.mean() >= 0.5 and (mf == 0).mean() < 0.25 else "WEAK/BIMODAL"
    print(f"\nverdict: the middle is {verdict} "
          f"(inject the full subgroup spectrum: {verdict == 'REAL'})")


if __name__ == "__main__":
    main()
