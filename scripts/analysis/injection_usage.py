"""How much of the injected block reaches the answer, and where in it.

Reads what is already on disk -- `fork_context` and `response` are stored on
every row -- so this needs no generation and no judge.

THE CONTRAST IS THE POINT. An absolute "62% of positions were used" means little
on its own, because graph positions and judge clusters are probably different
target sets (the reward's within-question concordance sits BELOW chance at every
threshold, which is the leading explanation). What is interpretable is
`merge_v2` against `merge_v2_rand` on the same 60 questions:

  same usage rate       the model ignores fork CONTENT either way, which
                        explains why the two arms score the same (+0.0498 vs
                        +0.0345, paired difference p=0.42)
  real used MORE        the content lands and still does not help -- a different
                        failure, and one that points at the judge rather than at
                        retrieval
  real used LESS        the irrelevant forks are somehow easier to absorb, which
                        would mean the usage measure is tracking fluency

WHAT THIS DOES NOT SHOW. Matching an answer unit to an injected position is
correlational: the model may have produced that sentence anyway. Only ablation
(drop a position, regenerate) is causal. This is descriptive, and cheap.

    python scripts/analysis/injection_usage.py \\
        --responses overton_responses_v11.jsonl \\
        --conditions merge_v2,merge_v2_rand
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Position lines in both renders are indented and carry a "LABEL: text" shape:
#   fork_context       "    A: <option text>"
#   fork_context_full  "    <subgroup>: <option text>"
# The block's header lines ("[fork 1] at ...", "  Perspective A (...):") have no
# text after the colon, which is what separates them.
_POS_LINE = re.compile(r"^\s{2,}(?:[A-Za-z0-9][^:]{0,60}):\s*(\S.*)$")


def parse_positions(ctx: str) -> list[str]:
    """The individual position statements inside a rendered fork block."""
    out = []
    for line in (ctx or "").splitlines():
        m = _POS_LINE.match(line)
        if m:
            s = m.group(1).strip()
            if len(s.split()) >= 3:            # a label fragment is not a position
                out.append(s)
    return out


def analyse(rows, conditions, embed_fn, thresholds):
    from alignment.reward import split_units

    import numpy as np

    out = {}
    for cond in conditions:
        sel = [r for r in rows if r.get("condition") == cond
               and (r.get("fork_context") or "").strip()
               and (r.get("response") or "").strip()]
        if not sel:
            print(f"  [skip] {cond}: no rows with both a fork block and a response")
            continue

        best_cos, first_loc, n_pos, n_units, unmatched = [], [], [], [], 0
        per_row_used = {t: [] for t in thresholds}
        for r in sel:
            positions = parse_positions(r["fork_context"])
            units = split_units(r["response"])
            if not positions or not units:
                continue
            n_pos.append(len(positions))
            n_units.append(len(units))
            emb = embed_fn(positions + units)
            P, U = emb[:len(positions)], emb[len(positions):]
            sim = P @ U.T                       # normalized -> cosine
            for i in range(len(positions)):
                j = int(np.argmax(sim[i]))
                c = float(sim[i, j])
                best_cos.append(c)
                # WHERE in the answer this position landed, 0=first unit, 1=last.
                first_loc.append(j / max(1, len(units) - 1))
            for t in thresholds:
                hit = [i for i in range(len(positions)) if float(sim[i].max()) >= t]
                per_row_used[t].append(len(hit) / len(positions))
                if t == thresholds[0] and not hit:
                    unmatched += 1

        out[cond] = {
            "rows": len(n_pos),
            "positions_per_row": st.mean(n_pos) if n_pos else float("nan"),
            "units_per_row": st.mean(n_units) if n_units else float("nan"),
            "cos_p25": _pct(best_cos, 25), "cos_p50": _pct(best_cos, 50),
            "cos_p75": _pct(best_cos, 75),
            "loc_p50": _pct(first_loc, 50),
            "rows_with_no_match": unmatched,
            **{f"used@{t}": (st.mean(per_row_used[t]) if per_row_used[t]
                             else float("nan")) for t in thresholds},
        }
    return out


def _pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def main():
    ap = argparse.ArgumentParser(description="How much of the injection is used")
    ap.add_argument("--responses", required=True)
    ap.add_argument("--conditions", default="merge_v2,merge_v2_rand")
    ap.add_argument("--thresholds", default="0.35,0.45,0.55")
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--out", default=None, help="csv of the summary rows")
    args = ap.parse_args()

    from alignment.reward import default_embed_fn

    rows = []
    with open(args.responses, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    thrs = [float(t) for t in args.thresholds.split(",")]

    print(f"{len(rows)} rows from {os.path.basename(args.responses)}")
    stats = analyse(rows, conds, default_embed_fn(args.embedder), thrs)
    if not stats:
        ap.error("nothing measurable; check --conditions against the file")

    hdr = (f"  {'condition':<16}{'rows':>6}{'pos/row':>9}{'units':>7}"
           f"{'cos p50':>9}{'loc p50':>9}"
           + "".join(f"{'used@' + str(t):>11}" for t in thrs))
    print("\n=== injected positions expressed in the answer ===")
    print(hdr)
    for cond, s in stats.items():
        print(f"  {cond:<16}{s['rows']:>6}{s['positions_per_row']:>9.1f}"
              f"{s['units_per_row']:>7.1f}{s['cos_p50']:>9.3f}{s['loc_p50']:>9.2f}"
              + "".join(f"{s[f'used@{t}']:>11.3f}" for t in thrs))
    print("  cos p50 = median best-matching answer unit per injected position")
    print("  loc p50 = WHERE that unit sits, 0.0 = first unit, 1.0 = last")
    print("  used@t  = mean fraction of a row's positions with some unit >= t")

    if len(stats) == 2:
        (ca, sa), (cb, sb) = stats.items()
        t = thrs[0]
        print(f"\n=== {ca} vs {cb} ===")
        print(f"  usage@{t}   {sa[f'used@{t}']:.3f}  vs  {sb[f'used@{t}']:.3f}"
              f"   ({sa[f'used@{t}'] - sb[f'used@{t}']:+.3f})")
        print(f"  median cos  {sa['cos_p50']:.3f}  vs  {sb['cos_p50']:.3f}"
              f"   ({sa['cos_p50'] - sb['cos_p50']:+.3f})")
        print("  A small gap means the model absorbs irrelevant forks about as")
        print("  readily as retrieved ones -- which would explain the arms")
        print("  scoring the same, and would locate the problem in RETRIEVAL")
        print("  rather than in how the injection is consumed.")
        print("  NOTE: cosine against an unrelated position is not zero. Read the")
        print("  DIFFERENCE, never the level.")

    if args.out:
        import csv as _csv
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=["condition", *next(iter(stats.values()))])
            w.writeheader()
            for cond, s in stats.items():
                w.writerow({"condition": cond, **s})
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
