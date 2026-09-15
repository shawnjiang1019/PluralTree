"""Score vital_responses.jsonl with coverage_reward. No judge exists for VITAL.

VITAL supplies no participants and no human ratings, so judge_overtonbench.py's
predict-a-1-5-rating machinery does not apply. This reuses coverage_reward
directly: each situation's {text, label} values become Position objects
(uniform prevalence -- VITAL carries none), and coverage is the same
recall*sqrt(precision) computation used for the GRPO reward.

THIS IS AN UNVALIDATED SCORER. There are no human ratings to check it against,
same weakness as coverage_reward everywhere else it is used. Valid for
ARM-VS-ARM comparisons (same generator, same length distribution) where a
consistent bias cancels; NOT valid for claiming an absolute coverage number, and
NOT valid for baseline vs an injected condition without a length control.

    python scripts/analysis/score_vital.py \\
        --situations vital_overton_valuekaleidoscope.json \\
        --responses vital_responses.jsonl --out vital_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, __import__("os").path.dirname(
    __import__("os").path.dirname(__import__("os").path.dirname(
        __import__("os").path.abspath(__file__)))))


def _bootstrap(deltas: list[float], n: int = 10000, seed: int = 0):
    if not deltas:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    idx = range(len(deltas))
    boots = sorted(st.mean(deltas[rng.choice(idx)] for _ in idx) for _ in range(n))
    lo, hi = boots[int(0.025 * n)], boots[int(0.975 * n)]
    n_le0 = sum(1 for b in boots if b <= 0)
    p = 2 * min(n_le0, n - n_le0) / n
    return st.mean(deltas), lo, hi, p


def main():
    ap = argparse.ArgumentParser(description="Score VITAL responses")
    ap.add_argument("--situations", required=True)
    ap.add_argument("--responses", required=True,
                    help="comma list; e.g. both geometry arms' files, whose "
                         "conditions eval_vital --tag made distinct")
    ap.add_argument("--min_values", type=int, default=2)
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--baseline", default=None,
                    help="if set, also print paired deltas vs this condition "
                         "(WARNING: length-bias risk, see module docstring)")
    ap.add_argument("--out", default="vital_scores.csv")
    args = ap.parse_args()

    from alignment.reward import Position, RewardConfig, coverage_reward, default_embed_fn
    from data.loaders.valueprism import load_situations

    situations = {s["situation_id"]: s
                 for s in load_situations(args.situations, min_values=args.min_values)}
    print(f"  {len(situations)} situations")

    rows = []
    for path in [p.strip() for p in args.responses.split(",") if p.strip()]:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    print(f"  {len(rows)} response rows")

    embed_fn = default_embed_fn(args.embedder)
    cfg = RewardConfig()

    per_row = []
    skipped = 0
    for r in rows:
        sit = situations.get(r["question_id"])
        if sit is None:
            skipped += 1
            continue
        positions = [Position(option=v["label"], embed_text=v["text"], prevalence=1.0)
                    for v in sit["values"]]
        score, breakdown = coverage_reward(r["response"], positions, embed_fn, cfg)
        per_row.append({"question_id": r["question_id"], "condition": r["condition"],
                        "rollout": r.get("rollout", 0), "coverage": score,
                        "n_words": len((r["response"] or "").split()),
                        "n_values": len(positions),
                        "recall": breakdown.get("recall"),
                        "precision": breakdown.get("precision")})
    if skipped:
        print(f"  {skipped} rows had no matching situation (id mismatch?)")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question_id", "condition", "rollout",
                                          "coverage", "n_words", "n_values",
                                          "recall", "precision"])
        w.writeheader()
        w.writerows(per_row)
    print(f"  wrote {args.out}")

    by_cond = defaultdict(list)
    words = defaultdict(list)
    by_q = defaultdict(dict)
    for r in per_row:
        by_cond[r["condition"]].append(r["coverage"])
        words[r["condition"]].append(r["n_words"])
        by_q[r["question_id"]].setdefault(r["condition"], []).append(r["coverage"])

    # Length next to every score: coverage is capped at floor(words/60)/K values
    # (docs/vital_task.tex, Proposition 1), so a delta between arms of different
    # length is partly a length difference.
    print("\nby condition (mean over rows):")
    for c in sorted(by_cond):
        print(f"  {c:<18}{st.mean(by_cond[c]):.4f}   words={st.mean(words[c]):.0f}"
              f"   (n={len(by_cond[c])})")

    conds = sorted(by_cond)
    print("\npaired deltas (arm - arm, bootstrap over questions):")
    for i, c1 in enumerate(conds):
        for c2 in conds[i + 1:]:
            if args.baseline and args.baseline not in (c1, c2):
                continue
            deltas = []
            for qid, cm in by_q.items():
                if c1 in cm and c2 in cm:
                    deltas.append(st.mean(cm[c1]) - st.mean(cm[c2]))
            if not deltas:
                continue
            m, lo, hi, p = _bootstrap(deltas)
            flag = "  [LENGTH-BIAS RISK]" if args.baseline in (c1, c2) else ""
            dw = st.mean(words[c1]) - st.mean(words[c2])
            print(f"  {c1} - {c2}: {m:+.4f} [{lo:+.4f},{hi:+.4f}] "
                  f"p={p:.4f} n={len(deltas)}  words {dw:+.0f}{flag}")


if __name__ == "__main__":
    main()
