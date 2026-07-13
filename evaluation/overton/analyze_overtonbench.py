"""Post-hoc analysis of an OvertonBench run: paired stats + reasoning-trace table.

Consumes the two artifacts of jobs/eval/job_overton_eval.sh:
  responses JSONL (eval_overtonbench.py — with fork_context/think/n_forks traces)
  scores CSV     (judge_overtonbench.py --score: condition,question_id,coverage)

Reports, per condition vs baseline:
  1. OvertonScore (mean coverage) by condition.
  2. Paired per-question deltas: wins/losses/ties + two-sided sign test.
  3. Trace health: fork fallbacks, missing <answer> tags, retrieved-relevance stats.
  4. Triage retention: did the model KEEP the injected forks or discard them in
     <think>? Cross-tabbed against win/loss — the mechanism table that says
     whether failures are retrieval quality (discarded) or prompting (kept, no win).

Stdlib only — safe for a CPU-only dependency job.

Usage:
    python -m evaluation.overton.analyze_overtonbench \
        overton_responses_v4.jsonl overton_scores_v4.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
from collections import defaultdict

_DISCARD_RE = re.compile(
    r"irrelevan|not (directly )?(relevant|related)|discard|none of (the|these)|"
    r"do(es)? not (directly )?(bear|address|apply|relate)", re.IGNORECASE)
_REL_RE = re.compile(r"relevance=([0-9.]+)")


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided sign-test p under Binom(wins+losses, 0.5)."""
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def load_scores(path: str) -> dict[tuple[str, int], float]:
    cov: dict[tuple[str, int], float] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cov[(row["condition"], int(row["question_id"]))] = float(row["coverage"])
    return cov


def load_traces(path: str) -> dict[tuple[str, int], dict]:
    """Per (condition, qid): fork/triage/tag facts derived from the saved traces."""
    out: dict[tuple[str, int], dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["condition"], int(r["question_id"]))
            think = r.get("think", "") or ""
            fork_ctx = r.get("fork_context", "") or ""
            raw = r.get("raw", "") or ""
            n_forks = int(r.get("n_forks", 0))
            rels = [float(x) for x in _REL_RE.findall(fork_ctx)]
            out[key] = {
                "n_forks": n_forks,
                "tagged": "<answer>" in raw.lower() if n_forks > 0 else True,
                "discarded": bool(_DISCARD_RE.search(think)) if n_forks > 0 else None,
                "rel_max": max(rels) if rels else float("nan"),
                "rel_mean": st.mean(rels) if rels else float("nan"),
            }
    return out


def main():
    ap = argparse.ArgumentParser(description="OvertonBench run analysis")
    ap.add_argument("responses", help="JSONL from eval_overtonbench.py")
    ap.add_argument("scores", help="CSV from judge_overtonbench.py --score")
    ap.add_argument("--baseline", default="baseline")
    args = ap.parse_args()

    cov = load_scores(args.scores)
    traces = load_traces(args.responses)
    conds = sorted({c for c, _ in cov})
    qids_of = {c: sorted(q for cc, q in cov if cc == c) for c in conds}

    # ---- 1. OvertonScore by condition ----
    print("=== OvertonScore (mean coverage over questions) ===")
    for c in conds:
        vals = [cov[(c, q)] for q in qids_of[c]]
        print(f"  {c:<10} {st.mean(vals):.4f}   (n={len(vals)})")

    # ---- 2. paired vs baseline ----
    base = args.baseline
    if base in conds:
        print(f"\n=== Paired per-question deltas vs {base} (sign test) ===")
        for c in conds:
            if c == base:
                continue
            shared = [q for q in qids_of[c] if (base, q) in cov]
            deltas = [cov[(c, q)] - cov[(base, q)] for q in shared]
            w = sum(d > 1e-9 for d in deltas)
            l = sum(d < -1e-9 for d in deltas)
            t = len(deltas) - w - l
            print(f"  {c:<10} mean_delta={st.mean(deltas):+.4f}  "
                  f"w/l/t={w}/{l}/{t}  p={sign_test_p(w, l):.4f}")

    # ---- 3. trace health per condition ----
    print("\n=== Trace health (injected conditions) ===")
    for c in conds:
        rows = [traces[(c, q)] for q in qids_of[c] if (c, q) in traces]
        if not rows:
            continue
        inj = [r for r in rows if r["n_forks"] > 0]
        fallbacks = sum(r["n_forks"] == 0 for r in rows)
        if not inj:
            print(f"  {c:<10} no injected rows (baseline or all fallbacks="
                  f"{fallbacks})")
            continue
        untagged = sum(not r["tagged"] for r in inj)
        rels = [r["rel_max"] for r in inj if r["rel_max"] == r["rel_max"]]
        print(f"  {c:<10} fallbacks={fallbacks}  missing_answer_tags={untagged}  "
              f"rel_max: mean={st.mean(rels):.3f} min={min(rels):.3f} "
              f"max={max(rels):.3f}" if rels else f"  {c:<10} no relevance data")

    # ---- 4. triage retention x win/loss (the mechanism table) ----
    if base in conds:
        print("\n=== Triage retention x outcome vs baseline ===")
        print("  (kept = <think> did NOT discard the forks; win = coverage > baseline)")
        for c in conds:
            if c == base:
                continue
            cell = defaultdict(int)
            for q in qids_of[c]:
                tr = traces.get((c, q))
                if tr is None or tr["discarded"] is None or (base, q) not in cov:
                    continue
                kept = "kept" if not tr["discarded"] else "discarded"
                d = cov[(c, q)] - cov[(base, q)]
                outcome = "win" if d > 1e-9 else ("loss" if d < -1e-9 else "tie")
                cell[(kept, outcome)] += 1
            n_kept = cell[("kept", "win")] + cell[("kept", "loss")] + cell[("kept", "tie")]
            n_disc = cell[("discarded", "win")] + cell[("discarded", "loss")] + cell[("discarded", "tie")]
            total = n_kept + n_disc
            if total == 0:
                print(f"  [{c}] no triage data")
                continue
            print(f"  [{c}] forks kept {n_kept}/{total} ({100 * n_kept / max(total, 1):.0f}%)")
            print(f"      {'':<12}{'win':>6}{'loss':>6}{'tie':>6}")
            for kept in ("kept", "discarded"):
                print(f"      {kept:<12}"
                      f"{cell[(kept, 'win')]:>6}{cell[(kept, 'loss')]:>6}"
                      f"{cell[(kept, 'tie')]:>6}")
        print("\n  reading: kept+win = injection helped | kept+loss = prompt/verbalization "
              "problem\n           discarded+loss = retrieval quality problem | "
              "discarded+tie = fail-soft OK")


if __name__ == "__main__":
    main()
