"""Does the answer deliver what its <think> trace planned?

PLURALISM_INSTRUCTION makes the model list, inside <think>, the distinct positions
it intends to cover, then answer. So the trace is a declared PLAN and the answer
is the EXECUTION, and the gap between them is measurable with no ground truth at
all (direction.txt item 8):

    execution     planned positions expressed in the answer / planned
    dropped       planned and absent
    unplanned     answer positions the trace never named
    ignored_forks injected fork positions named in neither

WHY IT MATTERS TWICE.
  * Diagnosis: `route` produced clean-looking reasoning and answers scoring 0.072.
    Plan and execution came apart and nothing in the current metrics can see it.
  * Reward candidate: the score needs NO cluster targets (which exist only for
    the 60 eval questions) and no judge, so it would apply to every graph
    question. Padding the plan raises the denominator, so the obvious hack is
    self-penalising.

THE NUMBER THAT DECIDES IT is not the execution rate: it is whether execution
ranks same-question answers the way the judge does. That is the same
within-question concordance gate the graph-position reward failed at 0.152
(chance 0.5), so it is reported here in the same form, ties included.

Works on any responses file; rows written after per-draft capture also carry each
draft's own <think>, which is reported separately (the merge's trace is a plan for
combining, not for covering).

    python scripts/analysis/trace_execution.py --responses overton_responses_flat.jsonl \\
        --scores overton_scores_flat.csv --conditions merge_v2,merge_v2_flat
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from alignment.reward import split_units                              # noqa: E402
from scripts.analysis.delta_regressor import _CONTRAST, _HEDGE        # noqa: E402
from scripts.analysis.reward_eval_correlation import _concordance_split  # noqa: E402


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_coverage(path: str) -> dict:
    """(qid, condition) -> judged coverage."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(int(r["question_id"]), r["condition"])] = float(r["coverage"])
    return out


def execution_stats(think: str, answer: str, fork_context: str, embed_fn,
                    theta: float = 0.45, max_units: int = 40) -> dict:
    """Plan/execution counts for ONE row. All matching is cosine over units."""
    plan = split_units(think or "", max_units)
    ans = split_units(answer or "", max_units)
    fork = split_units(fork_context or "", max_units)
    out = {"n_planned": len(plan), "n_answer": len(ans), "n_fork": len(fork),
           "execution": float("nan"), "dropped": float("nan"),
           "unplanned": float("nan"), "ignored_forks": float("nan")}
    if not plan or not ans:
        return out
    texts = plan + ans + fork
    V = np.asarray(embed_fn(texts), dtype=float)
    P, A = V[:len(plan)], V[len(plan):len(plan) + len(ans)]
    F = V[len(plan) + len(ans):]

    delivered = (P @ A.T).max(axis=1) >= theta if A.size else np.zeros(len(plan), bool)
    out["execution"] = float(delivered.mean())
    out["dropped"] = float((~delivered).sum())
    if A.size:
        matched = (A @ P.T).max(axis=1) >= theta
        out["unplanned"] = float((~matched).mean())
    if F.size:
        # A fork position is "ignored" when neither the plan nor the answer
        # touches it -- the retrieval reached the prompt and left no trace.
        seen = np.maximum((F @ P.T).max(axis=1), (F @ A.T).max(axis=1)) >= theta
        out["ignored_forks"] = float((~seen).mean())
    return out


def marker_stats(think: str) -> dict:
    """Contrast/hedge markers per 100 words -- the delta_regressor readout, so a
    signal found here plugs into that feature set rather than a parallel one."""
    w = len((think or "").split())
    return {"t_words": w,
            "t_contrast_rate": 100.0 * len(_CONTRAST.findall(think or "")) / w if w else 0.0,
            "t_hedge_rate": 100.0 * len(_HEDGE.findall(think or "")) / w if w else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description="Plan (<think>) vs execution (answer)")
    ap.add_argument("--responses", required=True)
    ap.add_argument("--scores", default=None,
                    help="judge scores csv: also test whether execution ranks "
                         "answers the way the judge does")
    ap.add_argument("--conditions", default="", help="default: all retrieved arms")
    ap.add_argument("--theta", type=float, default=0.45)
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--stub_embed", action="store_true", help="tests only")
    ap.add_argument("--drafts", action="store_true",
                    help="also score each draft's own <think> (new rows only)")
    ap.add_argument("--out", default="docs/trace_execution.csv")
    args = ap.parse_args()

    rows = load_rows(args.responses)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()] or \
        sorted({r["condition"] for r in rows if (r.get("think") or "").strip()})
    if args.stub_embed:
        from alignment.group_reward import _topic_embed as embed_fn
    else:
        from alignment.reward import default_embed_fn
        embed_fn = default_embed_fn(args.embedder)

    per_row, by_cond = [], defaultdict(list)
    for r in rows:
        cond = r["condition"]
        if cond not in conds:
            continue
        s = execution_stats(r.get("think", ""), r.get("response", ""),
                            r.get("fork_context", ""), embed_fn, args.theta)
        s.update(marker_stats(r.get("think", "")))
        s.update({"question_id": int(r["question_id"]), "condition": cond,
                  "rollout": int(r.get("rollout", 0)), "scope": "merge"})
        per_row.append(s)
        if s["execution"] == s["execution"]:
            by_cond[cond].append(s["execution"])
        if args.drafts:
            for d in (r.get("draft_traces") or []):
                ds = execution_stats(d.get("think", ""), d.get("text", ""),
                                     r.get("fork_context", ""), embed_fn, args.theta)
                ds.update(marker_stats(d.get("think", "")))
                ds.update({"question_id": int(r["question_id"]),
                           "condition": f"{cond}:{d.get('label', '?')}",
                           "rollout": int(r.get("rollout", 0)), "scope": "draft"})
                per_row.append(ds)

    if not per_row:
        print("no rows with a <think> trace in the requested conditions")
        return 1

    print(f"=== plan vs execution (theta={args.theta}) ===")
    print(f"  {'condition':<22}{'rows':>6}{'planned':>9}{'execution':>11}"
          f"{'unplanned':>11}{'ignored_forks':>15}")
    for cond in sorted({s["condition"] for s in per_row}):
        ss = [s for s in per_row if s["condition"] == cond]
        def m(k):
            v = [s[k] for s in ss if s[k] == s[k]]
            return st.mean(v) if v else float("nan")
        print(f"  {cond:<22}{len(ss):>6}{m('n_planned'):>9.1f}{m('execution'):>11.3f}"
              f"{m('unplanned'):>11.3f}{m('ignored_forks'):>15.3f}")
    print("  execution = share of planned positions the answer delivers; "
          "ignored_forks = injected positions neither planned nor answered.")

    if args.scores:
        cov = load_coverage(args.scores)
        pairs, by_q = [], defaultdict(list)
        for s in per_row:
            if s["scope"] != "merge" or s["execution"] != s["execution"]:
                continue
            c = cov.get((s["question_id"], s["condition"]))
            if c is not None:
                by_q[s["question_id"]].append((s["execution"], c))
        for q, vals in by_q.items():
            for a in range(len(vals)):
                for b in range(a + 1, len(vals)):
                    pairs.append((vals[b][1] - vals[a][1], vals[b][0] - vals[a][0]))
        if pairs:
            tie, sep, n_sep = _concordance_split(pairs)
            agree = sum(1 for j, x in pairs
                        if abs(j) > 1e-9 and (j > 0) == (x > 0) and abs(x) > 1e-9)
            used = sum(1 for j, _ in pairs if abs(j) > 1e-9)
            print(f"\n=== does execution rank like the judge? ({used} comparable "
                  f"pairs, chance 0.5) ===")
            print(f"  concordance {agree / used:.3f}   tie_rate {tie:.2f}   "
                  f"concordance|separated {sep:.3f}")
            print("  Above chance means a judge-free, target-free signal tracks "
                  "the metric -- a reward candidate. At chance it is a diagnostic "
                  "only.")
        else:
            print("\n(no within-question pairs: need >1 condition per question "
                  "in --scores)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = ["question_id", "condition", "rollout", "scope", "n_planned", "n_answer",
            "n_fork", "execution", "dropped", "unplanned", "ignored_forks",
            "t_words", "t_contrast_rate", "t_hedge_rate"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{k: s.get(k) for k in cols} for s in per_row])
    print(f"\nwrote {args.out}")
    return 0


def _selftest() -> None:
    from alignment.group_reward import _deep, _topic_embed

    plan = "\n".join([_deep("taxes", 1), _deep("guns", 1), _deep("climate", 1)])
    answer = "\n".join([_deep("taxes"), _deep("guns"), _deep("health")])
    s = execution_stats(plan, answer, _deep("climate"), _topic_embed, theta=0.55)
    assert s["n_planned"] == 3, s
    assert abs(s["execution"] - 2 / 3) < 1e-9, s          # taxes+guns delivered
    assert s["dropped"] == 1.0, s                        # climate planned, dropped
    assert abs(s["unplanned"] - 1 / 3) < 1e-9, s         # health never planned
    assert s["ignored_forks"] == 0.0, s                  # climate WAS planned

    ignored = execution_stats(plan, answer, _deep("health"), _topic_embed, theta=0.55)
    assert ignored["ignored_forks"] == 0.0, "health is in the answer, not ignored"
    unused = execution_stats(_deep("taxes", 1), _deep("taxes"), _deep("guns"),
                             _topic_embed, theta=0.55)
    assert unused["ignored_forks"] == 1.0, unused

    empty = execution_stats("", _deep("taxes"), "", _topic_embed)
    assert empty["execution"] != empty["execution"], "no plan -> undefined, not 0"
    mk = marker_stats("They disagree sharply, however it might be unclear.")
    assert mk["t_contrast_rate"] > 0 and mk["t_hedge_rate"] > 0, mk
    print("trace_execution self-test OK (execution, dropped, unplanned, "
          "ignored_forks, markers)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sys.exit(main())
