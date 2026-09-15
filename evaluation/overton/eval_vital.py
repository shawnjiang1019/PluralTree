"""Generate VITAL-overton answers under any condition. Mirrors eval_overtonbench.py.

VITAL has no participants, no clusters, no human ratings -- judge_overtonbench.py
cannot score it. This exists only to GENERATE; scripts/analysis/score_vital.py
scores the output with coverage_reward against each situation's values.

SCOPE: arm-vs-arm only (geometry c0.5/c0, merge_v2/divrand/flat). Do not compare
baseline vs an injected condition here -- neither VITAL's own NLI scorer (which
the VITAL authors document as length-biased) nor coverage_reward has a measured
length bias, and injected answers run ~5x longer than baseline's ~69 words.

Usage:
    python -m evaluation.overton.eval_vital \\
        --situations vital_overton_valuekaleidoscope.json \\
        --embeddings embeddings_opinionqa.pt --dataset opinionqa \\
        --base_url http://localhost:8000/v1 --model Qwen/Qwen2.5-72B-Instruct-AWQ \\
        --conditions baseline,merge_v2 --out vital_responses.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.loaders.graphs import DATASETS, load_graph


def main():
    ap = argparse.ArgumentParser(description="VITAL-overton answer generation")
    ap.add_argument("--situations", required=True,
                    help="vital_overton_valuekaleidoscope.json (NOT the "
                         "steerable_opinionqa or distributional_globalopinionqa "
                         "files -- those ARE this project's graph source)")
    ap.add_argument("--min_values", type=int, default=2)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--dataset", choices=list(DATASETS), default="opinionqa")
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42,
                    help="graph split seed; MUST match train.py --seed (42) or node "
                         "ids and embedding rows disagree")
    ap.add_argument("--text_feat", default=None)
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", default="baseline,merge_v2")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--max_questions", type=int, default=0)
    ap.add_argument("--n_rollouts", type=int, default=1)
    ap.add_argument("--seed_situations", type=int, default=0,
                    help="shuffle before truncating -- id 0 is the trolley "
                         "problem, the file is not representative in order")
    ap.add_argument("--tag", default="",
                    help="suffix written into the condition name (merge_v2@c0). "
                         "Two geometry arms run the SAME condition on different "
                         "embeddings; without a tag score_vital cannot tell "
                         "them apart when their files are scored together.")
    ap.add_argument("--out", default="vital_responses.jsonl")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    import random

    import torch
    from pluraltree.manifolds.poincare import PoincareBall
    from retrieval.answer import CONDITIONS, answer
    from retrieval.scout import ScoutConfig, embed_question, load_or_compute_text_feat

    from data.loaders.valueprism import DEFAULT_TEMPLATE, load_situations

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        ap.error(f"unknown conditions {unknown}; choose from {sorted(CONDITIONS)}")
    if "baseline" in conditions and len(conditions) > 1:
        print("WARNING: comparing baseline against an injected condition here "
              "crosses an unmeasured length-bias risk in coverage_reward. See "
              "the module docstring. Proceed only for a length-CONTROLLED run.",
              file=sys.stderr)

    graph = load_graph(args.dataset, split_seed=args.seed, leakage_safe=True)
    h_all = torch.load(args.embeddings, map_location="cpu")
    if not isinstance(h_all, torch.Tensor):
        h_all = h_all["h_all"]
    manifold = PoincareBall(c=args.curvature)
    text_feat = load_or_compute_text_feat(graph, args.dataset, args.text_feat)

    situations = load_situations(args.situations, min_values=args.min_values)
    if args.max_questions:
        rng = random.Random(args.seed_situations)
        situations = sorted(situations, key=lambda s: s["situation_id"])
        rng.shuffle(situations)
        situations = situations[: args.max_questions]
    # Same string the gate measured (valueprism.questions_only): VITAL's shipped
    # `input` prompt, else DEFAULT_TEMPLATE. The bare situation is a 7-word action
    # phrase -- retrieving and answering on it measures a different input than
    # the gate did.
    questions = [(s["situation_id"],
                  (s.get("prompt") or "").strip()
                  or DEFAULT_TEMPLATE.format(situation=s["situation"]).strip())
                 for s in situations]
    print(f"{len(questions)} situations x {len(conditions)} conditions -> {args.out}")

    done: set[tuple[str, str, int]] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add((r["question_id"], r["condition"], r.get("rollout", 0)))
        print(f"  resuming: {len(done)} rows already present")

    with open(args.out, "a", encoding="utf-8") as f:
        for qid, question in questions:
            q_emb = embed_question(question)
            for cond in conditions:
                name = f"{cond}@{args.tag}" if args.tag else cond
                cfg = None
                if cond == "scout" and args.tau is not None:
                    cfg = ScoutConfig(tau=args.tau, alpha=CONDITIONS["scout"].alpha)
                for rollout in range(args.n_rollouts):
                    if (qid, name, rollout) in done:
                        continue
                    resp, trace = answer(question, cond, graph=graph, h_all=h_all,
                                         text_feat=text_feat, manifold=manifold,
                                         base_url=args.base_url, model=args.model,
                                         cfg=cfg, q_emb=q_emb, dry_run=args.dry_run,
                                         with_trace=True)
                    # think/fork_context/drafts ride along for the trace
                    # analyses (scripts/analysis/trace_execution.py); VITAL has
                    # no judge, so these are the only window into WHY an arm
                    # scored what it did.
                    row = {"question_id": qid, "question": question,
                           "condition": name, "rollout": rollout,
                           "response": resp, "n_forks": trace.get("n_forks", 0),
                           "think": trace.get("think", ""),
                           "fork_context": trace.get("fork_context", "")}
                    if "draft_traces" in trace:
                        row["draft_traces"] = trace["draft_traces"]
                    f.write(json.dumps(row) + "\n")
                    f.flush()
            print(f"  {qid}  {len(conditions)} conditions x {args.n_rollouts} rollouts")

    print(f"\nDone -> {args.out}")
    print(f"Score with: python scripts/analysis/score_vital.py --situations "
          f"{args.situations} --responses {args.out}")


if __name__ == "__main__":
    main()
