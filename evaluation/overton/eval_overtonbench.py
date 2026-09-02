"""Generate OvertonBench answers under baseline / scout / div_only conditions.

Pulls the 60 unique questions from HF ``elinorpd/overtonbench``, generates one
answer per (question, condition) with retrieval/answer.py, and writes a JSONL
of ``{question_id, question, condition, response, raw, think, fork_context,
n_forks}`` — 'response' is what the judge scores; the remaining fields are the
complete reasoning trace (what the scout injected, how the model triaged it in
<think>, and the untruncated generation) for post-hoc analysis. Judged by
evaluation/judge_overtonbench.py. See docs/overtonbench_eval.txt.

Usage (needs a vLLM/OpenAI-compatible endpoint serving the generator):
    python -m evaluation.overton.eval_overtonbench \
        --embeddings embeddings_goqa.pt --curvature 0.5 --text_feat feats_goqa.pt \
        --base_url http://localhost:8000/v1 --model Qwen/Qwen2.5-72B-Instruct \
        --conditions baseline,scout,div_only --out overton_responses.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_questions(split: str = "full") -> list[tuple[int, str]]:
    """Unique (question_id, question) pairs, sorted by id."""
    from datasets import load_dataset

    ds = load_dataset("elinorpd/overtonbench", split=split)
    seen: dict[int, str] = {}
    for row in ds:
        seen.setdefault(int(row["question_id"]), row["question"])
    return sorted(seen.items())


def main():
    ap = argparse.ArgumentParser(description="OvertonBench answer generation")
    ap.add_argument("--embeddings", required=True, help=".pt of h_all on the ball")
    ap.add_argument("--dataset", choices=["globalopinionqa", "opinionqa"],
                    default="globalopinionqa")
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text_feat", default=None)
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", default="baseline,scout,div_only")
    ap.add_argument("--tau", type=float, default=None,
                    help="override the scout condition's relevance gate "
                         "(GOQA cross-domain rel ceiling is ~0.14 — use ~0.1)")
    ap.add_argument("--split", default="full")
    ap.add_argument("--max_questions", type=int, default=0, help="0 = all")
    ap.add_argument("--n_rollouts", type=int, default=1,
                    help="samples per (question, condition); >1 enables the "
                         "across-sample coverage@K metric (judge --k_rollouts). "
                         "Generation is already stochastic (temp 0.7).")
    ap.add_argument("--out", default="overton_responses.jsonl")
    ap.add_argument("--dry_run", action="store_true",
                    help="write assembled prompts instead of calling the LLM")
    args = ap.parse_args()

    import torch
    from pluraltree.manifolds.poincare import PoincareBall
    from retrieval.answer import CONDITIONS, answer
    from retrieval.scout import ScoutConfig, embed_question, load_or_compute_text_feat

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        ap.error(f"unknown conditions {unknown}; choose from {sorted(CONDITIONS)}")

    if args.dataset == "opinionqa":
        from data.loaders.opinionqa import load_opinionqa
        graph = load_opinionqa(split_seed=args.seed, leakage_safe=True)
    else:
        from data.loaders.globalopinionqa import load_globalopinionqa
        graph = load_globalopinionqa(split_seed=args.seed, leakage_safe=True)
    h_all = torch.load(args.embeddings, map_location="cpu")
    if not isinstance(h_all, torch.Tensor):
        h_all = h_all["h_all"]
    manifold = PoincareBall(c=args.curvature)
    text_feat = load_or_compute_text_feat(graph, args.dataset, args.text_feat)

    questions = load_questions(args.split)
    if args.max_questions:
        questions = questions[: args.max_questions]
    print(f"{len(questions)} questions x {len(conditions)} conditions -> {args.out}")

    # Resume: skip (question_id, condition, rollout) triples already present.
    # Rows written before --n_rollouts existed have no 'rollout' key -> rollout 0.
    done: set[tuple[int, str, int]] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add((r["question_id"], r["condition"], r.get("rollout", 0)))
        print(f"  resuming: {len(done)} rows already present")

    with open(args.out, "a", encoding="utf-8") as f:
        for qid, question in questions:
            q_emb = embed_question(question)          # shared across conditions
            for cond in conditions:
                cfg = None
                if cond == "scout" and args.tau is not None:
                    cfg = ScoutConfig(tau=args.tau, alpha=CONDITIONS["scout"].alpha)
                for rollout in range(args.n_rollouts):
                    if (qid, cond, rollout) in done:
                        continue
                    resp, trace = answer(question, cond, graph=graph, h_all=h_all,
                                         text_feat=text_feat, manifold=manifold,
                                         base_url=args.base_url, model=args.model,
                                         dry_run=args.dry_run, q_emb=q_emb, cfg=cfg,
                                         with_trace=True)
                    # Complete reasoning record per row — the judge reads only
                    # 'response'; the rest is for observing retrieval -> triage ->
                    # answer: 'fork_context' = what the scout injected, 'think' =
                    # how the model triaged it, 'raw' = the full generation.
                    row = {"question_id": qid, "question": question,
                           "condition": cond, "rollout": rollout,
                           "response": resp, "raw": trace["raw"],
                           "think": trace["think"],
                           "fork_context": trace["fork_context"],
                           "n_forks": trace["n_forks"]}
                    # multi-pass conditions (merge) also carry their drafts, so a
                    # merge that LOSES coverage can be diagnosed against them.
                    # merge_v2 adds merge_fallback/merge_fail/merge_stats: the
                    # rate at which its lossless-merge guard fired is a finding.
                    # n_personas: persona_merge falls back to plain-only when the
                    # anchor has <3 opinion leaves, and such a row is a BASELINE
                    # row under the condition's name. Without it the v11 run could
                    # not say how many rows actually tested the condition.
                    for k in ("draft_a", "draft_b", "merge_fallback",
                              "merge_fail", "merge_stats", "labels",
                              "random_fork", "n_personas"):
                        if k in trace:
                            row[k] = trace[k]
                    # merge_v2_rand skips questions with no comparable unrelated
                    # anchor. Writing an EMPTY response would score 0 and drag the
                    # control arm down on exactly the questions where matching is
                    # hardest; omitting the row drops that question from the
                    # pairing instead, which is what the analysis expects.
                    if trace.get("skipped"):
                        print(f"  Q{qid} [{cond}] SKIPPED "
                              f"({trace['skipped']}) -- row not written")
                        continue
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    tag = f" r{rollout}" if args.n_rollouts > 1 else ""
                    print(f"  Q{qid} [{cond}{tag}] {len(resp)} chars")


if __name__ == "__main__":
    main()
