"""Sample N responses per INFINITY-CHAT query under baseline / scout / div_only.

Reproduces the Artificial Hivemind repetition setup (Jiang et al., NeurIPS 2025,
Fig 4): draw many samples per open-ended query at top-p=0.9 / T=1.0, so
evaluation/hivemind/diversity_metrics.py can measure intra-pool self-similarity.
Here we additionally compare an unconditioned generator (baseline) against
scout-injected divergence forks — does retrieved divergence *lower* mode collapse?

The scout is deterministic per query, so its forks are computed once and the
prompt is reused across all N samples (only the sampler varies). baseline needs
no graph/embeddings at all — a pure, graph-free mode-collapse measurement.

Usage (needs a vLLM/OpenAI-compatible endpoint):
    # graph-free baseline only:
    python -m evaluation.hivemind.generate_hivemind --conditions baseline \
        --model Qwen/... --base_url http://localhost:8000/v1 \
        --num_samples 50 --out hivemind_gen.jsonl
    # add injected conditions (needs embeddings + text_feat):
    python -m evaluation.hivemind.generate_hivemind \
        --conditions baseline,scout,div_only \
        --embeddings embeddings_opinionqa.pt --text_feat feats_opinionqa.pt \
        --dataset opinionqa --model Qwen/... --num_samples 50 --out hivemind_gen.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser(description="INFINITY-CHAT diversity generation")
    ap.add_argument("--conditions", default="baseline")
    ap.add_argument("--num_samples", type=int, default=50,
                    help="responses per (query, condition) — paper uses 50")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=1024,
                    help="baseline cap; injected conditions force 4096 (think+answer)")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    # query source
    ap.add_argument("--hf_name", default="liweijiang/infinite-chats-eval",
                    help="INFINITY-CHAT100 queries. NOTE: "
                         "liweijiang/artificial-hivemind is a HF *collection*, "
                         "not a loadable dataset — see data/loaders/infinity_chat.py")
    ap.add_argument("--split", default="train")
    ap.add_argument("--config", default=None)
    ap.add_argument("--num_queries", type=int, default=100)
    ap.add_argument("--category", default=None, help="restrict to one taxonomy label")
    ap.add_argument("--no_subset100", action="store_true",
                    help="sample from the full 26K instead of INFINITY-CHAT100")
    # scout / graph (only needed for non-baseline conditions)
    ap.add_argument("--embeddings", default=None, help=".pt of h_all on the ball")
    ap.add_argument("--dataset", choices=["globalopinionqa", "opinionqa"],
                    default="opinionqa")
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text_feat", default=None)
    ap.add_argument("--tau", type=float, default=None,
                    help="override the scout condition's relevance gate")
    ap.add_argument("--out", default="hivemind_gen.jsonl")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    from data.loaders.infinity_chat import load_hivemind_queries
    from retrieval.answer import CONDITIONS, chat, extract_answer

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        ap.error(f"unknown conditions {unknown}; choose from {sorted(CONDITIONS)}")
    injected = [c for c in conditions if CONDITIONS[c] is not None]

    graph = h_all = text_feat = manifold = None
    if injected:
        if not args.embeddings:
            ap.error(f"conditions {injected} need --embeddings (+ --text_feat)")
        import torch
        from pluraltree.manifolds.poincare import PoincareBall
        from retrieval.scout import load_or_compute_text_feat

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

    queries = load_hivemind_queries(
        args.num_queries, hf_name=args.hf_name, split=args.split,
        config=args.config, seed=args.seed, category=args.category,
        subset100=not args.no_subset100)
    print(f"{len(queries)} queries x {len(conditions)} conditions "
          f"x {args.num_samples} samples -> {args.out}")

    # Resume: count samples already present per (query_id, condition).
    have: dict[tuple[int, str], int] = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                k = (r["query_id"], r["condition"])
                have[k] = have.get(k, 0) + 1
        print(f"  resuming: {sum(have.values())} samples already present")

    with open(args.out, "a", encoding="utf-8") as f:
        for qid, question, category in queries:
            # Build each condition's prompt once (scout is deterministic).
            prompts = _build_prompts(question, conditions, graph, h_all,
                                     text_feat, manifold, args)
            for cond in conditions:
                messages, is_inj = prompts[cond]
                start = have.get((qid, cond), 0)
                if args.dry_run:
                    if start == 0:
                        f.write(json.dumps({
                            "query_id": qid, "category": category,
                            "condition": cond, "sample_idx": -1,
                            "response": "\n\n".join(
                                f"<{m['role']}>\n{m['content']}" for m in messages),
                            "raw": ""}) + "\n")
                    continue
                cap = 4096 if is_inj else args.max_tokens
                for i in range(start, args.num_samples):
                    raw = chat(args.base_url, args.model, messages,
                               temperature=args.temperature, top_p=args.top_p,
                               max_tokens=cap)
                    resp = extract_answer(raw)[0] if is_inj else raw
                    f.write(json.dumps({
                        "query_id": qid, "category": category, "condition": cond,
                        "sample_idx": i, "response": resp, "raw": raw}) + "\n")
                    f.flush()
                print(f"  Q{qid} [{cond}] {args.num_samples} samples")


def _build_prompts(question, conditions, graph, h_all, text_feat, manifold, args):
    """{condition: (messages, is_injected)} — forks computed once per query."""
    from retrieval.answer import CONDITIONS, build_prompt
    from retrieval.scout import ScoutConfig, scout

    out: dict[str, tuple[list[dict], bool]] = {}
    for cond in conditions:
        base = CONDITIONS[cond]
        if base is None:                             # baseline: no retrieval
            out[cond] = (build_prompt(question, None, graph), False)
            continue
        cfg = base
        if cond == "scout" and args.tau is not None:
            cfg = ScoutConfig(tau=args.tau, alpha=base.alpha)
        forks = scout(question, graph, h_all, text_feat, manifold, cfg=cfg)
        if not forks:
            print(f"  warning: 0 forks (tau={cfg.tau}) — baseline prompt for "
                  f"[{cond}]: {question[:60]}", file=sys.stderr)
        out[cond] = (build_prompt(question, forks, graph), bool(forks))
    return out


if __name__ == "__main__":
    main()
