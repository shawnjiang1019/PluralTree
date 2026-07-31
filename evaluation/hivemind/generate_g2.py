"""Generate INFINITY-CHAT response pools with G2 decoding (arXiv:2511.00432).

Writes the SAME JSONL schema as evaluation/hivemind/generate_hivemind.py, so the
existing panel (evaluation/hivemind/diversity_metrics.py) scores it unchanged:
    {query_id, category, condition, sample_idx, response, raw}

Why a separate generator: G2 needs three forward passes per token and direct
logit access, which the OpenAI-compatible vLLM endpoint cannot express. This runs
a LOCAL HF model instead (default Qwen2.5-7B-Instruct, ~15GB bf16).

The comparison is exactly controlled: `baseline` runs the identical sampler, the
identical prompt and the identical code path with theta=0 (no contrast), so the
ONLY difference between conditions is G2's contrastive term. That matters here --
our earlier hivemind run measured vendi 1.4/8 and mean_cos 0.92, i.e. near-total
mode collapse, and we want to attribute any movement to the mechanism rather
than to a prompt or sampler change.

    python -m evaluation.hivemind.generate_g2 --model Qwen/Qwen2.5-7B-Instruct \
        --conditions baseline,g2 --num_queries 20 --num_samples 8 \
        --out hivemind_g2.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from retrieval.g2 import G2Config, g2_generate


def main():
    ap = argparse.ArgumentParser(description="G2 pools for the INFINITY-CHAT eval")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="LOCAL HF model (needs logits; not a vLLM endpoint)")
    ap.add_argument("--conditions", default="baseline,g2",
                    help="baseline = same code path with theta=0 (controlled)")
    ap.add_argument("--num_queries", type=int, default=20)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--theta", type=float, default=0.3,
                    help="G2 intervention strength (paper sweeps .15/.3/.5/.7)")
    ap.add_argument("--beta", type=float, default=0.1, help="entropy gate (nats)")
    ap.add_argument("--k_repr", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--hf_name", default="liweijiang/infinite-chats-eval")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_embedder", default="sentence-transformers/all-mpnet-base-v2",
                    help="for Center Selection over prior answers (held-out)")
    ap.add_argument("--out", default="hivemind_g2.jsonl")
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in ("baseline", "g2")]
    if unknown:
        ap.error(f"unknown conditions {unknown}; choose from baseline,g2")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from data.loaders.infinity_chat import load_hivemind_queries

    queries = load_hivemind_queries(args.num_queries, hf_name=args.hf_name,
                                    split=args.split, seed=args.seed)
    print(f"{len(queries)} queries x {len(conditions)} conditions "
          f"x {args.num_samples} samples -> {args.out}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None).eval()
    print(f"loaded {args.model} on {next(model.parameters()).device}")

    embed_fn = None
    if args.k_repr > 0:
        from retrieval.contestedness import default_embed_fn
        embed_fn = default_embed_fn(args.eval_embedder)

    # Resume: count samples already present per (query_id, condition).
    have: dict[tuple[int, str], int] = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                k = (r["query_id"], r["condition"])
                have[k] = have.get(k, 0) + 1
        print(f"  resuming: {sum(have.values())} samples already present")

    base_cfg = dict(beta=args.beta, k_repr=args.k_repr,
                    temperature=args.temperature, top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens)

    with open(args.out, "a", encoding="utf-8") as f:
        for qid, question, category in queries:
            for cond in conditions:
                # theta=0 makes the g2 path reduce EXACTLY to base sampling, so
                # baseline differs from g2 only by the contrastive term.
                cfg = G2Config(theta=(args.theta if cond == "g2" else 0.0),
                               **base_cfg)
                start = have.get((qid, cond), 0)
                # prior answers must be rebuilt for a resumed pool: G2 conditions
                # answer i on answers < i, so a partial pool is not restartable
                # from scratch without them.
                pool: list[str] = []
                if start:
                    with open(args.out, encoding="utf-8") as g:
                        for line in g:
                            r = json.loads(line)
                            if r["query_id"] == qid and r["condition"] == cond:
                                pool.append(r["response"])
                    pool = pool[:start]
                for i in range(start, args.num_samples):
                    resp = g2_generate(model, tok, question, pool, cfg, embed_fn)
                    pool.append(resp)
                    f.write(json.dumps({
                        "query_id": qid, "category": category, "condition": cond,
                        "sample_idx": i, "response": resp, "raw": resp}) + "\n")
                    f.flush()
                print(f"  Q{qid} [{cond}] {args.num_samples} samples")

    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
