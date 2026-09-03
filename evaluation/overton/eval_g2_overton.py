"""OvertonBench generation under G2 decoding, incl. the graph-guided variant.

    logits = z + alpha_t * (z+ - z-),   alpha_t = theta if H(softmax(z)) >= beta

WHY. Prompt injection over-anchors: `ctx0` 0.0992 vs `base7b` 0.3942 at 7B, and
coverage is monotone in the CAD alpha across five points. The forks are read
(injection_usage: ~61% of positions surface) and reading them is what breaks the
answer. G2 moves the graph OUT of the answering context entirely -- the base
stream `z` sees only the question, and the forks condition the two GUIDE streams,
which contribute a direction rather than text. There is nothing to anchor on.

The entropy gate is the other half: alpha is 0 wherever the base model is already
confident, so the intervention cannot reorganise the parts of the answer that
were fine. Prompt injection has no such gate, which is the `ctx0` collapse.

WHAT IS NEW HERE. G2 was replicated on INFINITY-CHAT (job_g2_diversity.sh) and
won on the diversity panel, but that is a different benchmark and a different
metric, and the GRAPH-GUIDED variant has never run: generate_g2.py hard-rejects
anything outside (baseline, g2). docs/hivemind_diversity_eval.txt Sec.7 records
why -- an opinion-domain graph yields off-topic forks on creative queries, so
there was nothing to aim at. OvertonBench is in-domain for the ATP graph, which
is exactly the condition that was missing.

ARMS. All three run through `g2_generate_ids`, so sampling, temperature, top-p
and RNG consumption are identical and only the decoding rule differs:

    g2_base    theta=0     N independent samples          the pool reference
    g2         theta>0     generic diversity guide        the paper's method
    g2_graph   theta>0     guide names a graph position   ours

BASE STREAM = `base7b`. The base messages are built with BASELINE_INSTRUCTION
through `build_prompt`, the same call eval_cad.py makes, so `g2_base` rollout 0
is that arm modulo seed. Using G2_BASE instead would make every comparison to
the CAD table cross a prompt change as well as a decoding change.

JUDGING. Pool members are written as `rollout` 0..N-1, so judge_overtonbench
reports both within-answer `coverage` and across-pool `coverage@K` with no
changes. coverage@K is the quantity G2 targets -- it is a pool method, and its
value is the union over samples. Compare arms to `g2_base` ONLY: this is a 7B,
and v10/v11/v12 are Qwen-72B-AWQ.

    python -m evaluation.overton.eval_g2_overton \\
        --embeddings embeddings_opinionqa.pt --dataset opinionqa \\
        --model <local 7B dir> --arms g2_base,g2,g2_graph \\
        --n_samples 4 --half a --out g2_responses_a.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.overton.eval_overtonbench import load_questions

ARMS = ("g2_base", "g2", "g2_graph")


def target_positions(forks, graph, n: int, min_prevalence: float = 0.05
                     ) -> list[str]:
    """Distinct graph positions across the forks' anchors, most prevalent first.

    Answer i is aimed at target i, so the ordering decides what gets covered
    early. Descending prevalence puts the common positions first -- which is what
    the judge's clusters look like, since cluster size tracks how many
    participants held the view -- and leaves the minority tail to the later
    answers, the ones that have priors to diverge from and so get the strongest
    contrast. Ascending would aim the weakest contrast at the rarest position.
    """
    from alignment.reward import positions_from_subtree

    seen: set[str] = set()
    out = []
    for f in forks or []:
        for p in positions_from_subtree(graph, f.anchor, min_prevalence):
            key = p.option.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(p)
    out.sort(key=lambda p: -p.prevalence)
    return [p.embed_text for p in out[:n]]


def main():
    ap = argparse.ArgumentParser(description="OvertonBench under G2 decoding")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--model", required=True, help="LOCAL HF dir (needs logits)")
    ap.add_argument("--dataset", choices=["globalopinionqa", "opinionqa"],
                    default="opinionqa")
    ap.add_argument("--text_feat", default=None)
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="full")
    ap.add_argument("--max_questions", type=int, default=0)
    ap.add_argument("--arms", default="g2_base,g2,g2_graph")
    ap.add_argument("--half", choices=["all", "a", "b"], default="a",
                    help="tune theta on 'a', report on 'b' -- picking the best "
                         "theta over all 60 is the mistake bestofk flagged")
    ap.add_argument("--n_samples", type=int, default=4,
                    help="pool size per (question, arm); judge --k_rollouts this")
    ap.add_argument("--theta", type=float, default=0.3,
                    help="paper sweeps .15/.3/.5/.7")
    ap.add_argument("--beta", type=float, default=0.1, help="entropy gate, nats")
    ap.add_argument("--k_repr", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="g2_responses.jsonl")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pluraltree.manifolds.poincare import PoincareBall
    from retrieval.answer import (BASELINE_INSTRUCTION, build_prompt,
                                  extract_answer, forks_to_context)
    from retrieval.g2 import (G2Config, g2_generate_ids, g2_messages,
                              graph_guided_messages, select_representatives)
    from retrieval.scout import (ScoutConfig, embed_question,
                                 load_or_compute_text_feat, scout)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        ap.error(f"unknown arms {unknown}; choose from {list(ARMS)}")

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
    cfg_scout = ScoutConfig(tau=args.tau, alpha=1.0)

    print(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype)).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # Held-out embedder, used ONLY to pick which priors the guides see (Center
    # Selection). Not the scout's MiniLM, and it never touches scoring.
    from alignment.reward import default_embed_fn
    embed_fn = default_embed_fn()

    questions = load_questions(args.split)
    if args.half != "all":                          # deterministic, id-based
        want = 0 if args.half == "a" else 1
        questions = [(q, t) for q, t in questions if q % 2 == want]
    if args.max_questions:
        questions = questions[: args.max_questions]

    # Resume needs the TEXT, not just the count: answer i is conditioned on
    # answers < i, so a resumed pool must rebuild its own priors or the
    # contrast is computed against nothing and the arm silently degrades.
    pools: dict[tuple[int, str], list[str]] = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    pools.setdefault((r["question_id"], r["condition"]),
                                     []).append(r["response"])
        print(f"resuming: {sum(len(v) for v in pools.values())} rows present")

    def ids_for(msgs):
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt")
        return enc["input_ids"] if hasattr(enc, "input_ids") else enc

    print(f"{len(questions)} questions x {len(arms)} arms x {args.n_samples} "
          f"samples -> {args.out}")
    with open(args.out, "a", encoding="utf-8") as out:
        for qid, question in questions:
            q_emb = embed_question(question)
            forks = scout(question, graph, h_all, text_feat, manifold,
                          cfg=cfg_scout, q_emb=q_emb)
            ctx_text = forks_to_context(forks, graph, False) if forks else ""
            targets = target_positions(forks, graph, args.n_samples)
            # The base stream is `base7b`: BASELINE_INSTRUCTION, no forks. The
            # graph enters only through the guides.
            ids_base = ids_for(build_prompt(question, None, graph,
                                            BASELINE_INSTRUCTION, False))

            for arm in arms:
                if arm == "g2_graph" and not targets:
                    print(f"  Q{qid} [{arm}] no graph positions, skipped")
                    continue
                pool = pools.setdefault((qid, arm), [])
                theta = 0.0 if arm == "g2_base" else args.theta
                cfg = G2Config(theta=theta, beta=args.beta, k_repr=args.k_repr,
                               temperature=args.temperature, top_p=args.top_p,
                               max_new_tokens=args.max_new_tokens)

                for i in range(len(pool), args.n_samples):
                    sel = pool
                    if len(sel) > cfg.k_repr:
                        sel = select_representatives(sel, cfg.k_repr, embed_fn)
                    tgt = targets[i] if arm == "g2_graph" and i < len(targets) else None
                    m_b, m_p, m_m = (graph_guided_messages(question, sel, tgt)
                                     if tgt else g2_messages(question, sel))
                    del m_b                       # base is `base7b`, not G2_BASE
                    if sel or tgt:
                        ids_p, ids_m = ids_for(m_p), ids_for(m_m)
                    else:
                        ids_p = ids_m = ids_base  # nothing to contrast yet

                    # Per (question, arm, sample) seed: reproducible, and the
                    # arms do not share an RNG stream, so one arm's token count
                    # cannot shift another's samples.
                    torch.manual_seed(hash((qid, arm, i)) & 0xFFFFFFFF)
                    gen, stats = g2_generate_ids(
                        model, ids_base, ids_p, ids_m, cfg,
                        eos_token_id=tok.eos_token_id, return_stats=True)
                    raw = tok.decode(gen, skip_special_tokens=True)
                    text, tagged = extract_answer(raw)
                    pool.append(text)

                    out.write(json.dumps({
                        "question_id": qid, "question": question,
                        "condition": arm, "rollout": i,
                        "response": text, "raw": raw, "think": "",
                        "fork_context": ctx_text, "n_forks": len(forks or []),
                        "g2_theta": theta, "g2_beta": args.beta,
                        "g2_target": tgt or "",
                        "frac_gated_off": round(stats["frac_gated_off"], 4),
                        "n_tokens": stats["n_tokens"],
                        "answer_tagged": tagged,
                    }) + "\n")
                    out.flush()
                    print(f"  Q{qid} [{arm} #{i}] {len(text)} chars  "
                          f"gated_off {stats['frac_gated_off']:.2f}"
                          f"{'' if not tgt else '  -> ' + tgt[:50]}")

    print(f"\nDone -> {args.out}")
    print(f"Judge with --k_rollouts {args.n_samples}. coverage@K is the quantity "
          f"G2 targets; `coverage` alone under-reads a pool method.")
    print("Compare arms to g2_base ONLY -- this is a 7B, v10/v11/v12 are 72B-AWQ.")


if __name__ == "__main__":
    main()
