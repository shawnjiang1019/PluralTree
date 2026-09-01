"""OvertonBench generation under context-aware decoding (docs/cad_experiment.md).

    logits = (1 + alpha) * logits(y | forks, x)  -  alpha * logits(y | x)

`alpha < 0` suppresses the injected forks — the anti-anchoring direction, which
no prompt can express, and the one the measured failure calls for (framing_hurts:
on-pole similarity 0.334 -> 0.389, corr(attraction, dcoverage) = -0.31).

WHY A SEPARATE DRIVER. CAD needs logits, so it needs local HF weights.
`eval_overtonbench.py` talks to a vLLM OpenAI endpoint, which exposes neither.
Rows are written in the SAME schema so `judge_overtonbench.py` scores this
without changes.

ARMS. Every arm runs through `cad_generate_ids`, including the two references,
so the sampling code, temperature, top-p and RNG consumption are identical and
the only difference between arms is the decoding rule:

    base7b   ctx=base       plain=base       alpha=0   the plain-pipeline reference
    ctx0     ctx=forks      plain=forks      alpha=0   plain with-context decoding
    cad<a>   ctx=forks      plain=noctx      alpha=a   the contrast
    cad_soft ctx=forks      plain=noctx      alpha=f(contestedness)

where `base` = BASELINE_INSTRUCTION with no forks, `noctx` = the SAME
PLURALISM_INSTRUCTION as `forks` but with the fork block removed. Contrasting
against `base` instead of `noctx` would make alpha scale the instruction change
as well as the forks, which is not the quantity under test.

`ctx0` is load-bearing. Without it, a CAD-vs-baseline difference confounds the
decoding change with the MODEL change (7B here, 72B-AWQ in v9/v10). These numbers
are NOT comparable to v9/v10 — `base7b` is the only valid reference.

    python -m evaluation.overton.eval_cad --embeddings embeddings_opinionqa.pt \\
        --dataset opinionqa --model <local 7B dir> \\
        --arms base7b,ctx0,cad-0.5,cad-0.25,cad0.25,cad0.5 \\
        --out cad_responses.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.overton.eval_overtonbench import load_questions


def parse_arm(arm: str):
    """'cad-0.5' -> ('cad', -0.5). Returns (kind, alpha)."""
    if arm in ("base7b", "ctx0"):
        return arm, 0.0
    if arm == "cad_soft":
        return "cad_soft", None
    if arm.startswith("cad"):
        try:
            return "cad", float(arm[3:])
        except ValueError as e:                       # noqa: BLE001
            raise SystemExit(f"bad arm {arm!r}: expected e.g. cad-0.5") from e
    raise SystemExit(f"unknown arm {arm!r}")


def main():
    ap = argparse.ArgumentParser(description="OvertonBench under CAD")
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
    ap.add_argument("--arms", default="base7b,ctx0,cad-0.5,cad-0.25,cad0.25,cad0.5")
    ap.add_argument("--half", choices=["all", "a", "b"], default="all",
                    help="tune alpha on 'a', report on 'b'; picking the best "
                         "alpha over all 60 is the mistake bestofk flagged")
    ap.add_argument("--labels", default="contestedness_labels.json",
                    help="cad_soft: per-question contestedness scores")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="cad_responses.jsonl")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pluraltree.manifolds.poincare import PoincareBall
    from retrieval.answer import (BASELINE_INSTRUCTION, PLURALISM_INSTRUCTION,
                                  build_prompt, extract_answer, forks_to_context)
    from retrieval.cad import CADConfig, alpha_from_contestedness, cad_generate_ids
    from retrieval.scout import (ScoutConfig, embed_question,
                                 load_or_compute_text_feat, scout)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    parsed = [(a, *parse_arm(a)) for a in arms]

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

    scores = {}
    if any(k == "cad_soft" for _, k, _ in parsed):
        if not os.path.exists(args.labels):
            ap.error(f"cad_soft needs {args.labels} (from jobs/train/job_probe.sh)")
        with open(args.labels, encoding="utf-8") as f:
            raw = json.load(f)
        rows = raw["rows"] if isinstance(raw, dict) and "rows" in raw else raw
        for r in rows:
            scores[int(r.get("qid", r.get("question_id")))] = float(r["score"])
        lo, hi = min(scores.values()), max(scores.values())
        rng = (hi - lo) or 1.0
        scores = {q: (v - lo) / rng for q, v in scores.items()}   # -> [0,1]
        print(f"cad_soft: {len(scores)} contestedness scores, raw range "
              f"[{lo:.3f}, {hi:.3f}] rescaled to [0,1]")

    print(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype)).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    questions = load_questions(args.split)
    if args.half != "all":                     # deterministic, id-based
        want = 0 if args.half == "a" else 1
        questions = [(q, t) for q, t in questions if q % 2 == want]
    if args.max_questions:
        questions = questions[: args.max_questions]

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["question_id"], r["condition"]))
        print(f"resuming: {len(done)} rows already present")

    def ids_for(msgs):
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt")
        return enc["input_ids"] if hasattr(enc, "input_ids") else enc

    print(f"{len(questions)} questions x {len(arms)} arms -> {args.out}")
    with open(args.out, "a", encoding="utf-8") as out:
        for qid, question in questions:
            q_emb = embed_question(question)
            forks = scout(question, graph, h_all, text_feat, manifold,
                          cfg=cfg_scout, q_emb=q_emb)
            # THREE prompts, not two. The CAD contrast must vary ONLY the fork
            # block: `msgs_noctx` keeps PLURALISM_INSTRUCTION and drops the
            # forks, so `A - B` is the forks' effect and nothing else. Pairing
            # the fork prompt against BASELINE_INSTRUCTION (the first version of
            # this file) made alpha scale "forks + a different instruction",
            # which is not the quantity the experiment is about.
            msgs_ctx = build_prompt(question, forks, graph,
                                    PLURALISM_INSTRUCTION, False)
            msgs_noctx = build_prompt(question, None, graph,
                                      PLURALISM_INSTRUCTION, False)
            msgs_base = build_prompt(question, None, graph,
                                     BASELINE_INSTRUCTION, False)
            ctx_text = forks_to_context(forks, graph, False) if forks else ""
            ids_ctx = ids_for(msgs_ctx)
            ids_noctx = ids_for(msgs_noctx)
            ids_base = ids_for(msgs_base)

            for arm, kind, alpha in parsed:
                if (qid, arm) in done:
                    continue
                if kind == "cad_soft":
                    if qid not in scores:
                        print(f"  Q{qid} [{arm}] no contestedness score, skipped")
                        continue
                    alpha = alpha_from_contestedness(scores[qid])
                if kind == "base7b":
                    a_ids, b_ids = ids_base, ids_base       # plain pipeline ref
                elif kind == "ctx0":
                    a_ids, b_ids = ids_ctx, ids_ctx         # with-context, a=0
                else:
                    if not forks:
                        print(f"  Q{qid} [{arm}] no forks, skipped")
                        continue
                    a_ids, b_ids = ids_ctx, ids_noctx       # forks only

                # Per (question, arm) seed: reproducible, and the arms do not
                # share an RNG stream, so one arm's token count cannot shift
                # another's samples.
                torch.manual_seed(hash((qid, arm)) & 0xFFFFFFFF)
                cfg = CADConfig(alpha=alpha, temperature=args.temperature,
                                top_p=args.top_p, max_new_tokens=args.max_new_tokens)
                gen = cad_generate_ids(model, a_ids, b_ids, cfg,
                                       eos_token_id=tok.eos_token_id)
                raw = tok.decode(gen, skip_special_tokens=True)
                text, tagged = extract_answer(raw)

                out.write(json.dumps({
                    "question_id": qid, "question": question, "condition": arm,
                    "rollout": 0, "response": text, "raw": raw,
                    "think": "", "fork_context": ctx_text,
                    "n_forks": len(forks or []),
                    "cad_alpha": alpha, "cad_kind": kind,
                    "answer_tagged": tagged,
                }) + "\n")
                out.flush()
                print(f"  Q{qid} [{arm} a={alpha:+.2f}] {len(text)} chars"
                      f"{'' if tagged else '  (no <answer> tags)'}")

    print(f"\nDone -> {args.out}")
    print("Judge it with judge_overtonbench.py --score, and compare arms ONLY to "
          "base7b/ctx0 -- not to v9/v10, which are a different model.")


if __name__ == "__main__":
    main()
