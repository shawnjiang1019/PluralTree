"""Shorten existing answers to the length humans actually rated, and nothing else.

WHY. OvertonBench's human ratings cover responses of 66-106 words (p50 88). Our
baseline sits inside that band (p50 65, 86% within it); merge_v2 does not (p50
672, 0.0% within it). So every baseline-vs-merge_v2 number is a 10x length
difference scored where no judge was ever calibrated -- and two judges extrapolate
oppositely there: rho(words, coverage) = +0.079 for Qwen2.5-72B-AWQ against
-0.244 for Qwen3.8-27B, while humans show -0.004 in band.

This takes the answers ALREADY GENERATED and rewrites them at ~90 words. Content
is held fixed (same retrieval, same merge, same drafts); only length changes. The
resulting arm can be compared against the untouched baseline rows INSIDE the
validated band, which no comparison in this project has been able to do.

Cheap by construction: one short call per existing answer, no retrieval, no
drafts, no regeneration. 180 rows is minutes.

NOT THE SAME as generating briefly. This asks "does merge_v2's content still help
when delivered at human-rated length"; a brief-merge condition would ask "can the
pipeline write better short answers". The first is the measurement question.

    python -m evaluation.overton.compress_responses \\
        --responses overton_responses_flat.jsonl --condition merge_v2 \\
        --base_url http://localhost:8000/v1 --model <model> \\
        --out overton_responses_compressed.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

COMPRESS_INSTRUCTION = (
    "You will see a question and a long answer to it.\n"
    "Rewrite the answer in about {n} words.\n"
    "Rules:\n"
    "- Keep as MANY of the distinct positions as you can. Drop elaboration, "
    "examples and repetition -- never drop a position to make room.\n"
    "- Keep any group attributions (which groups hold which view).\n"
    "- Add nothing that is not in the original answer.\n"
    "Put the rewritten answer inside <answer></answer> tags. The reader sees "
    "ONLY what is inside those tags."
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compress answers into the human-rated band")
    ap.add_argument("--responses", required=True, help="existing responses jsonl")
    ap.add_argument("--condition", default="merge_v2", help="condition to compress")
    ap.add_argument("--keep", default="baseline",
                    help="comma list of conditions copied through UNCHANGED, so "
                         "the judge scores both arms from one file and pairs them")
    ap.add_argument("--label", default=None,
                    help="output condition name (default: <condition>_compressed)")
    ap.add_argument("--target_words", type=int, default=90,
                    help="90 = the median length of the human-rated responses")
    ap.add_argument("--match_condition", default=None,
                    help="match each rewrite to THIS condition's word count for "
                         "the same question instead of a fixed target (e.g. "
                         "baseline). Measured: a fixed 'about 90 words' target "
                         "came back at p50 128, and baseline is itself often "
                         "below the 66-word band floor -- so matching arm to arm "
                         "is what makes the comparison length-controlled.")
    ap.add_argument("--retry_over", type=float, default=1.25,
                    help="rewrite once more, with a hard cap, when the first "
                         "attempt exceeds this multiple of the target. 0 = off.")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_questions", type=int, default=0)
    ap.add_argument("--out", default="overton_responses_compressed.jsonl")
    ap.add_argument("--dry_run", action="store_true", help="print one prompt and exit")
    args = ap.parse_args()

    from retrieval.answer import chat, extract_answer

    label = args.label or f"{args.condition}_compressed"
    keep = {c.strip() for c in args.keep.split(",") if c.strip()}

    rows = []
    with open(args.responses, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    qids = sorted({int(r["question_id"]) for r in rows})
    if args.max_questions:
        qids = qids[: args.max_questions]
    wanted = set(qids)
    todo = [r for r in rows if r["condition"] == args.condition
            and int(r["question_id"]) in wanted]
    passthrough = [r for r in rows if r["condition"] in keep
                   and int(r["question_id"]) in wanted]
    if not todo:
        print(f"no {args.condition!r} rows in {args.responses}"); return 1
    print(f"{len(todo)} to compress -> {label}; {len(passthrough)} copied unchanged "
          f"({', '.join(sorted(keep))})")

    if args.dry_run:
        r = todo[0]
        print(COMPRESS_INSTRUCTION.format(n=args.target_words))
        print(f"\nQuestion: {r['question']}\n\nAnswer:\n{r['response'][:400]}...")
        return 0

    done: set[tuple[int, str, int]] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    done.add((int(d["question_id"]), d["condition"], int(d.get("rollout", 0))))
        print(f"  resuming: {len(done)} rows already present")

    # Per-question length of the arm we are matching, so each rewrite gets its
    # own target rather than one global number.
    match_len: dict[tuple[int, int], int] = {}
    if args.match_condition:
        for r in rows:
            if r["condition"] == args.match_condition:
                match_len[(int(r["question_id"]), int(r.get("rollout", 0)))] = \
                    len((r.get("response") or "").split())
        if not match_len:
            print(f"no {args.match_condition!r} rows to match against"); return 1
        vals = sorted(match_len.values())
        print(f"  matching lengths to {args.match_condition}: "
              f"p50={vals[len(vals) // 2]} words")

    n_long = n_empty = n_retry = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for r in passthrough:
            key = (int(r["question_id"]), r["condition"], int(r.get("rollout", 0)))
            if key not in done:
                f.write(json.dumps(r) + "\n")
        f.flush()
        for i, r in enumerate(todo, 1):
            key = (int(r["question_id"]), label, int(r.get("rollout", 0)))
            if key in done:
                continue
            src = r.get("response") or ""
            qkey = (int(r["question_id"]), int(r.get("rollout", 0)))
            target = match_len.get(qkey, args.target_words) if args.match_condition \
                else args.target_words
            target = max(30, target)          # a 12-word target is not answerable
            user = f"Question: {r['question']}\n\nAnswer:\n{src}"

            def _rewrite(instr: str) -> str:
                raw = chat(args.base_url, args.model,
                           [{"role": "system", "content": instr},
                            {"role": "user", "content": user}],
                           temperature=0.0, max_tokens=max(256, target * 3))
                t, _tagged = extract_answer(raw)
                return t.strip()

            text = _rewrite(COMPRESS_INSTRUCTION.format(n=target))
            n_out = len(text.split())
            # Measured overshoot: "about N words" comes back ~1.4x N. One retry
            # with a hard ceiling costs a call and keeps the arms matched.
            if args.retry_over and n_out > args.retry_over * target:
                n_retry += 1
                retry = _rewrite(
                    COMPRESS_INSTRUCTION.format(n=target)
                    + f"\nHARD LIMIT: the answer must be at most {target} words. "
                      f"Your previous attempt was {n_out} words, which is too long.")
                if retry and len(retry.split()) < n_out:
                    text, n_out = retry, len(retry.split())
            if not text:
                n_empty += 1
            elif n_out > 2 * target:
                n_long += 1
            f.write(json.dumps({
                "question_id": int(r["question_id"]), "question": r["question"],
                "condition": label, "rollout": int(r.get("rollout", 0)),
                "response": text, "n_forks": r.get("n_forks", 0),
                "words_before": len(src.split()), "words_after": n_out,
                "target_words": target, "source_condition": args.condition}) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(todo)} ...")

    if n_retry:
        print(f"  {n_retry} rewrites needed the hard-limit retry")
    if n_empty:
        print(f"  WARNING {n_empty} empty rewrites -- they will score 0")
    if n_long:
        print(f"  WARNING {n_long} rewrites over 2x the target: the model ignored "
              f"the length instruction, and the arm is not length-matched")
    print(f"\nDone -> {args.out}")
    print("Check the manipulation BEFORE scoring: the compressed arm must land in "
          "the 66-106 word band, like baseline does.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
