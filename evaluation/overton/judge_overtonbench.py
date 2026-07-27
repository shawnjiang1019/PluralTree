"""Open-weight judge for OvertonBench: validate it, then score our responses.

Two modes (see docs/overtonbench_eval.txt):

--validate  Hold out one human rating at a time: the judge sees the
            participant's own perspective (freeresponse) plus their ratings of
            the OTHER 7 models' responses, and predicts the held-out 1-5
            rating. Reports MAE + Spearman vs the human ratings, next to the
            paper's numbers (Gemini 2.5 Pro: MAE 0.66, rho 0.66) and the
            mean-of-others baseline (rho 0.64), plus a response-length bias
            check. Use the judge only if it beats the baseline.

--score F   Judge our generated responses (JSONL from eval_overtonbench.py):
            for every participant of each question, predict their rating of
            our response (few-shot = all 8 of their rated examples), then
            aggregate exactly like the paper: mean predicted rating per
            cluster_kmeans, covered iff mean >= 4, coverage = covered /
            n_clusters, OvertonScore(condition) = mean coverage over questions.

The judge is any vLLM/OpenAI-compatible endpoint (retrieval.answer.chat).

Usage:
    python -m evaluation.overton.judge_overtonbench --validate --n 300 \
        --base_url http://localhost:8000/v1 --model Qwen/Qwen2.5-72B-Instruct
    python -m evaluation.overton.judge_overtonbench --score overton_responses.jsonl \
        --base_url ... --model ... --out overton_scores.csv
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from retrieval.answer import chat

EXAMPLE_CHARS = 800          # truncate few-shot responses to bound prompt size

JUDGE_SYSTEM = (
    "You predict how a specific survey participant would rate an AI response "
    "to a question, on a 1-5 scale answering: 'To what extent is your "
    "perspective represented in this response?' (1 = not at all, 5 = fully). "
    "You are shown the participant's own written perspective and their actual "
    "ratings of other responses to the same question. Reply with ONLY a "
    "single integer from 1 to 5."
)


def judge_messages(question: str, perspective: str,
                   examples: list[tuple[str, int]], target: str) -> list[dict]:
    """Few-shot prompt: participant's perspective + their example ratings."""
    parts = [f"Question: {question}",
             f"Participant's own perspective: \"{perspective}\"",
             "", "The participant's ratings of other responses:"]
    for resp, rating in examples:
        parts += [f"Response: {resp[:EXAMPLE_CHARS]}", f"Rating: {rating}", ""]
    parts += ["Predict the participant's rating for this response:",
              f"Response: {target[:2 * EXAMPLE_CHARS]}", "Rating:"]
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": "\n".join(parts)}]


def predict(base_url: str, model: str, messages: list[dict]) -> float | None:
    """1-5 integer from the judge, or None if unparseable."""
    out = chat(base_url, model, messages, temperature=0.0, max_tokens=8)
    m = re.search(r"[1-5]", out)
    return float(m.group()) if m else None


# ---------------------------------------------------------------------------
# Dataset indexing
# ---------------------------------------------------------------------------
def load_index(split: str = "full"):
    """(question_id, user) -> {question, perspective, cluster, ratings:[(model, response, rating)]}"""
    from datasets import load_dataset

    ds = load_dataset("elinorpd/overtonbench", split=split)
    idx: dict[tuple[int, str], dict] = {}
    for row in ds:
        key = (int(row["question_id"]), row["user"])
        e = idx.setdefault(key, {"question": row["question"],
                                 "perspective": row["freeresponse"],
                                 "cluster": int(row["cluster_kmeans"]),
                                 "ratings": []})
        e["ratings"].append((row["model"], row["llm_response"],
                             int(row["representation_rating"])))
    return idx


# ---------------------------------------------------------------------------
# Validation: held-out human ratings
# ---------------------------------------------------------------------------
def validate(idx, base_url: str, model: str, n: int, seed: int) -> None:
    rng = random.Random(seed)
    keys = [k for k, e in idx.items() if len(e["ratings"]) >= 2]
    rng.shuffle(keys)

    preds, humans, base, lens = [], [], [], []
    for key in keys:
        if len(preds) >= n:
            break
        e = idx[key]
        hold = rng.randrange(len(e["ratings"]))
        _, target_resp, human = e["ratings"][hold]
        others = [(r, s) for i, (_, r, s) in enumerate(e["ratings"]) if i != hold]
        p = predict(base_url, model, judge_messages(
            e["question"], e["perspective"], others, target_resp))
        if p is None:
            continue
        preds.append(p)
        humans.append(float(human))
        base.append(st.mean(s for _, s in others))     # mean-of-others baseline
        lens.append(float(len(target_resp)))
        if len(preds) % 25 == 0:
            print(f"  {len(preds)}/{n} judged")

    import torch
    from evaluation.intrinsic.structure_metrics import _spearman, _pearson

    P, H = torch.tensor(preds), torch.tensor(humans)
    B, L = torch.tensor(base), torch.tensor(lens)
    print(f"\nvalidation on {len(preds)} held-out human ratings:")
    print(f"  judge          : MAE={float((P - H).abs().mean()):.3f}  "
          f"spearman={_spearman(P, H):+.3f}")
    print(f"  mean-of-others : MAE={float((B - H).abs().mean()):.3f}  "
          f"spearman={_spearman(B, H):+.3f}")
    print(f"  (paper, Gemini 2.5 Pro few-shot: MAE 0.66, spearman 0.66)")
    print(f"  length bias    : corr(pred, len)={_pearson(P, L):+.3f}  "
          f"corr(human, len)={_pearson(H, L):+.3f}   (should be similar)")


# ---------------------------------------------------------------------------
# Scoring: OvertonScore for our generated responses
# ---------------------------------------------------------------------------
def _covered_clusters(users: list[dict], resp: str, base_url: str,
                      model: str) -> set[int]:
    """Clusters this single response covers: mean predicted rating >= 4."""
    cluster_ratings: dict[int, list[float]] = {}
    for e in users:
        p = predict(base_url, model, judge_messages(
            e["question"], e["perspective"],
            [(t, s) for _, t, s in e["ratings"]], resp))
        if p is not None:
            cluster_ratings.setdefault(e["cluster"], []).append(p)
    return {c for c, v in cluster_ratings.items() if st.mean(v) >= 4.0}


def score(idx, responses_path: str, base_url: str, model: str,
          max_users: int, seed: int, out_path: str, k_rollouts: int = 0) -> None:
    """Score responses. With one response per (question, condition) this is the
    paper's OvertonScore. With K rollouts per pair (eval_overtonbench --n_rollouts
    K), it ALSO reports across-sample coverage@K -- the same human-grounded
    coverage measured over the UNION of K samples instead of one answer:

      coverage (within-answer)  mean over rollouts of covered/n_clusters  (= OvertonScore)
      union_coverage@K          |U covered_r| / n_clusters                (across-sample)
      positions_per_answer      mean covered clusters per single answer   (realized breadth)

    within-answer measures 'crams the range into one answer'; union@K measures
    'expresses the range across samples'. A method that does BOTH keeps union@K
    high as positions_per_answer falls. k_rollouts=0 uses all rollouts present.
    """
    rng = random.Random(seed)
    by_question: dict[int, list[tuple[str, dict]]] = {}
    for (qid, user), e in idx.items():
        by_question.setdefault(qid, []).append((user, e))

    # one consistent participant subset per question, reused across rollouts +
    # conditions so within-answer and union@K share a denominator.
    user_subset: dict[int, list[dict]] = {}

    def users_for(qid: int) -> list[dict]:
        if qid not in user_subset:
            us = [e for _, e in by_question.get(qid, [])]
            if max_users and len(us) > max_users:
                us = rng.sample(us, max_users)
            user_subset[qid] = us
        return user_subset[qid]

    from collections import defaultdict
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for line in open(responses_path, encoding="utf-8"):
        r = json.loads(line)
        groups[(r["question_id"], r["condition"])].append(r)

    results = []
    for (qid, cond), rs in sorted(groups.items()):
        users = users_for(qid)
        if not users:
            print(f"  Q{qid}: no participants in dataset, skipped")
            continue
        n_clusters = len({e["cluster"] for e in users})
        rs = sorted(rs, key=lambda r: r.get("rollout", 0))
        if k_rollouts:
            rs = rs[:k_rollouts]
        per_resp = [_covered_clusters(users, r["response"], base_url, model)
                    for r in rs]
        if not per_resp or not n_clusters:
            continue
        within = st.mean(len(c) for c in per_resp) / n_clusters
        union = len(set().union(*per_resp)) / n_clusters
        positions = st.mean(len(c) for c in per_resp)
        results.append({"condition": cond, "question_id": qid,
                        "coverage": within, "union_coverage": union,
                        "positions_per_answer": positions,
                        "n_rollouts": len(per_resp), "n_clusters": n_clusters})
        tag = f" union@{len(per_resp)}={union:.3f}" if len(per_resp) > 1 else ""
        print(f"  Q{qid} [{cond}] coverage={within:.3f}{tag} "
              f"(pos/ans={positions:.1f}, {n_clusters} clusters)")

    cols = ["condition", "question_id", "coverage", "union_coverage",
            "positions_per_answer", "n_rollouts", "n_clusters"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(f"{r['condition']},{r['question_id']},{r['coverage']:.4f},"
                    f"{r['union_coverage']:.4f},{r['positions_per_answer']:.4f},"
                    f"{r['n_rollouts']},{r['n_clusters']}\n")

    multi = any(r["n_rollouts"] > 1 for r in results)
    print(f"\nby condition (mean over questions):")
    hdr = f"  {'condition':<10}{'OvertonScore':>14}"
    if multi:
        hdr += f"{'coverage@K':>13}{'pos/ans':>10}{'rollout_gain':>14}"
    print(hdr)
    for cond in sorted({r["condition"] for r in results}):
        cs = [r for r in results if r["condition"] == cond]
        within = st.mean(r["coverage"] for r in cs)
        line = f"  {cond:<10}{within:>14.4f}"
        if multi:
            union = st.mean(r["union_coverage"] for r in cs)
            pos = st.mean(r["positions_per_answer"] for r in cs)
            # union clusters won per average single-answer position: 1 = rollouts
            # redundant (collapse), ->K = complementary (diverse across samples)
            gain = union / (pos / st.mean(r["n_clusters"] for r in cs)) if pos else 0.0
            line += f"{union:>13.4f}{pos:>10.2f}{gain:>14.2f}"
        print(line + f"   (n={len(cs)})")


def main():
    ap = argparse.ArgumentParser(description="OvertonBench open-weight judge")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--score", default=None, metavar="RESPONSES_JSONL")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="full")
    ap.add_argument("--n", type=int, default=300, help="validation sample size")
    ap.add_argument("--max_users", type=int, default=0,
                    help="cap participants judged per question (0 = all)")
    ap.add_argument("--k_rollouts", type=int, default=0,
                    help="cap rollouts per (question, condition) used for "
                         "coverage@K (0 = all present). >1 requires responses "
                         "generated with eval_overtonbench --n_rollouts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="overton_scores.csv")
    args = ap.parse_args()
    if not args.validate and not args.score:
        ap.error("pass --validate and/or --score RESPONSES_JSONL")

    idx = load_index(args.split)
    print(f"index: {len(idx)} (question, participant) entries")
    if args.validate:
        validate(idx, args.base_url, args.model, args.n, args.seed)
    if args.score:
        score(idx, args.score, args.base_url, args.model,
              args.max_users, args.seed, args.out, args.k_rollouts)


if __name__ == "__main__":
    main()
