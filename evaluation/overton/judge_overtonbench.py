"""Open-weight judge for OvertonBench: validate it, then score our responses.

Two modes (see docs/overtonbench_eval.txt):

--validate  Hold out one human rating at a time: the judge sees the
            participant's own perspective (freeresponse) plus their ratings of
            the OTHER 7 models' responses, and predicts the held-out 1-5
            rating. Reports MAE + Spearman vs the human ratings, next to the
            paper's numbers (Gemini 2.5 Pro: MAE 0.66, rho 0.66) and the
            mean-of-others baseline (rho 0.64), plus a response-length bias
            check, prediction DISPERSION (an under-dispersed judge rarely emits
            the >=4 that coverage thresholds on) and a length-strata table
            (does it under-rate SHORT answers, which are out-of-distribution
            for a judge calibrated on verbose reference responses?).

--within_n N  Within-participant discrimination on N participants: leave-one-out
            over all of one person's ratings, then rank-correlate predicted vs
            human WITHIN that person. The pooled Spearman above is dominated by
            between-participant generosity -- a response-blind baseline scores
            ~0.67 on it -- so this is the discrimination that condition
            comparison actually relies on.

--validate_aggregate  Judge vs human OvertonScore across the reference models
            (the paper's rho=0.88 check). Per-rating noise cancels under
            aggregation, so this certifies the level we actually use the judge
            at: does it rank systems the way humans do?

--score F   Judge our generated responses (JSONL from eval_overtonbench.py):
            for every participant of each question, predict their rating of
            our response (few-shot = all 8 of their rated examples), then
            aggregate exactly like the paper: mean predicted rating per
            cluster_kmeans, covered iff mean >= 4, coverage = covered /
            n_clusters, OvertonScore(condition) = mean coverage over questions.

The judge is any vLLM/OpenAI-compatible endpoint (retrieval.answer.chat).

Usage:
    python -m evaluation.overton.judge_overtonbench --validate --n 300 \
        --within_n 40 --validate_aggregate --agg_max_users 5 \
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


def _avg_ranks(xs: list[float]) -> list[float]:
    """Average (tie-aware) ranks. structure_metrics._ranks breaks ties
    arbitrarily, which is noise here: ratings are integers 1-5 over ~8 items,
    so ties dominate."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _rank_corr(xs: list[float], ys: list[float]) -> float:
    """Tie-aware Spearman. NaN if either side is constant (no ordering to score)."""
    if len(xs) < 3:
        return float("nan")
    rx, ry = _avg_ranks(xs), _avg_ranks(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


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

    # -- dispersion -----------------------------------------------------------
    # An UNDER-DISPERSED judge (predictions bunched near the mean) scores a good
    # MAE while ranking badly -- and, because coverage thresholds at >=4, it
    # rarely emits a 4, silently depressing coverage for every condition. If
    # P(pred>=4) << P(human>=4), coverage numbers are compressed, not the answers.
    from collections import Counter
    ph, hh = Counter(int(round(p)) for p in preds), Counter(int(h) for h in humans)
    print(f"\n  dispersion     : std(pred)={float(P.std()):.3f}  "
          f"std(human)={float(H.std()):.3f}")
    print(f"    pred  hist 1-5: {[ph.get(r, 0) for r in range(1, 6)]}")
    print(f"    human hist 1-5: {[hh.get(r, 0) for r in range(1, 6)]}")
    p4, h4 = float((P >= 4).float().mean()), float((H >= 4).float().mean())
    print(f"    P(pred>=4)={p4:.3f}  P(human>=4)={h4:.3f}   "
          f"<- the coverage threshold; gap {p4 - h4:+.3f}")

    # -- length strata --------------------------------------------------------
    # Route-style SHORT answers are out-of-distribution for a judge few-shot
    # calibrated on verbose reference responses; a flat pooled corr(pred,len) can
    # still hide under-rating at the short end. Negative bias = judge under-rates.
    order = sorted(range(len(preds)), key=lambda i: lens[i])
    q = max(1, len(order) // 4)
    strata = (("Q1 shortest", order[:q]), ("Q2", order[q:2 * q]),
              ("Q3", order[2 * q:3 * q]), ("Q4 longest", order[3 * q:]))
    print("\n  length strata (bias = mean pred - mean human; negative = under-rates):")
    for name, sl in strata:
        if not sl:
            continue
        bias = st.mean(preds[i] for i in sl) - st.mean(humans[i] for i in sl)
        print(f"    {name:<12} n={len(sl):<4} mean_chars={st.mean(lens[i] for i in sl):7.0f} "
              f"bias={bias:+.3f}  MAE={st.mean(abs(preds[i] - humans[i]) for i in sl):.3f}")


# ---------------------------------------------------------------------------
# Within-participant discrimination
# ---------------------------------------------------------------------------
def validate_within(idx, base_url: str, model: str, n_participants: int,
                    seed: int) -> None:
    """Can the judge RANK one participant's responses against each other?

    The pooled Spearman in validate() is dominated by BETWEEN-participant
    variance (how generous each rater is) -- mean-of-others is response-blind
    yet scores ~0.67 on it, so it barely tests response discrimination. But
    condition comparison (baseline vs scout vs route) is exactly a
    within-participant, between-response judgement. This measures that directly:
    leave-one-out over ALL of a participant's ratings, then rank-correlate the
    predicted vs human ratings within that participant, averaged over people.

    Also reports the DEGENERATE fraction -- participants for whom the judge gave
    every response the same rating (zero discrimination, undefined correlation).
    """
    rng = random.Random(seed)
    keys = [k for k, e in idx.items() if len(e["ratings"]) >= 4]
    rng.shuffle(keys)
    keys = keys[:n_participants]

    rhos, degenerate, n_calls = [], 0, 0
    for ki, key in enumerate(keys, 1):
        e = idx[key]
        preds, humans = [], []
        for i, (_, resp, rating) in enumerate(e["ratings"]):
            others = [(r, s) for j, (_, r, s) in enumerate(e["ratings"]) if j != i]
            p = predict(base_url, model, judge_messages(
                e["question"], e["perspective"], others, resp))
            n_calls += 1
            if p is None:
                continue
            preds.append(p)
            humans.append(float(rating))
        if len(preds) < 3:
            continue
        if len(set(preds)) == 1:            # judge gave everything the same score
            degenerate += 1
            continue
        r = _rank_corr(preds, humans)
        if r == r:                          # not NaN
            rhos.append(r)
        if ki % 10 == 0:
            print(f"  {ki}/{len(keys)} participants ({n_calls} calls)")

    n_ok = len(rhos)
    print(f"\nwithin-participant discrimination ({n_ok} scorable participants, "
          f"{degenerate} degenerate, {n_calls} judge calls):")
    if n_ok:
        print(f"  mean within-participant spearman = {st.mean(rhos):+.3f}")
        print(f"  median                           = {st.median(rhos):+.3f}")
        print(f"  fraction with rho > 0            = {sum(r > 0 for r in rhos)/n_ok:.2f}")
    print(f"  degenerate (all responses same rating) = "
          f"{degenerate/max(1, degenerate + n_ok):.2f}")
    print("  (this is the discrimination condition-comparison needs; a "
          "response-blind\n   baseline scores exactly 0 here, unlike the pooled "
          "spearman)")


# ---------------------------------------------------------------------------
# Aggregate validation: judge vs human OvertonScore over the reference models
# ---------------------------------------------------------------------------
def validate_aggregate(idx, base_url: str, model: str, max_users: int,
                       seed: int) -> None:
    """Reproduce the paper's aggregate check (they report rho=0.88) for OUR judge.

    Per-rating accuracy is dominated by participant noise (see validate_within),
    but OvertonScore averages over many participants and clusters, which cancels
    it -- so the level that matters is: does the judge RANK the 8 reference models
    the way humans do? Human OvertonScore comes free from the real ratings; the
    judge's comes from leave-one-out predictions of the same ratings.

    Cost ~= questions x max_users x n_models judge calls.
    """
    from collections import defaultdict

    rng = random.Random(seed)
    by_question = defaultdict(list)
    for (qid, _user), e in idx.items():
        by_question[qid].append(e)
    models = sorted({m for e in idx.values() for m, _, _ in e["ratings"]})

    est = sum(min(len(u), max_users) if max_users else len(u)
              for u in by_question.values()) * len(models)
    print(f"aggregate validation: {len(by_question)} questions x {len(models)} "
          f"reference models  (~{est} judge calls)")

    human_cov, judge_cov = defaultdict(list), defaultdict(list)
    for n_done, (qid, users) in enumerate(sorted(by_question.items()), 1):
        if max_users and len(users) > max_users:
            users = rng.sample(users, max_users)
        h_cr = defaultdict(lambda: defaultdict(list))     # model -> cluster -> ratings
        j_cr = defaultdict(lambda: defaultdict(list))
        for e in users:
            for i, (m, resp, rating) in enumerate(e["ratings"]):
                h_cr[m][e["cluster"]].append(float(rating))
                others = [(r, s) for j, (_, r, s) in enumerate(e["ratings"]) if j != i]
                p = predict(base_url, model, judge_messages(
                    e["question"], e["perspective"], others, resp))
                if p is not None:
                    j_cr[m][e["cluster"]].append(p)
        for m in models:
            for src, dst in ((h_cr, human_cov), (j_cr, judge_cov)):
                if src[m]:
                    covered = sum(1 for v in src[m].values() if st.mean(v) >= 4.0)
                    dst[m].append(covered / len(src[m]))
        if n_done % 10 == 0:
            print(f"  {n_done}/{len(by_question)} questions")

    rows = [(m, st.mean(human_cov[m]), st.mean(judge_cov[m]))
            for m in models if human_cov[m] and judge_cov[m]]
    rows.sort(key=lambda r: -r[1])
    print(f"\nOvertonScore per reference model:")
    print(f"  {'model':<34}{'human':>9}{'judge':>9}{'diff':>9}")
    for m, h, j in rows:
        print(f"  {m:<34}{h:>9.4f}{j:>9.4f}{j - h:>+9.4f}")
    if len(rows) >= 3:
        hs = [h for _, h, _ in rows]
        js = [j for _, _, j in rows]
        rho = _rank_corr(js, hs)
        mae = st.mean(abs(j - h) for _, h, j in rows)
        if len(set(js)) == 1:
            print(f"\n  *** judge OvertonScore is CONSTANT ({js[0]:.4f}) across all "
                  f"{len(rows)} models —\n      it cannot rank systems at all "
                  f"(spearman undefined). If that constant is 0.0000 the judge "
                  f"never\n      emits >=4: every condition scores 0 regardless "
                  f"of quality. ***")
        print(f"\n  spearman(judge, human) over {len(rows)} models = {rho:+.3f}"
              f"   (paper: +0.88)")
        print(f"  mean |judge - human| OvertonScore      = {mae:.4f}")
        print("  ^ THIS is the validity level our condition comparison relies on:\n"
              "    it certifies the judge ranks systems like humans do, which\n"
              "    per-rating accuracy does not.")


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
    ap.add_argument("--validate", action="store_true",
                    help="per-rating: MAE/spearman vs mean-of-others, length "
                         "bias, dispersion, length strata")
    ap.add_argument("--within_n", type=int, default=0, metavar="N_PARTICIPANTS",
                    help="within-participant discrimination over N participants "
                         "(~8 judge calls each); 0 = skip")
    ap.add_argument("--validate_aggregate", action="store_true",
                    help="judge vs human OvertonScore over the reference models "
                         "(the paper's rho=0.88 check)")
    ap.add_argument("--agg_max_users", type=int, default=5,
                    help="participants per question for --validate_aggregate")
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
    if not (args.validate or args.within_n or args.validate_aggregate or args.score):
        ap.error("pass --validate / --within_n N / --validate_aggregate / "
                 "--score RESPONSES_JSONL")

    idx = load_index(args.split)
    print(f"index: {len(idx)} (question, participant) entries")
    if args.validate:
        validate(idx, args.base_url, args.model, args.n, args.seed)
    if args.within_n:
        validate_within(idx, args.base_url, args.model, args.within_n, args.seed)
    if args.validate_aggregate:
        validate_aggregate(idx, args.base_url, args.model, args.agg_max_users,
                           args.seed)
    if args.score:
        score(idx, args.score, args.base_url, args.model,
              args.max_users, args.seed, args.out, args.k_rollouts)


if __name__ == "__main__":
    main()
