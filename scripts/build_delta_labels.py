"""Delta labels on GRAPH questions, so the router stops being fit on its own eval set.

WHY. `scripts/analysis/delta_regressor.py` learns to predict
`coverage(injected) - coverage(baseline)` per question and route on it. Oracle
routing is worth +0.137 over always-baseline, over 4x the 0.027 noise floor, and
nothing currently collects it. But its labels come from scored OvertonBench runs:
**60 questions per run**, and those same 60 questions ARE the evaluation set. Two
consequences, and the second is the binding one:

  * statistical -- 60 questions x a handful of conditions is a few hundred rows
    with near-ties dominating; leave-one-question-out is the only CV that is not
    noise, and the routing metric over a 12-question test split moves by more
    than the entire +0.137 gap on resampling.
  * epistemic  -- a router fit on those 60 and then scored on OvertonBench is fit
    and evaluated on the same questions. Cross-validation removes the crudest
    form of it and nothing else (feature set, cost ratio, alpha grid and the
    decision to build the thing were all chosen while looking at them). No amount
    of CV turns that into a number reportable ON OvertonBench.

The graph holds ~1,492 ATP + ~2,500 GOQA questions that are not the eval set.
This script turns them into labels in the exact schema delta_regressor already
consumes, so the router can be FIT off-benchmark and REPORTED on it.

    graph question -> scout -> anchor -> positions_from_subtree
                   -> answer(baseline) , answer(scout)      [retrieval.answer]
                   -> coverage_reward both                  [alignment.reward]
                   -> delta = reward(scout) - reward(baseline)

Nothing here is a second generation path: generation is `retrieval.answer.answer`
and scoring is `alignment.reward.coverage_reward`, both imported.

LABEL QUALITY, STATED UP FRONT. `coverage_reward` currently FAILS its validation
gate (docs/reward_gate_failure.md, results/reward_corr.txt): within-question
pairwise concordance with the OvertonBench judge is **0.078** against a 0.500
chance level. That number is mostly UNDEFINED rather than anti-correlated -- a
reward tie is charged as a disagreement, and the reward is zero on 92% of
responses at the default d=60. Decomposed: tie_rate 0.858, and on the pairs it
CAN separate concordance is **0.735** (n=34). The diagnosis is a miscalibrated
`match_thr`: 0.50 sits above the 75th percentile of the cosine distribution it
thresholds (pos_best p50=0.341, p75=0.475, only 18% of positions clear 0.50).
Wrong threshold, not wrong objective -- but that is a hypothesis with n=34
behind it, not a finding.

WHY BUILD IT NOW ANYWAY. The gate that failed is GRPO's gate, and GRPO is the
harshest possible consumer: its advantage is computed strictly within a group of
rollouts sharing one prompt, so a reward that ties 85% of pairs supplies no
gradient at all. A ridge regression aggregating hundreds of labels is a different
consumer -- independent label noise costs it SAMPLE EFFICIENCY (an attenuated
coefficient, a wider CV interval), it does not corrupt the objective the way a
noisy per-rollout advantage does. And the recalibration is queued; this pipeline
is what consumes it (`--score_only`, below, rescores without regenerating).

THE METRIC TO WATCH is `both_zero_rate`: the fraction of questions where the
reward scored BOTH answers 0.0. Those rows carry delta = 0 by construction and no
information about routing. They are EMITTED, flagged with a `both_zero` column,
and reported -- never silently dropped, because dropping them would make the
label set look healthy exactly as it got emptier. At the current match_thr expect
it high (the v6 run was 92% zero per response); the recalibration succeeds if and
only if this falls.

LABELS ARE MODEL-SPECIFIC. "Did injection help" is a statement about a particular
baseline: a 72B that already covers the spectrum unaided has nothing to gain from
injection, a 7B has plenty. Labels generated with Qwen-72B-AWQ do NOT transfer to
a 7B policy. Whatever model you route for must be the model that generates these.
The model name rides in every row and in the sidecar meta.

    # generate (needs vLLM + graph + embeddings; cluster-side)
    python scripts/build_delta_labels.py --embeddings embeddings_opinionqa.pt \
        --text_feat feats_opinionqa.pt --n 500 \
        --base_url http://localhost:8000/v1 --model Qwen/Qwen2.5-72B-Instruct-AWQ \
        --out results/delta_labels_reward_scores.csv

    # RESCORE after the reward is recalibrated -- no endpoint, no GPU, no regeneration
    python scripts/build_delta_labels.py --score_only --match_thr 0.35 \
        --out results/delta_labels_reward_scores.csv

    # consume
    python scripts/analysis/delta_regressor.py \
        --scores results/delta_labels_reward_scores.csv --features causal

    python scripts/build_delta_labels.py --selftest     # no endpoint, no graph
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                        # repo root
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts", "analysis"))

# Labels from this file are REWARD-labelled; OvertonBench labels are JUDGE-labelled
# and they are not the same measurement (the two agree at 0.078 within-question).
# The tag rides on every row so a pooled fit cannot mix them without saying so.
LABEL_SOURCE = "reward_coverage_v2"
BASELINE_COND = "baseline"

# scores csv: the three columns delta_regressor.load_coverage reads, then
# provenance. Extra columns are ignored by its DictReader, so this is free.
SCORE_FIELDS = ["question_id", "condition", "coverage", "label_source", "model",
                "match_thr", "min_depth_words", "anchor", "n_positions",
                "n_units", "n_mentioned", "recall", "precision", "both_zero"]


# ---------------------------------------------------------------------------
# Question sampling (the holdout guard is not optional)
# ---------------------------------------------------------------------------
def sample_questions(candidates, holdout, seed: int) -> tuple[list, int]:
    """Shuffle graph questions deterministically, dropping the eval holdout.

    ``candidates`` is [(qid, text)]; ``holdout`` the normalized OvertonBench
    question texts from alignment.rollout_dataset.load_eval_holdout_texts().
    Returns (pool, n_dropped). The whole point of this file is labels that are
    NOT the eval set, so a leak here silently reintroduces the contamination it
    exists to remove -- hence the count is returned and printed, not assumed 0.
    """
    from alignment.rollout_dataset import _norm

    kept, dropped = [], 0
    for qid, text in candidates:
        if _norm(text) in holdout:
            dropped += 1
            continue
        kept.append((qid, text))
    # shuffled, not head-of-list: graph question ids are ordered by ATP wave, so
    # the first N would all come from the same few surveys.
    random.Random(seed).shuffle(kept)
    return kept, dropped


def graph_questions(graph) -> list:
    """(node_id, text) for every 'q:' node -- same enumeration as rollout_dataset."""
    return [(nid, graph.entity_text.get(nid) or name)
            for nid, name in enumerate(graph.id_to_entity) if name.startswith("q:")]


# ---------------------------------------------------------------------------
# Generation (resumable; the expensive half)
# ---------------------------------------------------------------------------
def load_done(path: str) -> dict:
    """(question_id, condition) -> row, from a partial responses jsonl.

    Same resume contract as evaluation/hivemind/generate_g2.py and
    eval_overtonbench: key on the cell, skip what is present. This is thousands
    of generations across a walltime-limited job, so a resubmit must not re-pay
    for what already landed.
    """
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            done[(int(r["question_id"]), r["condition"])] = r
    return done


def generate(pool, *, n, inject_cond, out_resp, embed_q_fn, anchor_fn, positions_at,
             answer_fn, min_positions=2, log_every=25):
    """Answer the first `n` usable questions under (baseline, inject_cond).

    Usable = scout returns a fork AND its anchor subtree yields >= min_positions.
    A question failing either is skipped and counted, never quietly padded --
    those counts are the sampling frame for whatever the labels end up saying.

    Seams (`anchor_fn`, `positions_at`, `answer_fn`) exist so --selftest can run
    the whole loop with stubs; main() wires them to scout / positions_from_subtree
    / retrieval.answer.answer.

    Returns (accepted qids, {qid: anchor}, stats).
    """
    conds = (BASELINE_COND, inject_cond)
    done = load_done(out_resp)
    anchors = {qid: r["anchor"] for (qid, _c), r in done.items()
               if r.get("anchor") is not None}
    n_resumed = len(done)

    accepted, n_no_fork, n_thin, n_calls = [], 0, 0, 0
    os.makedirs(os.path.dirname(out_resp) or ".", exist_ok=True)
    with open(out_resp, "a", encoding="utf-8") as f:
        for qid, question in pool:
            if len(accepted) >= n:
                break
            if all((qid, c) in done for c in conds):
                accepted.append(qid)         # already complete: costs nothing
                continue
            q_emb = embed_q_fn(question)
            anchor = anchor_fn(question, q_emb)
            if anchor is None:
                n_no_fork += 1
                continue
            if len(positions_at(anchor)) < min_positions:
                n_thin += 1
                continue
            for cond in conds:
                if (qid, cond) in done:
                    continue
                resp, trace = answer_fn(question, cond, q_emb)
                n_calls += 1
                # delta_regressor.load_responses reads response/raw/think/
                # fork_context; `anchor` is ours, and it is what makes --score_only
                # possible without re-running the scout.
                row = {"question_id": int(qid), "question": question,
                       "condition": cond, "rollout": 0, "anchor": int(anchor),
                       "response": resp, "raw": trace["raw"],
                       "think": trace["think"],
                       "fork_context": trace["fork_context"],
                       "n_forks": trace["n_forks"],
                       "label_source": LABEL_SOURCE}
                f.write(json.dumps(row) + "\n")
                f.flush()
                done[(qid, cond)] = row
            anchors[qid] = int(anchor)
            accepted.append(qid)
            if log_every and len(accepted) % log_every == 0:
                print(f"  {len(accepted)}/{n} questions "
                      f"({n_calls} generations this run)")
    stats = {"n_resumed_rows": n_resumed, "n_generated": n_calls,
             "n_skipped_no_fork": n_no_fork, "n_skipped_thin_subtree": n_thin,
             "n_pool": len(pool)}
    return accepted, anchors, stats


# ---------------------------------------------------------------------------
# Scoring (cheap, idempotent, always recomputed -- this is the recalibration path)
# ---------------------------------------------------------------------------
def score(out_resp, *, inject_cond, positions_at, embed_fn, cfg, model=""):
    """coverage_reward on both answers per question -> long-form score rows.

    Rewritten in full on every run rather than resumed: the whole point of
    exposing --match_thr is that the labels can be REGENERATED at a recalibrated
    threshold, and a resumable score file would silently mix thresholds.

    coverage_reward re-embeds the positions on each of the two calls. That is
    ~4-8 short texts of redundant work per question against hundreds of response
    units, and paying it keeps the scoring path literally `coverage_reward`
    rather than a reimplementation of it around embed_positions/_score_v2.
    """
    from alignment.reward import coverage_reward

    by_q = {}
    with open(out_resp, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by_q.setdefault(int(r["question_id"]), {})[r["condition"]] = r

    rows, n_both_zero, deltas, n_thin = [], 0, [], 0
    for qid in sorted(by_q):
        cell = by_q[qid]
        if BASELINE_COND not in cell or inject_cond not in cell:
            continue                     # half-generated question: not a label yet
        anchor = cell[BASELINE_COND].get("anchor")
        if anchor is None:
            continue
        positions = positions_at(int(anchor))
        if len(positions) < 2:
            n_thin += 1
            continue
        scored = {}
        for cond in (BASELINE_COND, inject_cond):
            scored[cond] = coverage_reward(cell[cond].get("response") or "",
                                           positions, embed_fn, cfg)
        both_zero = all(v[0] <= 0.0 for v in scored.values())
        n_both_zero += int(both_zero)
        deltas.append(scored[inject_cond][0] - scored[BASELINE_COND][0])
        for cond in (BASELINE_COND, inject_cond):
            cov, bd = scored[cond]
            rows.append({"question_id": qid, "condition": cond, "coverage": cov,
                         "label_source": LABEL_SOURCE, "model": model,
                         "match_thr": cfg.match_thr,
                         "min_depth_words": cfg.min_depth_words,
                         "anchor": int(anchor), "n_positions": len(positions),
                         "n_units": bd["n_units"], "n_mentioned": bd["n_mentioned"],
                         "recall": bd["recall"], "precision": bd["precision"],
                         "both_zero": int(both_zero)})

    n_q = len(deltas)
    pos = sum(1 for d in deltas if d > 1e-12)
    neg = sum(1 for d in deltas if d < -1e-12)
    stats = {
        "n_questions": n_q, "n_rows": len(rows),
        "n_both_zero": n_both_zero,
        # THE label-quality metric. See module docstring: at the un-recalibrated
        # match_thr=0.50 the reward was zero on 92% of OvertonBench responses, so
        # a high rate here is expected and is the thing recalibration must fix.
        "both_zero_rate": (n_both_zero / n_q) if n_q else float("nan"),
        "n_positive_delta": pos, "n_negative_delta": neg,
        "n_zero_delta": n_q - pos - neg,
        "mean_delta": (sum(deltas) / n_q) if n_q else float("nan"),
        "n_skipped_thin_subtree": n_thin,
    }
    return rows, stats


def write_scores(path: str, rows) -> None:
    import csv

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCORE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def check_out_path(path: str) -> str:
    """Reject output names that would collide with an OvertonBench run tag.

    delta_regressor.run_tag reads 'vN' out of the basename and prints it as the
    run in every per-run table. A file named ..._v5_scores.csv here would appear
    as run 'v5' beside the judge-labelled OvertonBench v5 -- the exact silent mix
    the label_source column exists to prevent. Also require 'scores' in the name,
    because derive_responses_path pairs the jsonl by substituting it.
    """
    import delta_regressor

    b = os.path.basename(path)
    if delta_regressor._VTAG.search(b):
        raise SystemExit(
            f"--out basename '{b}' contains a vN tag; delta_regressor would file "
            f"these reward labels under an OvertonBench run tag. Rename it.")
    if "scores" not in b:
        raise SystemExit(
            f"--out basename '{b}' must contain 'scores' -- delta_regressor pairs "
            f"the responses jsonl by substituting 'scores'->'responses'.")
    return delta_regressor.derive_responses_path(path)


def write_meta(path: str, meta: dict) -> str:
    p = os.path.splitext(path)[0] + ".meta.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return p


def report(stats: dict, gen_stats: dict) -> None:
    print("\n=== label set ===")
    print(f"  questions labelled      {stats['n_questions']}  "
          f"({stats['n_rows']} rows, 2 per question)")
    print(f"  delta > 0 / < 0 / == 0  {stats['n_positive_delta']} / "
          f"{stats['n_negative_delta']} / {stats['n_zero_delta']}")
    print(f"  mean delta              {stats['mean_delta']:+.4f}")
    if gen_stats:
        print(f"  pool={gen_stats['n_pool']}  generated={gen_stats['n_generated']}  "
              f"resumed_rows={gen_stats['n_resumed_rows']}  "
              f"skipped: no_fork={gen_stats['n_skipped_no_fork']} "
              f"thin_subtree={gen_stats['n_skipped_thin_subtree']}")
    print("\n=== LABEL QUALITY (read this before using the labels) ===")
    print(f"  both-zero questions     {stats['n_both_zero']}/{stats['n_questions']} "
          f"= {stats['both_zero_rate']:.3f}")
    print("  ^ reward scored BOTH answers 0.0: delta==0 by construction, no routing")
    print("    signal. EMITTED with both_zero=1, never dropped. coverage_reward")
    print("    currently fails its GRPO gate (within-question concordance 0.078 vs")
    print("    0.500 chance; 0.735 on the pairs it separates) -- diagnosed as")
    print("    match_thr=0.50 sitting above the p75=0.475 of the cosines it")
    print("    thresholds. Rerun with --score_only --match_thr <recalibrated>;")
    print("    this rate falling is what says the recalibration worked.")


# ---------------------------------------------------------------------------
# Self-test: stub chat + stub embedder, no endpoint, no graph, no GPU
# ---------------------------------------------------------------------------
def _stub_embed_fn():
    """Collision-free bag-of-words embedder (alignment.reward._selftest's trick).

    Each distinct word owns a dimension, so a canned answer built from repeated
    option tokens matches that option's Position at cosine ~1.0 and everything
    else at ~0. Lets the fixture PLANT an exact coverage pattern -- which is the
    only way an offline test can assert on both-zero accounting. It says nothing
    about whether match_thr is calibrated for mpnet on real prose; that blind
    spot is precisely how the gate failure went unnoticed (reward_gate_failure
    §7), so it is stated rather than papered over.
    """
    import numpy as np

    vocab: dict = {}

    def embed(texts):
        V = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", str(t).lower()):
                V[i, vocab.setdefault(w, len(vocab) % 64)] += 1.0
            nrm = np.linalg.norm(V[i])
            if nrm:
                V[i] /= nrm
        return V
    return embed


def _stub_chat(canned):
    """An injectable `chat` for retrieval.answer.answer: canned text, no HTTP.

    Keyed on whether the system prompt is the baseline instruction, so it
    exercises answer()'s real branch selection rather than being told the
    condition.
    """
    from retrieval.answer import BASELINE_INSTRUCTION

    calls = []

    def chat_fn(base_url, model, messages, **kw):
        is_base = messages[0]["content"] == BASELINE_INSTRUCTION
        calls.append(("baseline" if is_base else "injected", kw.get("max_tokens")))
        return canned["baseline"] if is_base else canned["injected"]
    chat_fn.calls = calls
    return chat_fn


def _fixture():
    """Synthetic questions with a PLANTED delta sign, plus planted both-zero rows.

    Mirrors the measured regime the router exists for: injection helps contested
    questions and hurts consensus ones. `dead` questions are answered entirely
    off-position, so the reward scores both answers 0 -- the label-quality case
    that must be counted, not dropped.

    `dilute` and `faint` sit at intermediate cosine (0.707 and 0.243) precisely
    so that match_thr has something to slice. Without them the stub emits only
    cosines of 0 or 1 and the threshold sweep passes vacuously -- which is the
    same shape of blind spot that let match_thr=0.50 ship unfitted.
    """
    from alignment.reward import Position

    def pos(opt, prev):
        return Position(option=opt, embed_text=f"{opt} {opt} {opt} {opt}",
                        prevalence=prev)

    def para(opt, k):
        return " ".join([opt] * k) + "."

    def mix(opt, k_opt, k_noise):
        """cos(unit, position) = k_opt / sqrt(k_opt^2 + k_noise^2), by construction."""
        return " ".join([opt] * k_opt + ["noise"] * k_noise) + "."

    positions = [pos("alpha", 0.4), pos("beta", 0.35), pos("gamma", 0.25)]
    all_three = "\n".join(para(o, 80) for o in ("alpha", "beta", "gamma"))
    one_deep = para("alpha", 80)
    two_deep = "\n".join(para(o, 80) for o in ("alpha", "beta"))
    junk = "\n".join(para("zebra", 80) for _ in range(2))
    dilute = "\n".join(mix("alpha", 40, 40) for _ in range(2))     # cos 0.707
    faint = "\n".join(mix("alpha", 20, 80) for _ in range(2))      # cos 0.243

    kinds = {                                  # kind -> (baseline, injected)
        "contested": (one_deep, all_three),    # delta > 0
        "consensus": (two_deep, one_deep),     # delta < 0
        "dead": (junk, junk),                  # both zero at every threshold
        "dilute": (junk, dilute),              # delta > 0 until thr > 0.707
        "faint": (junk, faint),                # both zero until thr < 0.243
    }
    qs, canned, kind_of = [], {}, {}
    for i, kind in enumerate(["contested", "consensus", "dead",
                              "contested", "consensus", "dead",
                              "contested", "consensus", "dilute", "faint"]):
        qid = 100 + i
        qs.append((qid, f"synthetic {kind} question {i} about policy and society"))
        canned[qid] = {"baseline": kinds[kind][0], "injected": kinds[kind][1]}
        kind_of[qid] = kind
    return qs, canned, positions, kind_of


def selftest(tmpdir: str) -> int:
    from alignment.reward import RewardConfig
    from retrieval.answer import answer

    import delta_regressor

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  [ok] " if cond else "  [FAIL] ") + msg)
        ok = ok and bool(cond)

    print("=== SELFTEST: stub chat + stub embedder, no endpoint/graph/GPU ===")
    qs, canned, positions, kind_of = _fixture()
    embed_fn = _stub_embed_fn()
    cfg = RewardConfig(min_depth_words=60)

    def n_kind(*kinds):
        return sum(1 for qid, _t in pool if kind_of[qid] in kinds)

    # ---- 1. the real answer() drives the injectable chat ------------------
    # scout needs a graph + hyperbolic embeddings that only exist cluster-side,
    # so the condition exercised end-to-end through retrieval.answer is the
    # baseline one; the injected side is canned at the answer_fn seam below.
    stub = _stub_chat({"baseline": "canned baseline", "injected": "canned injected"})
    a, trace = answer("does it route?", BASELINE_COND, chat_fn=stub, with_trace=True)
    check(a == "canned baseline" and trace["fork_context"] == "" and stub.calls,
          f"retrieval.answer.answer drove the stub chat ({stub.calls[0][0]} path)")

    # ---- 2. holdout guard --------------------------------------------------
    holdout_text = "synthetic contested question 0 about policy and society"
    from alignment.rollout_dataset import _norm
    pool, dropped = sample_questions(qs, {_norm(holdout_text)}, seed=0)
    check(dropped == 1 and all(q[0] != 100 for q in pool),
          f"holdout guard dropped {dropped} eval question(s); {len(pool)} remain")

    # ---- 3. generate + score, with a call-counting answer_fn ---------------
    n_calls = [0]

    def answer_fn(question, condition, q_emb):
        qid = next(q for q, t in qs if t == question)
        key = "baseline" if condition == BASELINE_COND else "injected"
        n_calls[0] += 1
        text = canned[qid][key]
        return text, {"raw": text, "think": "", "fork_context": "", "n_forks": 0}

    out = os.path.join(tmpdir, "selftest_delta_scores.csv")
    out_resp = check_out_path(out)
    for p in (out, out_resp):
        if os.path.exists(p):
            os.remove(p)

    kw = dict(n=len(pool), inject_cond="scout", out_resp=out_resp,
              embed_q_fn=lambda q: None, anchor_fn=lambda q, e: 7,
              positions_at=lambda a: positions, answer_fn=answer_fn, log_every=0)
    accepted, anchors, gstats = generate(pool, **kw)
    check(n_calls[0] == 2 * len(pool),
          f"generated {n_calls[0]} answers for {len(pool)} questions (2 each)")

    rows, stats = score(out_resp, inject_cond="scout",
                        positions_at=lambda a: positions, embed_fn=embed_fn,
                        cfg=cfg, model="stub")
    write_scores(out, rows)

    # ---- 4. both-zero rows counted AND still present -----------------------
    # at the default thr=0.50 the zero-signal set is `dead` plus `faint`
    n_zero = n_kind("dead", "faint")
    emitted_zero = len({r["question_id"] for r in rows if r["both_zero"]})
    check(stats["n_both_zero"] == n_zero and emitted_zero == n_zero,
          f"both-zero: planted {n_zero}, counted {stats['n_both_zero']}, "
          f"emitted (not dropped) {emitted_zero}, rate={stats['both_zero_rate']:.3f}")
    check(stats["n_positive_delta"] == n_kind("contested", "dilute")
          and stats["n_negative_delta"] == n_kind("consensus")
          and stats["n_zero_delta"] == n_zero,
          f"planted delta signs recovered: +{stats['n_positive_delta']} "
          f"-{stats['n_negative_delta']} 0:{stats['n_zero_delta']}")

    # ---- 5. schema round-trips into delta_regressor's loader ---------------
    check(delta_regressor.derive_responses_path(out) == out_resp,
          "responses jsonl is where delta_regressor derives it")
    dr_rows, meta = delta_regressor.build_rows([out], [], BASELINE_COND, {"scout"})
    tag = delta_regressor.run_tag(out)
    check(len(dr_rows) == stats["n_questions"] and meta[tag]["responses"] == out_resp,
          f"delta_regressor.build_rows loaded {len(dr_rows)} labels from the csv "
          f"and paired the jsonl (run tag '{tag}')")
    mine = {r["question_id"]: r["coverage"] for r in rows if r["condition"] == "scout"}
    base = {r["question_id"]: r["coverage"] for r in rows
            if r["condition"] == BASELINE_COND}
    check(all(abs(r.delta - (mine[r.qid] - base[r.qid])) < 1e-12 for r in dr_rows),
          "delta_regressor's Row.delta == reward(scout) - reward(baseline)")
    resp = delta_regressor.load_responses(out_resp)
    check(all(BASELINE_COND in resp[r.qid] and "scout" in resp[r.qid] for r in dr_rows),
          "load_responses sees both conditions per question (feature extraction ok)")

    # ---- 6. resume skips existing rows -------------------------------------
    n_calls[0] = 0
    accepted2, _, gstats2 = generate(pool, **kw)
    rows2, stats2 = score(out_resp, inject_cond="scout",
                          positions_at=lambda a: positions, embed_fn=embed_fn,
                          cfg=cfg, model="stub")
    check(n_calls[0] == 0 and accepted2 == accepted,
          f"resume: {gstats2['n_resumed_rows']} rows present, 0 regenerated, "
          f"same {len(accepted2)} questions accepted")
    check(rows2 == rows, "rescoring a resumed file reproduces the labels exactly")

    # ---- 7. match_thr is a live knob (the recalibration path) --------------
    # STRICT inequalities: `dilute` (cos 0.707) drops out above 0.707 and `faint`
    # (cos 0.243) comes back below it, so a threshold silently ignored by score()
    # would fail here rather than pass on a tie.
    hi = RewardConfig(min_depth_words=60, match_thr=0.95)
    _, s_hi = score(out_resp, inject_cond="scout", positions_at=lambda a: positions,
                    embed_fn=embed_fn, cfg=hi, model="stub")
    lo = RewardConfig(min_depth_words=60, match_thr=0.05)
    _, s_lo = score(out_resp, inject_cond="scout", positions_at=lambda a: positions,
                    embed_fn=embed_fn, cfg=lo, model="stub")
    check(s_lo["n_both_zero"] == n_kind("dead")
          and stats["n_both_zero"] == n_kind("dead", "faint")
          and s_hi["n_both_zero"] == n_kind("dead", "faint", "dilute"),
          f"both_zero_rate STRICTLY decreases as match_thr falls: "
          f"{s_lo['both_zero_rate']:.3f} (thr=0.05) < "
          f"{stats['both_zero_rate']:.3f} (thr=0.50) < "
          f"{s_hi['both_zero_rate']:.3f} (thr=0.95)")

    # ---- 8. the run-tag guard ----------------------------------------------
    try:
        check_out_path("results/delta_labels_v5_scores.csv")
        check(False, "vN output name rejected")
    except SystemExit:
        check(True, "vN output name rejected (would alias an OvertonBench run tag)")

    report(stats, gstats)
    print("\nSELFTEST: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Delta labels on graph questions for the injection router")
    ap.add_argument("--out", default="results/delta_labels_reward_scores.csv",
                    help="scores csv; the responses jsonl is derived from it the "
                         "way delta_regressor derives it")
    ap.add_argument("--n", type=int, default=500,
                    help="questions to label. 2-3 features + ridge need hundreds, "
                         "not thousands; 500 is ~1000 generations.")
    ap.add_argument("--inject_cond", default="scout",
                    help="the injected condition to compare against baseline")
    ap.add_argument("--dataset", choices=["opinionqa", "globalopinionqa"],
                    default="opinionqa")
    ap.add_argument("--embeddings", default=None,
                    help="h_all .pt on the ball (required unless --score_only)")
    ap.add_argument("--text_feat", default=None)
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42,
                    help="graph split seed AND sampling seed; must match the "
                         "embed job's seed or node ids do not line up")
    ap.add_argument("--tau", type=float, default=0.25,
                    help="scout relevance gate (on-domain opinionqa 0.25; GOQA ~0.1)")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="",
                    help="LABELS ARE MODEL-SPECIFIC -- must be the model you "
                         "intend to route for")
    ap.add_argument("--min_positions", type=int, default=2)
    # reward knobs, exposed so labels can be regenerated at a recalibrated
    # threshold without editing code (docs/reward_gate_failure.md §9)
    ap.add_argument("--match_thr", type=float, default=None,
                    help="cosine gate for 'position mentioned'. Default = "
                         "RewardConfig's 0.50, which is ABOVE the p75=0.475 of "
                         "the cosines it slices -- see the module docstring.")
    ap.add_argument("--min_depth_words", type=int, default=None)
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2",
                    help="held-out reward embedder (NOT the scout's MiniLM)")
    ap.add_argument("--score_only", action="store_true",
                    help="rescore the existing responses jsonl and exit. No "
                         "endpoint, no embeddings, no generation -- this is the "
                         "path to run after the reward is recalibrated.")
    ap.add_argument("--generate_only", action="store_true",
                    help="generate and stop. The two halves want opposite "
                         "resources -- generation is an HTTP client that must "
                         "keep its hands off the GPUs vLLM is holding, scoring "
                         "is a GPU embedder pass. Splitting them lets the job "
                         "release vLLM before scoring (see the job script).")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            return selftest(td)

    if args.score_only and args.generate_only:
        ap.error("--score_only and --generate_only are mutually exclusive")

    out_resp = check_out_path(args.out)
    if not args.score_only:
        if not args.embeddings:
            ap.error("--embeddings is required unless --score_only")
        if not args.model:
            ap.error("--model is required: labels are model-specific")

    from alignment.reward import (RewardConfig, default_embed_fn,
                                  positions_from_subtree)

    cfg_kw = {}
    if args.match_thr is not None:
        cfg_kw["match_thr"] = args.match_thr
    if args.min_depth_words is not None:
        cfg_kw["min_depth_words"] = args.min_depth_words
    cfg = RewardConfig(**cfg_kw)

    if args.dataset == "opinionqa":
        from data.loaders.opinionqa import load_opinionqa
        graph = load_opinionqa(split_seed=args.seed, leakage_safe=True)
    else:
        from data.loaders.globalopinionqa import load_globalopinionqa
        graph = load_globalopinionqa(split_seed=args.seed, leakage_safe=True)

    def positions_at(anchor):
        return positions_from_subtree(graph, anchor)

    gen_stats = {}
    if not args.score_only:
        import torch
        from pluraltree.manifolds.poincare import PoincareBall
        from retrieval.answer import CONDITIONS, answer
        from retrieval.scout import (ScoutConfig, embed_question,
                                     load_or_compute_text_feat, scout)
        from alignment.rollout_dataset import load_eval_holdout_texts

        if args.inject_cond not in CONDITIONS or args.inject_cond == BASELINE_COND:
            ap.error(f"--inject_cond must be one of "
                     f"{sorted(set(CONDITIONS) - {BASELINE_COND})}")

        h_all = torch.load(args.embeddings, map_location="cpu")
        if not isinstance(h_all, torch.Tensor):
            h_all = h_all["h_all"]
        manifold = PoincareBall(c=args.curvature)
        text_feat = load_or_compute_text_feat(graph, args.dataset, args.text_feat)
        base_cfg = CONDITIONS[args.inject_cond]
        scfg = ScoutConfig(tau=args.tau, alpha=base_cfg.alpha)

        holdout = load_eval_holdout_texts()
        pool, dropped = sample_questions(graph_questions(graph), holdout, args.seed)
        print(f"{len(pool)} graph questions in the pool "
              f"({dropped} OvertonBench eval questions dropped by the holdout "
              f"guard); labelling the first {args.n} usable ones")

        def anchor_fn(question, q_emb):
            # The scout runs twice per question: once here for the anchor the
            # reward scores against, once inside answer(). It is deterministic
            # given the same cfg and q_emb, so both see the same fork; there is
            # no seam on answer() that returns the fork, and duplicating the
            # GENERATION path to make one would be worse than a second matmul.
            forks = scout(question, graph, h_all, text_feat, manifold,
                          cfg=scfg, q_emb=q_emb)
            return forks[0].anchor if forks else None

        def answer_fn(question, condition, q_emb):
            return answer(question, condition, graph=graph, h_all=h_all,
                          text_feat=text_feat, manifold=manifold,
                          base_url=args.base_url, model=args.model,
                          q_emb=q_emb,
                          cfg=(scfg if condition != BASELINE_COND else None),
                          with_trace=True)

        _, _, gen_stats = generate(
            pool, n=args.n, inject_cond=args.inject_cond, out_resp=out_resp,
            embed_q_fn=embed_question, anchor_fn=anchor_fn,
            positions_at=positions_at, answer_fn=answer_fn,
            min_positions=args.min_positions)
        if args.generate_only:
            print(f"\ngenerated={gen_stats['n_generated']} "
                  f"resumed={gen_stats['n_resumed_rows']} "
                  f"no_fork={gen_stats['n_skipped_no_fork']} "
                  f"thin={gen_stats['n_skipped_thin_subtree']} -> {out_resp}")
            print(f"score it: python {os.path.relpath(__file__)} --score_only "
                  f"--out {args.out}")
            return 0

    if not os.path.exists(out_resp):
        print(f"no responses at {out_resp} -- nothing to score")
        return 1

    print(f"\nscoring with match_thr={cfg.match_thr} "
          f"min_depth_words={cfg.min_depth_words} embedder={args.embedder}")
    embed_fn = default_embed_fn(args.embedder)
    rows, stats = score(out_resp, inject_cond=args.inject_cond,
                        positions_at=positions_at, embed_fn=embed_fn, cfg=cfg,
                        model=args.model)
    if not rows:
        print("no (baseline, injected) pairs scored")
        return 1
    write_scores(args.out, rows)
    meta_path = write_meta(args.out, {
        "label_source": LABEL_SOURCE, "model": args.model,
        "dataset": args.dataset, "inject_cond": args.inject_cond,
        "seed": args.seed, "tau": args.tau, "n_requested": args.n,
        "embedder": args.embedder,
        "reward_config": {"match_thr": cfg.match_thr,
                          "min_depth_words": cfg.min_depth_words,
                          "weight": cfg.weight, "l_precision": cfg.l_precision,
                          "l_verbose": cfg.l_verbose,
                          "min_prevalence": cfg.min_prevalence,
                          "max_units": cfg.max_units},
        "generation": gen_stats, "labels": stats,
    })
    report(stats, gen_stats)
    print(f"\nwrote {args.out}\n      {out_resp}\n      {meta_path}")
    print(f"consume: python scripts/analysis/delta_regressor.py --scores {args.out} "
          f"--baseline_cond {BASELINE_COND} --inject_conds {args.inject_cond}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
