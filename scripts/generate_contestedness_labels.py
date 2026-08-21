"""Weak contestedness labels for alignment/probe.py, from self-consistency.

probe.py wants `--labels contestedness_labels.json`; nothing produced it. This
does, and it deliberately does NOT use graph divergence.

  graph divergence  route_signal measured corr(w_raw, help-delta) = +0.19, and
                    `relevance` = -0.26. The scout SELECTS max-Wasserstein
                    forks, so W has no variance left to predict with. Training
                    the probe on that label inherits the +0.19 ceiling by
                    construction -- the probe could at best re-learn the signal
                    that already failed.
  self-report       asking the model (route, v6) scored 0.072 vs baseline
                    0.479. Not a label either.
  self-consistency  retrieval/contestedness.py: sample K answers under the
                    COMMIT-forcing prompt and measure how far apart the stances
                    land. This is a MEASUREMENT, not an assertion, and it is
                    the one the probe docstring asks for.

What that makes the probe learn: "will I collapse to one stance on this
question" -- a property of the MODEL, not of the data. That is the point. The
graph says whether the POPULATION is divided; this says whether the MODEL acts
divided. Their DISAGREEMENT (population divided, model not) is the routing
signal we actually want, and it is unobtainable from either side alone.

Two things this script must get right:

1. SAME MODEL. The labels describe the model whose hidden states the probe
   reads. Sampling from the served 72B AWQ while probing the local 7B would
   label a different model's behaviour. Default backend is therefore `local`
   (HF weights, same directory the probe uses); `--base_url` switches to the
   endpoint only when you know they are the same weights.

2. TOPICS. probe.py holds out by topic, and OvertonBench ships no topic field,
   so we mint one: k-means over the held-out mpnet embeddings of the questions
   (pure torch, mirrors data/loaders/opinionqa.py -- no sklearn). Adjacent
   OvertonBench questions are near-paraphrases; without grouping, random k-fold
   scores a topic-memorizing probe as a win (see probe.py --selftest).

Cost: K generations x N questions, paid ONCE. Probe inference afterwards is a
single forward pass with no generation.

    python scripts/generate_contestedness_labels.py --selftest
    python scripts/generate_contestedness_labels.py \
        --model /path/to/Qwen2.5-7B-Instruct --k 8 --k_topics 8 \
        --scores overton_scores_v5.csv --out contestedness_labels.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from retrieval.contestedness import (PROBE_INSTRUCTION, ContestConfig,
                                     contestedness_score, stance_spread)

SampleFn = Callable[[str], list[str]]        # question -> K committed stances


# ---------------------------------------------------------------------------
# Topics (for probe.py's leave-one-topic-out holdout)
# ---------------------------------------------------------------------------
def _kmeans(X: torch.Tensor, k: int, seed: int, iters: int = 50) -> torch.Tensor:
    """(n,) cluster assignment; deterministic given seed. Same implementation
    as data/loaders/opinionqa.py -- no sklearn anywhere in this repo."""
    k = max(1, min(k, X.shape[0]))
    gen = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(X.shape[0], generator=gen)[:k]].clone()
    assign = torch.zeros(X.shape[0], dtype=torch.long)
    for _ in range(iters):
        assign = torch.cdist(X, C).argmin(dim=1)
        for j in range(k):
            m = assign == j
            if m.any():
                C[j] = X[m].mean(0)
    return assign


def assign_topics(questions: Sequence[str], embed_fn, k_topics: int = 8,
                  seed: int = 0) -> list[int]:
    """Topic id per question, clustered on the HELD-OUT embedder -- the same one
    that scores stance spread, and deliberately not the scout's MiniLM, so the
    grouping is independent of the retrieval the gate governs."""
    X = torch.as_tensor(np.asarray(embed_fn(list(questions)), dtype=np.float32))
    return [int(t) for t in _kmeans(X, k_topics, seed)]


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------
class LocalSampler:
    """K committed stances per question from LOCAL HF weights.

    Same weights the probe reads hidden states from -- the label is a property
    of THIS model, so it must not be measured on a different one. One batched
    generate() call per question (num_return_sequences=K)."""

    def __init__(self, model_name: str, cfg: ContestConfig,
                 device: str | None = None, dtype: str = "bfloat16",
                 seed: int = 0):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg, self.seed = cfg, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:          # Llama-family have none
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype)).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, question: str) -> list[str]:
        text = self.tok.apply_chat_template(
            [{"role": "system", "content": PROBE_INSTRUCTION},
             {"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        torch.manual_seed(self.seed)               # reproducible K-sample draw
        out = self.model.generate(
            **enc, do_sample=True, temperature=self.cfg.temperature,
            top_p=self.cfg.top_p, max_new_tokens=self.cfg.max_tokens,
            num_return_sequences=self.cfg.k,
            pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]   # strip the prompt
        return [t.strip() for t in
                self.tok.batch_decode(gen, skip_special_tokens=True)]


def endpoint_sampler(base_url: str, model: str, cfg: ContestConfig) -> SampleFn:
    """Sampling through the OpenAI-compatible endpoint. Only valid when the
    served weights ARE the weights the probe will read."""
    from retrieval.contestedness import sample_stances
    return lambda q: sample_stances(q, base_url, model, cfg)


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------
def build_labels(questions: Sequence[tuple[int, str]], sample_fn: SampleFn,
                 embed_fn, cfg: ContestConfig, k_topics: int = 8,
                 seed: int = 0, verbose: bool = True) -> list[dict]:
    """[(qid, question)] -> label rows probe.py can read directly.

    Row schema: question_id, question, topic, score, mean_dist, vendi, k,
    stances. `score` is contestedness_score in [0,1]; probe.py median-splits it
    (a weak label -- the threshold is a knob, the ranking is the signal)."""
    topics = assign_topics([q for _, q in questions], embed_fn, k_topics, seed)
    rows = []
    for (qid, q), topic in zip(questions, topics):
        stances = sample_fn(q)
        spread = stance_spread(stances, embed_fn)
        score = contestedness_score(spread)
        rows.append({"question_id": int(qid), "question": q, "topic": int(topic),
                     "score": score, "mean_dist": spread["mean_dist"],
                     "vendi": spread["vendi"], "k": spread["k"],
                     "stances": stances})
        if verbose:
            print(f"  Q{qid} topic={topic} score={score:.3f} "
                  f"(dist={spread['mean_dist']:.3f} vendi={spread['vendi']:.2f}) "
                  f"{q[:60]}")
    return rows


def summarize(rows: list[dict]) -> None:
    s = sorted(r["score"] for r in rows)
    if not s:
        print("no rows"); return
    med = s[len(s) // 2]
    from collections import Counter
    sizes = Counter(r["topic"] for r in rows)
    print(f"\n{len(rows)} rows | score min {s[0]:.3f} median {med:.3f} "
          f"max {s[-1]:.3f} | {len(sizes)} topics, sizes "
          f"{sorted(sizes.values(), reverse=True)}")
    if s[-1] - s[0] < 0.05:
        # A flat score column median-splits into noise and the probe trains on
        # coin flips. Usually K too small or the commit prompt not applied.
        print("WARNING: score column is nearly constant -- the probe will train "
              "on noise. Raise --k / --temperature, and check the commit prompt "
              "actually reached the model.")
    if min(sizes.values()) < 3:
        print("WARNING: a topic has <3 questions -- leave-one-topic-out folds "
              "that small give an unstable per-fold AUC (read oof_auc instead).")


# ---------------------------------------------------------------------------
def _fake_embed_factory():
    """Deterministic bag-of-words embedder: shared words -> high cosine. Enough
    to exercise the spread math without sentence-transformers."""
    import re
    vocab: dict[str, int] = {}

    def embed(texts):
        V = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", str(t).lower()):
                V[i, vocab.setdefault(w, len(vocab) * 7 % 64)] += 1.0
            n = np.linalg.norm(V[i])
            if n:
                V[i] /= n
        return V
    return embed


def _selftest() -> None:
    """Planted structure: consensus questions get identical stances, contested
    ones get scattered stances. The generator must separate them -- and a
    SHUFFLED control, where the stance sets are reassigned to the wrong
    questions, must destroy that separation. Without the control, a generator
    that emitted anything monotone in question index would pass."""
    embed = _fake_embed_factory()
    cfg = ContestConfig(k=5)

    consensus_stances = ["water is wet indeed", "water is wet indeed",
                         "water is wet indeed", "water is wet indeed",
                         "water is wet indeed"]
    contested_stances = ["strongly support alpha", "firmly oppose beta gamma",
                         "delta unsure entirely", "epsilon different framing",
                         "zeta orthogonal claim"]
    # Topic-correlated wording, so the k-means grouping has something to find.
    qs = [(i, f"tax policy question {i} about revenue brackets") for i in range(6)]
    qs += [(6 + i, f"immigration question {i} about borders and visas")
           for i in range(6)]
    truth = {qid: (qid % 2 == 1) for qid, _ in qs}         # odd = contested
    plan = {qid: (contested_stances if c else consensus_stances)
            for qid, c in truth.items()}

    rows = build_labels(qs, lambda q: plan[_qid_of(q, qs)], embed, cfg,
                        k_topics=2, seed=0, verbose=False)
    by = {r["question_id"]: r for r in rows}
    con = [by[q]["score"] for q, c in truth.items() if not c]
    ctd = [by[q]["score"] for q, c in truth.items() if c]
    assert max(con) < 0.05, con
    assert min(ctd) - max(con) > 0.30, (ctd, con)

    # CONTROL: same stance sets, wrong questions -> the label must stop tracking
    # the planted contestedness. The reassignment is orthogonal BY
    # CONSTRUCTION (half the contested sets land on truly-contested questions,
    # half on consensus ones), so the expected corr is exactly 0 -- a rotation
    # would have given -1.0, which is a perfectly informative label, not a
    # control.
    shuffled = {qid: (contested_stances if (qid // 2) % 2 == 0
                      else consensus_stances) for qid, _ in qs}
    rows_sh = build_labels(qs, lambda q: shuffled[_qid_of(q, qs)], embed, cfg,
                           k_topics=2, seed=0, verbose=False)
    by_sh = {r["question_id"]: r for r in rows_sh}
    real = _corr([by[q]["score"] for q, _ in qs], [float(truth[q]) for q, _ in qs])
    ctrl = _corr([by_sh[q]["score"] for q, _ in qs], [float(truth[q]) for q, _ in qs])
    assert real > 0.95, real
    assert abs(ctrl) < 0.40, ctrl

    # Topics: deterministic, and they partition the questions.
    t1 = assign_topics([q for _, q in qs], embed, 2, seed=0)
    t2 = assign_topics([q for _, q in qs], embed, 2, seed=0)
    assert t1 == t2, (t1, t2)
    assert len(set(t1)) == 2, t1
    assert all(isinstance(t, int) for t in t1)

    # Schema probe.py actually reads, and JSON round-trips.
    need = {"question_id", "question", "topic", "score"}
    assert all(need <= set(r) for r in rows)
    assert json.loads(json.dumps(rows))[0]["score"] == rows[0]["score"]

    print("label self-test OK")
    print(f"  planted   consensus max {max(con):.3f} < contested min "
          f"{min(ctd):.3f}   corr(score, truth) = {real:+.3f}")
    print(f"  shuffled  corr(score, truth) = {ctrl:+.3f}  <- control: stance "
          "sets reattached to the wrong questions must kill the signal")
    print(f"  topics    {len(set(t1))} clusters, deterministic, sizes "
          f"{[t1.count(t) for t in sorted(set(t1))]}")


def _qid_of(question: str, qs) -> int:
    return next(qid for qid, q in qs if q == question)


def _corr(xs, ys) -> float:
    import statistics as st
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser(
        description="Self-consistency contestedness labels for alignment/probe.py")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="LOCAL directory of the SAME weights the probe reads")
    ap.add_argument("--base_url", default=None,
                    help="use the OpenAI-compatible endpoint instead of local "
                         "weights -- only if it serves the same model")
    ap.add_argument("--questions", default=None,
                    help="text file, one per line; omit to use OvertonBench")
    ap.add_argument("--k", type=int, default=8,
                    help="samples per question; spread is unstable below ~5")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="must be >0 or every sample is identical and every "
                         "score is 0")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--k_topics", type=int, default=8,
                    help="leave-one-topic-out folds for probe.py")
    ap.add_argument("--embedder",
                    default="sentence-transformers/all-mpnet-base-v2",
                    help="held-out embedder (local dir preferred); NOT the "
                         "scout's MiniLM, so the label is independent of the "
                         "retrieval it gates")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_questions", type=int, default=0)
    ap.add_argument("--scores", default=None,
                    help="overton_scores_vN.csv -- also score the LABEL itself "
                         "as a gate (tells you the ceiling the probe is "
                         "distilling before you train it)")
    ap.add_argument("--inject_cond", default="scout")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--out", default="contestedness_labels.json")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    cfg = ContestConfig(k=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, threshold=args.threshold)
    if args.questions:
        qs = [(i, l.strip()) for i, l in
              enumerate(open(args.questions, encoding="utf-8")) if l.strip()]
    else:
        from evaluation.overton.eval_overtonbench import load_questions
        qs = load_questions()
    if args.max_questions:
        qs = qs[: args.max_questions]

    from retrieval.contestedness import default_embed_fn
    embed_fn = default_embed_fn(args.embedder)

    if args.base_url:
        print(f"sampling via endpoint {args.base_url} (model {args.model})")
        sample_fn = endpoint_sampler(args.base_url, args.model, cfg)
    else:
        print(f"sampling locally from {args.model}")
        sample_fn = LocalSampler(args.model, cfg, dtype=args.dtype, seed=args.seed)

    print(f"{len(qs)} questions x K={cfg.k} samples, {args.k_topics} topics")
    rows = build_labels(qs, sample_fn, embed_fn, cfg, k_topics=args.k_topics,
                        seed=args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    summarize(rows)
    print(f"wrote {args.out}")

    if args.scores and os.path.exists(args.scores):
        # What the LABEL alone is worth as a gate. The probe distils this, so
        # this is the number it is trying to reach with one forward pass.
        from retrieval.contestedness import evaluate_gate
        evaluate_gate(rows, args.scores, cfg, inject_cond=args.inject_cond)


if __name__ == "__main__":
    main()
