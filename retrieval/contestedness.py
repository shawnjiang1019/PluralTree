"""Self-consistency contestedness: does the MODEL disagree with itself?

Adaptation 1 of 3 (see docs/adaptive_injection.md). Two routing attempts failed:

  route_signal  no GRAPH signal separates helped-from-hurt injection; graph
                divergence measures demographic subgroup splits, which correlates
                only ~+0.20 with human contestedness.
  route (v6)    ASKING the model whether the question is contested scored 0.072.
                The 2026 adaptive-RAG literature reaches the same conclusion:
                a self-reported retrieval decision is itself hallucination-prone
                and does not reliably indicate what the model actually needs.

So neither the graph nor the model's self-report works. This module MEASURES
contestedness instead: sample K answers under a COMMIT-forcing prompt and see
whether the model lands on the same stance every time. Wide spread = the model
itself is not settled = genuinely contested.

Why the commit prompt matters: sampled under a pluralism prompt the model
enumerates every position on every sample, so spread is LOW on contested
questions -- backwards. Forcing one stance per sample makes spread mean what we
want. This is the standard self-consistency uncertainty signal, applied to
POSITIONS rather than facts (ordinary triggers ask "do I know this?"; the
pluralistic question is "do people disagree?").

Measured with a held-out embedder (mpnet), NOT the scout's MiniLM, so the signal
is independent of the retrieval it gates.

    python -m retrieval.contestedness --questions q.txt \
        --base_url http://localhost:8000/v1 --model Qwen/... --out contested.json
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from retrieval.answer import chat

EmbedFn = Callable[[Sequence[str]], np.ndarray]      # texts -> (n, d) unit-norm

# Force ONE stance per sample. Without this the model hedges identically every
# time and the spread signal vanishes (see module docstring).
PROBE_INSTRUCTION = (
    "Answer the question directly and take a clear position. State the single "
    "view you think is most defensible in two or three sentences. Do not "
    "enumerate multiple positions, do not hedge, and do not describe the debate "
    "-- commit to one answer."
)


@dataclass
class ContestConfig:
    k: int = 5                    # samples per question
    temperature: float = 1.0      # need real spread; 0 would be degenerate
    top_p: float = 0.95
    max_tokens: int = 256
    threshold: float = 0.35       # inject iff score > threshold (tune per model)


# ---------------------------------------------------------------------------
# Spread metrics (pure)
# ---------------------------------------------------------------------------
def _vendi(sim: np.ndarray) -> float:
    """Effective number of distinct stances: exp(entropy(eigvals(S/n)))."""
    n = sim.shape[0]
    if n < 2:
        return float(n)
    w = np.linalg.eigvalsh(sim / n)
    w = w[w > 1e-10]
    return float(np.exp(-(w * np.log(w)).sum()))


def stance_spread(stances: Sequence[str], embed_fn: EmbedFn) -> dict:
    """How much do the K sampled stances disagree?

    Returns {mean_dist, vendi, n_effective, k}: mean_dist in [0, 1] is
    1 - mean pairwise cosine (0 = identical stance every sample, higher =
    the model keeps changing its mind); vendi is the effective number of
    distinct stances (1 = one mode, k = all different).
    """
    texts = [s for s in stances if s and s.strip()]
    k = len(texts)
    if k < 2:
        return {"mean_dist": 0.0, "vendi": float(k), "n_effective": float(k), "k": k}
    U = np.asarray(embed_fn(texts), dtype=float)
    S = np.clip(U @ U.T, -1.0, 1.0)
    iu = np.triu_indices(k, 1)
    mean_dist = float(1.0 - S[iu].mean())
    # Pass the raw Gram matrix: it is PSD with unit diagonal by construction
    # (rows of U are unit vectors), which is exactly the Vendi kernel requirement
    # (arXiv:2210.02410). Clipping S to [0,1] first would DESTROY positive
    # semidefiniteness -- and would do so precisely when cosines go negative,
    # i.e. when the stances are genuinely far apart, the case we most want to
    # measure. Negative eigenvalues from float error are handled inside _vendi.
    v = _vendi(S)
    return {"mean_dist": mean_dist, "vendi": v, "n_effective": v, "k": k}


def contestedness_score(spread: dict) -> float:
    """Single [0, 1] contestedness score from a spread dict.

    Combines the two views: mean_dist (how far apart the stances are) and the
    normalized effective mode count (how MANY distinct stances there are). Both
    matter -- two far-apart modes and five near-duplicates are different things.
    """
    k = max(2, spread.get("k", 2))
    mode_frac = (spread.get("vendi", 1.0) - 1.0) / (k - 1)      # 0 = one mode
    return float(np.clip(0.5 * spread["mean_dist"] + 0.5 * mode_frac, 0.0, 1.0))


def should_inject(score: float, cfg: ContestConfig = ContestConfig()) -> bool:
    """The gate: inject retrieved perspectives only on contested questions."""
    return score > cfg.threshold


# ---------------------------------------------------------------------------
# Sampling (needs an endpoint)
# ---------------------------------------------------------------------------
def sample_stances(question: str, base_url: str, model: str,
                   cfg: ContestConfig = ContestConfig()) -> list[str]:
    """K independently sampled committed answers to the question."""
    msgs = [{"role": "system", "content": PROBE_INSTRUCTION},
            {"role": "user", "content": question}]
    return [chat(base_url, model, msgs, temperature=cfg.temperature,
                 top_p=cfg.top_p, max_tokens=cfg.max_tokens) for _ in range(cfg.k)]


def measure(question: str, base_url: str, model: str, embed_fn: EmbedFn,
            cfg: ContestConfig = ContestConfig()) -> dict:
    """Full signal for one question: stances -> spread -> score -> gate."""
    stances = sample_stances(question, base_url, model, cfg)
    spread = stance_spread(stances, embed_fn)
    score = contestedness_score(spread)
    return {"question": question, "score": score, "inject": should_inject(score, cfg),
            "stances": stances, **spread}


# ---------------------------------------------------------------------------
# Post-hoc gate evaluation (free: reuses an existing scores CSV)
# ---------------------------------------------------------------------------
def evaluate_gate(rows: list[dict], scores_csv: str, cfg: ContestConfig,
                  inject_cond: str = "scout", base_cond: str = "baseline") -> None:
    """What WOULD the gate have scored on a finished run?

    We already have per-question coverage for baseline and scout, so a gate that
    picks one or the other per question can be scored without generating
    anything -- the same post-hoc design route_signal.py used for the graph
    signals (which all failed). Reports the gate against always-baseline,
    always-inject, and the oracle ceiling.
    """
    import csv
    from collections import defaultdict

    cov = defaultdict(dict)
    for r in csv.DictReader(open(scores_csv, encoding="utf-8")):
        cov[int(r["question_id"])][r["condition"]] = float(r["coverage"])

    paired = [(r, cov[r["question_id"]]) for r in rows
              if r.get("question_id") in cov
              and base_cond in cov[r["question_id"]]
              and inject_cond in cov[r["question_id"]]]
    if not paired:
        print(f"\nno overlap with {scores_csv} — skipping gate evaluation")
        return
    n = len(paired)
    base = [c[base_cond] for _, c in paired]
    inj = [c[inject_cond] for _, c in paired]
    sig = [r["score"] for r, _ in paired]
    oracle = sum(max(b, i) for b, i in zip(base, inj)) / n

    gated = sum(i if s > cfg.threshold else b
                for s, b, i in zip(sig, base, inj)) / n
    n_inj = sum(s > cfg.threshold for s in sig)
    delta = [i - b for b, i in zip(base, inj)]
    c = _corr(sig, delta)
    helped = [s for s, d in zip(sig, delta) if d > 0.01]
    hurt = [s for s, d in zip(sig, delta) if d < -0.01]

    print(f"\n=== gate evaluation vs {scores_csv} (n={n}) ===")
    print(f"  always-{base_cond:<9} {sum(base)/n:.4f}")
    print(f"  always-{inject_cond:<9} {sum(inj)/n:.4f}")
    print(f"  contestedness gate  {gated:.4f}   (injects {n_inj}/{n} at "
          f"threshold {cfg.threshold})")
    print(f"  oracle              {oracle:.4f}")
    print(f"\n  corr(score, delta) = {c:+.3f}")
    print(f"  score on helped questions {sum(helped)/len(helped):.3f} (n={len(helped)})"
          if helped else "  no helped questions")
    print(f"  score on hurt   questions {sum(hurt)/len(hurt):.3f} (n={len(hurt)})"
          if hurt else "  no hurt questions")
    print("  ^ the signal is only usable if helped and hurt clearly separate;\n"
          "    graph signals did not (route_signal: corr +0.17..-0.26).")

    best, best_t = sum(base) / n, None
    for t in sorted(set(sig)):
        g = sum(i if s > t else b for s, b, i in zip(sig, base, inj)) / n
        if g > best:
            best, best_t = g, t
    print(f"  best threshold {best_t if best_t is None else f'{best_t:.3f}'} -> "
          f"{best:.4f}  (overfit ceiling for this signal)")


def _corr(xs, ys) -> float:
    import statistics as _st
    if len(xs) < 3:
        return float("nan")
    mx, my = _st.mean(xs), _st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


def default_embed_fn(model_name: str = "sentence-transformers/all-mpnet-base-v2") -> EmbedFn:
    """Held-out embedder -- deliberately NOT the scout's MiniLM, so the gate is
    independent of the retrieval it gates."""
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model_name)

    def _embed(texts: Sequence[str]) -> np.ndarray:
        return enc.encode(list(texts), normalize_embeddings=True,
                          show_progress_bar=False, batch_size=32)
    return _embed


# ---------------------------------------------------------------------------
def _selftest() -> None:
    """Contested (stances disagree) must score above consensus (all agree)."""
    import re
    rng = np.random.default_rng(0)
    vocab: dict[str, int] = {}

    def fake_embed(texts):
        V = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                V[i, vocab.setdefault(w, int(rng.integers(0, 64)))] += 1.0
            n = np.linalg.norm(V[i])
            if n:
                V[i] /= n
        return V

    consensus = ["water is wet indeed"] * 5
    contested = ["strongly support alpha", "firmly oppose beta gamma",
                 "delta unsure entirely", "strongly support alpha",
                 "epsilon different framing"]
    s_con = contestedness_score(stance_spread(consensus, fake_embed))
    s_ctd = contestedness_score(stance_spread(contested, fake_embed))
    assert s_con < 0.05, s_con
    assert s_ctd > s_con + 0.3, (s_ctd, s_con)
    assert not should_inject(s_con) and should_inject(s_ctd)
    print(f"contestedness self-test OK: consensus {s_con:.3f} < contested {s_ctd:.3f}")


def _main():
    import argparse
    import json
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ap = argparse.ArgumentParser(description="Self-consistency contestedness")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--questions", default=None,
                    help="text file, one question per line; omit to use OvertonBench")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--max_questions", type=int, default=0)
    ap.add_argument("--out", default="contestedness.json")
    ap.add_argument("--scores", default=None,
                    help="overton_scores_vN.csv — score the gate post-hoc "
                         "against baseline/scout/oracle (no generation needed)")
    ap.add_argument("--inject_cond", default="scout")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    cfg = ContestConfig(k=args.k, temperature=args.temperature,
                        threshold=args.threshold)
    if args.questions:
        qs = [(i, l.strip()) for i, l in enumerate(open(args.questions, encoding="utf-8"))
              if l.strip()]
    else:
        from evaluation.overton.eval_overtonbench import load_questions
        qs = load_questions()
    if args.max_questions:
        qs = qs[: args.max_questions]

    embed_fn = default_embed_fn()
    rows = []
    for qid, q in qs:
        r = measure(q, args.base_url, args.model, embed_fn, cfg)
        r["question_id"] = qid
        rows.append(r)
        print(f"  Q{qid} score={r['score']:.3f} inject={r['inject']} "
              f"(dist={r['mean_dist']:.3f} vendi={r['vendi']:.2f}) {q[:50]}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    inj = sum(r["inject"] for r in rows)
    print(f"\nwrote {args.out}: {len(rows)} questions, "
          f"{inj} would inject ({inj/max(1,len(rows)):.0%})")
    if args.scores:
        evaluate_gate(rows, args.scores, cfg, inject_cond=args.inject_cond)


if __name__ == "__main__":
    _main()
