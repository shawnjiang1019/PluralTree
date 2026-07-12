"""Output-diversity panel for INFINITY-CHAT generations (Artificial Hivemind).

Measures how much a pool of N responses to one open-ended query spreads across
distinct modes. Mode collapse -> pool concentrates on one mode -> high
self-similarity / few effective modes. We compare per-condition (baseline vs
scout vs div_only); the baseline-vs-condition **delta** is the claim, not the
absolute value. See docs/hivemind_diversity_eval.txt.

Design decisions baked in:
- **Evaluation independence.** The scout retrieves by MiniLM cosine; scoring
  diversity with MiniLM too would share the representation between retrieval and
  measurement. The semantic axis therefore uses a *held-out* embedder
  (``--eval_model``, default BAAI/bge-large-en-v1.5), and we also report
  embedding-free lexical metrics as a cross-check (if the two axes agree, a gain
  is real, not an encoder artifact).
- **Multi-axis.** semantic (cosine + Vendi effective-modes), lexical (distinct-n,
  self-BLEU), so no single threshold/encoder decides the result.
- **Quality guardrail.** A degeneracy filter drops empty/truncated/token-degenerate
  samples before scoring (``frac_dropped`` kept per pool) — diversity is trivially
  maximized by garbage. Judge-based quality scoring is a later pass.

Usage:
    python -m evaluation.hivemind.diversity_metrics hivemind_gen.jsonl \
        --out hivemind_diversity.csv --eval_model BAAI/bge-large-en-v1.5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_WORD = re.compile(r"\w+")

# Metric direction: +1 = higher is more diverse, -1 = lower is more diverse.
# Drives the paired win/loss counting so every axis reads "condition improved?".
METRIC_DIR = {
    "mean_cos": -1, "pct_pairs_gt80": -1, "pct_pairs_gt70": -1,
    "vendi": +1, "distinct_2": +1, "distinct_3": +1, "inv_self_bleu": +1,
}
_PANEL_KEYS = list(METRIC_DIR)


# ---------------------------------------------------------------------------
# Lexical axis (no embedder -> zero circularity with the scout)
# ---------------------------------------------------------------------------
def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(responses: list[str], n: int) -> float:
    """Unique n-grams / total n-grams pooled over the responses (0..1).

    High = the pool uses varied phrasing; low = the same n-grams recur across
    samples (surface-level mode collapse). Reference-free.
    """
    total, uniq = 0, set()
    for r in responses:
        grams = _ngrams(_tokens(r), n)
        total += len(grams)
        uniq.update(grams)
    return len(uniq) / total if total else float("nan")


def _bleu(hyp: list[str], refs: list[list[str]], max_n: int = 4) -> float:
    """Sentence BLEU of ``hyp`` against reference token lists (smoothed, clipped)."""
    if not hyp:
        return 0.0
    weight = 1.0 / max_n
    log_p = 0.0
    for n in range(1, max_n + 1):
        hyp_ng = Counter(_ngrams(hyp, n))
        total = sum(hyp_ng.values())
        if total == 0:
            log_p += weight * math.log(1e-9)          # no n-grams of this order
            continue
        max_ref: Counter = Counter()
        for ref in refs:
            for g, c in Counter(_ngrams(ref, n)).items():
                if c > max_ref[g]:
                    max_ref[g] = c
        clipped = sum(min(c, max_ref[g]) for g, c in hyp_ng.items())
        log_p += weight * math.log((clipped / total) or 1e-9)
    c = len(hyp)
    r = min((len(ref) for ref in refs), key=lambda rl: (abs(rl - c), rl))
    bp = 1.0 if c > r else math.exp(1 - r / c) if c > 0 else 0.0
    return bp * math.exp(log_p)


def self_bleu(responses: list[str], cap: int = 30) -> float:
    """Mean BLEU of each response vs the rest (Zhu et al. 2018).

    High self-BLEU = the pool repeats itself. Pool is capped at ``cap`` samples
    to bound the O(N^2) cost. Returned raw (higher = *less* diverse); the panel
    reports ``inv_self_bleu = 1 - self_bleu`` so all axes read higher = better.
    """
    toks = [_tokens(r) for r in responses[:cap] if r.strip()]
    if len(toks) < 2:
        return float("nan")
    scores = [_bleu(toks[i], toks[:i] + toks[i + 1:]) for i in range(len(toks))]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Semantic axis (held-out embedder)
# ---------------------------------------------------------------------------
def _pairwise(embs):
    """Upper-triangle cosine values of a (unit-normalized) pool, or None if <2."""
    import numpy as np

    e = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    sim = e @ e.T
    iu = np.triu_indices(len(e), k=1)
    if iu[0].size == 0:
        return None, sim
    return sim[iu], sim


def vendi_score(sim) -> float:
    """Effective number of distinct modes = exp(entropy of the cosine-Gram spectrum).

    K = sim / N is PSD with unit trace; its eigenvalues are a distribution whose
    Shannon entropy, exponentiated, is the Vendi Score. 1.0 = full collapse (one
    mode), N = all samples distinct. Threshold-free — the headline number.
    """
    import numpy as np

    n = len(sim)
    if n < 2:
        return float("nan")
    w = np.linalg.eigvalsh(sim / n)
    w = np.clip(w, 0.0, None)
    w = w[w > 1e-12]
    if w.size == 0:
        return float("nan")
    w = w / w.sum()
    return float(np.exp(-(w * np.log(w)).sum()))


# ---------------------------------------------------------------------------
# Quality guardrail
# ---------------------------------------------------------------------------
def degeneracy_filter(responses: list[str], *, min_chars: int = 3,
                      min_ttr: float = 0.1, ttr_min_tokens: int = 12
                      ) -> tuple[list[str], float]:
    """Drop empty / near-empty / token-degenerate samples; return (kept, frac_dropped).

    Token-degenerate = a long response whose type/token ratio is tiny (a single
    token repeated, the classic truncation/loop artifact that confounded
    OvertonBench). ``frac_dropped`` is carried per pool so a later quality stage
    can attach to the same rows.
    """
    kept = []
    for r in responses:
        s = r.strip()
        if len(s) < min_chars:
            continue
        tok = _tokens(s)
        if len(tok) >= ttr_min_tokens and len(set(tok)) / len(tok) < min_ttr:
            continue
        kept.append(r)
    dropped = len(responses) - len(kept)
    return kept, (dropped / len(responses) if responses else float("nan"))


def pool_panel(responses: list[str], embedder) -> dict:
    """All three axes for one pool of responses (after degeneracy filtering)."""
    kept, frac_dropped = degeneracy_filter(responses)
    out = {k: float("nan") for k in _PANEL_KEYS}
    out.update(n=len(kept), frac_dropped=frac_dropped)
    if len(kept) < 2:
        return out
    embs = embedder.encode(kept, convert_to_numpy=True, batch_size=64,
                           normalize_embeddings=True)
    pair, sim = _pairwise(embs)
    if pair is not None:
        out["mean_cos"] = float(pair.mean())
        out["pct_pairs_gt80"] = float((pair > 0.8).mean())
        out["pct_pairs_gt70"] = float((pair > 0.7).mean())
        out["vendi"] = vendi_score(sim)
    out["distinct_2"] = distinct_n(kept, 2)
    out["distinct_3"] = distinct_n(kept, 3)
    sb = self_bleu(kept)
    out["inv_self_bleu"] = (1.0 - sb) if sb == sb else float("nan")
    return out


# ---------------------------------------------------------------------------
# Paired significance (baseline vs each condition, per query)
# ---------------------------------------------------------------------------
def _sign_test_p(wins: int, losses: int) -> float:
    """Two-sided sign-test p-value under Binom(wins+losses, 0.5)."""
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def paired_sign_test(by_cond: dict[str, dict[int, dict]], baseline: str = "baseline"
                     ) -> dict[str, dict[str, dict]]:
    """For each condition vs baseline and each metric: wins/losses/ties + p + mean delta.

    A "win" means the condition improved diversity in that metric's natural
    direction (``METRIC_DIR``), counted over queries present in both pools.
    """
    import numpy as np

    out: dict[str, dict[str, dict]] = {}
    base = by_cond.get(baseline, {})
    for cond, per_q in by_cond.items():
        if cond == baseline:
            continue
        shared = sorted(set(base) & set(per_q))
        out[cond] = {}
        for m in _PANEL_KEYS:
            d = METRIC_DIR[m]
            w = l = t = 0
            deltas = []
            for q in shared:
                bv, cv = base[q].get(m), per_q[q].get(m)
                if bv is None or cv is None or bv != bv or cv != cv:
                    continue
                deltas.append(d * (cv - bv))             # +delta = improvement
                diff = d * (cv - bv)
                if diff > 1e-9:
                    w += 1
                elif diff < -1e-9:
                    l += 1
                else:
                    t += 1
            out[cond][m] = {"wins": w, "losses": l, "ties": t,
                            "p": _sign_test_p(w, l),
                            "mean_delta": float(np.mean(deltas)) if deltas else float("nan")}
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="INFINITY-CHAT diversity panel")
    ap.add_argument("infile", help="JSONL from generate_hivemind.py")
    ap.add_argument("--out", default="hivemind_diversity.csv")
    ap.add_argument("--eval_model", default="BAAI/bge-large-en-v1.5",
                    help="held-out embedder for the semantic axis (NOT the scout's MiniLM)")
    ap.add_argument("--min_samples", type=int, default=2,
                    help="skip (query, condition) pools with fewer kept samples")
    args = ap.parse_args()

    import numpy as np
    from sentence_transformers import SentenceTransformer

    # pool[(condition, query_id)] = [responses...]
    pool: dict[tuple[str, int], list[str]] = defaultdict(list)
    cat_of: dict[tuple[str, int], str] = {}
    with open(args.infile, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("sample_idx", 0) < 0:               # dry_run prompt row
                continue
            key = (r["condition"], r["query_id"])
            pool[key].append(r["response"])
            cat_of[key] = r.get("category", "uncategorized")

    print(f"loading eval embedder: {args.eval_model}")
    enc = SentenceTransformer(args.eval_model, device="cpu")

    rows = []
    # by_cond[condition][query_id] = panel dict  (for paired stats)
    by_cond: dict[str, dict[int, dict]] = defaultdict(dict)
    for (cond, qid), responses in pool.items():
        panel = pool_panel(responses, enc)
        if panel["n"] < args.min_samples:
            continue
        row = {"condition": cond, "query_id": qid,
               "category": cat_of[(cond, qid)], **panel}
        rows.append(row)
        by_cond[cond][qid] = panel

    fields = ["condition", "query_id", "category", "n", "frac_dropped", *_PANEL_KEYS]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- per-condition aggregate (mean over query pools) ----
    print(f"\n{len(rows)} pools -> {args.out}\n")
    agg_cols = ["n", "frac_dropped", "mean_cos", "vendi", "distinct_2", "inv_self_bleu"]
    print(f"{'condition':<12}{'pools':>7}" + "".join(f"{c:>14}" for c in agg_cols))
    for cond in sorted(by_cond):
        panels = list(by_cond[cond].values())
        cells = []
        for c in agg_cols:
            cells.append(f"{np.nanmean([p[c] for p in panels]):>14.3f}")
        print(f"{cond:<12}{len(panels):>7}" + "".join(cells))
    print("  (mean_cos: lower=more diverse; vendi/distinct/inv_self_bleu: higher=more diverse)")

    # ---- per-category x condition (mean Vendi = effective #modes) ----
    cats = sorted({r["category"] for r in rows})
    conds = sorted(by_cond)
    if len(cats) > 1:
        print("\nvendi (effective #modes) by category x condition:")
        print(f"{'category':<24}" + "".join(f"{c:>12}" for c in conds))
        for cat in cats:
            cells = []
            for c in conds:
                vals = [r["vendi"] for r in rows
                        if r["category"] == cat and r["condition"] == c]
                cells.append(f"{np.nanmean(vals):>12.3f}" if vals else f"{'-':>12}")
            print(f"{cat[:24]:<24}" + "".join(cells))

    # ---- paired significance vs baseline ----
    if "baseline" in by_cond and len(conds) > 1:
        stats = paired_sign_test(by_cond)
        print("\npaired vs baseline (win = condition more diverse; p = two-sided sign test):")
        for cond in sorted(stats):
            print(f"  [{cond}]")
            for m in _PANEL_KEYS:
                s = stats[cond][m]
                print(f"    {m:<16} w/l/t={s['wins']:>3}/{s['losses']:>3}/{s['ties']:>3}"
                      f"  mean_delta={s['mean_delta']:+.4f}  p={s['p']:.4f}")


if __name__ == "__main__":
    main()
