"""Does injection actually collapse the answer onto the two injected poles?

docs/framing_hurts.png (left panel) ASSERTS a mechanism: the scout injects a
max-W branch pair (Perspective A / B), the model mirrors that binary, and the
middle positions it would otherwise have produced get crowded out. The measured
right panel (injection hurts most where baseline was already broad) is CONSISTENT
with that story but does not establish it -- a ceiling effect would look the same.

This script tests the mechanism directly on the stored v5 traces. Three
falsifiable predictions:

  P1 attraction  scout responses sit CLOSER to the injected pole texts than
                 baseline responses do            -> pole_attraction > 0
  P2 causal link the more a response moves onto the poles, the more coverage it
                 loses                            -> corr(attraction, delta) < 0
  P3 collapse    scout responses contain FEWER distinct positions than baseline,
                 worst where baseline was broad   -> d_modes < 0, worst at high base

P3 MUST be length-controlled: injected answers are ~5x longer than baseline
(330 vs 69 words), and Vendi grows mechanically with unit count -- raw Vendi
shows a spurious +1.27 "scout is more diverse". We therefore report Vendi at
MATCHED unit count (subsample both to k=min(n_base,n_scout)) plus mean-pairwise
cosine, which does not scale with length. Both then show the real effect.

Measurement embedder is all-mpnet-base-v2, deliberately NOT the scout's MiniLM
(retrieval optimizes MiniLM cosine; measuring with it would be circular).

CPU-only, ~1-2 min. Needs overton_responses_v5.jsonl + overton_scores_v5.csv.

Usage:
    python scripts/analysis/measure_pole_collapse.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from collections import defaultdict

import numpy as np

BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
BOLD = re.compile(r"\*+")


def _corr(xs, ys):
    if len(xs) < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


def parse_poles(fork_context: str) -> list[str]:
    """Injected pole texts: the 'Perspective A/B (...)' heads + their driver lines.

    fork_context format (retrieval/answer.py:fork_context):
        [fork k] at '<anchor>' (divergence=..., relevance=...):
          Perspective A (<subgroup> answered: "opt" 15%, ...):
          Perspective B (<subgroup> answered: ...):
            A: <subgroup> re "<survey q>": "opt" 40%, ...
            B: ...
    """
    poles = []
    for line in fork_context.splitlines():
        s = line.strip()
        m = re.match(r"^Perspective [AB] \((.*)\):?$", s)
        if m:
            poles.append(m.group(1))
            continue
        m = re.match(r"^[AB]:\s*(.+)$", s)
        if m:
            poles.append(m.group(1))
    return [p for p in poles if len(p) > 15]


def split_units(text: str) -> list[str]:
    """Response -> position units (enumerated items / bullets, else sentences)."""
    units = []
    for raw in text.splitlines():
        s = BOLD.sub("", BULLET.sub("", raw.strip())).strip()
        if len(s.split()) >= 8:
            units.append(s)
    if len(units) < 3:                       # prose answer: fall back to sentences
        units = [s.strip() for s in re.split(r"(?<=[.!?])\s+", BOLD.sub("", text))
                 if len(s.split()) >= 8]
    return units[:40]


MIN_K = 3          # need >=3 units on BOTH sides for a matched-k comparison
N_SUB = 20         # subsamples averaged per matched-k estimate


def vendi(sim: np.ndarray) -> float:
    """Effective number of distinct modes: exp(Shannon entropy of eigvals of S/n).

    Pass the RAW Gram matrix of unit vectors -- PSD with unit diagonal, the
    Vendi kernel requirement (arXiv:2210.02410). Do not clip S to [0,1] first:
    that breaks PSD, and it fires exactly when cosines are negative, i.e. on
    the most diverse pools.

    Grows with n -- only compare Vendi across responses of MATCHED unit count.
    """
    n = sim.shape[0]
    if n < 2:
        return float(n)
    w = np.linalg.eigvalsh(sim / n)
    w = w[w > 1e-10]
    return float(np.exp(-(w * np.log(w)).sum()))


def matched_vendi(U: np.ndarray, k: int, rng) -> float:
    """Vendi at exactly k units, averaged over subsamples -- length-controlled."""
    return float(np.mean([vendi(U[i] @ U[i].T)
                          for i in (rng.choice(len(U), k, replace=False)
                                    for _ in range(N_SUB))]))


def mean_pairwise_cos(U: np.ndarray) -> float:
    """Mean unit-to-unit cosine (higher = positions more alike). Length-robust."""
    if len(U) < 2:
        return float("nan")
    iu = np.triu_indices(len(U), 1)
    return float((U @ U.T)[iu].mean())


def main():
    ap = argparse.ArgumentParser(description="Test the pole-collapse mechanism")
    ap.add_argument("--responses", default="overton_responses_v5.jsonl")
    ap.add_argument("--scores", default="overton_scores_v5.csv")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2",
                    help="held-out embedder (NOT the scout's MiniLM)")
    ap.add_argument("--out", default="docs/pole_collapse.csv")
    ap.add_argument("--seed", type=int, default=0, help="matched-k subsampling")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    cov = defaultdict(dict)
    for r in csv.DictReader(open(args.scores, encoding="utf-8")):
        cov[int(r["question_id"])][r["condition"]] = float(r["coverage"])

    resp = defaultdict(dict)
    for line in open(args.responses, encoding="utf-8"):
        r = json.loads(line)
        resp[r["question_id"]][r["condition"]] = r

    enc = SentenceTransformer(args.model)
    rng = np.random.default_rng(args.seed)

    def embed(texts):
        return enc.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                          batch_size=64)

    rows = []
    for qid in sorted(resp):
        byc = resp[qid]
        if "scout" not in byc or "baseline" not in byc or qid not in cov:
            continue
        poles = parse_poles(byc["scout"].get("fork_context") or "")
        if not poles:                        # scout fell back to the baseline prompt
            continue
        P = embed(poles)
        rec = {"qid": qid, "n_poles": len(poles),
               "base_cov": cov[qid]["baseline"],
               "delta": cov[qid]["scout"] - cov[qid]["baseline"],
               "w_baseline": len((byc["baseline"].get("response") or "").split()),
               "w_scout": len((byc["scout"].get("response") or "").split())}
        emb = {}
        for c in ("baseline", "scout", "div_only"):
            if c not in byc:
                continue
            units = split_units(byc[c].get("response") or "")
            if len(units) < 2:
                rec[f"pole_{c}"] = rec[f"modes_{c}"] = float("nan")
                continue
            U = embed(units)
            emb[c] = U
            rec[f"pole_{c}"] = float((U @ P.T).max(axis=1).mean())  # on-pole mass
            rec[f"modes_{c}"] = vendi(U @ U.T)       # raw: LENGTH-BIASED
            rec[f"mpc_{c}"] = mean_pairwise_cos(U)                  # length-robust
            rec[f"nunits_{c}"] = len(units)
        # length-controlled breadth: Vendi at matched unit count (see module docstring)
        if "baseline" in emb and "scout" in emb:
            k = min(len(emb["baseline"]), len(emb["scout"]))
            if k >= MIN_K:
                rec["k"] = k
                for c in ("baseline", "scout"):
                    rec[f"vendi_k_{c}"] = matched_vendi(emb[c], k, rng)
                rec["d_vendi_k"] = rec["vendi_k_scout"] - rec["vendi_k_baseline"]
        rows.append(rec)

    ok = [r for r in rows if not np.isnan(r.get("pole_scout", float("nan")))
          and not np.isnan(r.get("pole_baseline", float("nan")))]
    n = len(ok)
    for r in ok:
        r["attraction"] = r["pole_scout"] - r["pole_baseline"]
        r["d_modes"] = r["modes_scout"] - r["modes_baseline"]

    print(f"n={n} questions with injected poles (of {len(resp)})  "
          f"embedder={args.model.split('/')[-1]}")

    att = [r["attraction"] for r in ok]
    dm = [r["d_modes"] for r in ok]
    dl = [r["delta"] for r in ok]

    print("\n--- P1 attraction: does scout output move ONTO the injected poles? ---")
    print(f"  pole-sim baseline {st.mean([r['pole_baseline'] for r in ok]):.4f}")
    print(f"  pole-sim scout    {st.mean([r['pole_scout'] for r in ok]):.4f}")
    if any("pole_div_only" in r and not np.isnan(r["pole_div_only"]) for r in ok):
        dv = [r["pole_div_only"] for r in ok if not np.isnan(r.get("pole_div_only", float('nan')))]
        print(f"  pole-sim div_only {st.mean(dv):.4f}   (control)")
    print(f"  attraction (scout-baseline) mean {st.mean(att):+.4f}  "
          f"median {st.median(att):+.4f}  frac>0 {sum(a > 0 for a in att)/n:.2f}")
    print(f"  P1 {'HOLDS' if st.mean(att) > 0 else 'FAILS'}: "
          f"predicted mean attraction > 0")

    print("\n--- P2 causal link: does moving onto poles predict coverage loss? ---")
    c_ad = _corr(att, dl)
    print(f"  corr(attraction, delta_coverage) = {c_ad:+.3f}")
    print(f"  corr(d_modes,    delta_coverage) = {_corr(dm, dl):+.3f}")
    print(f"  P2 {'HOLDS' if c_ad < -0.15 else 'FAILS'}: predicted corr < 0")

    print("\n--- P3 collapse: fewer distinct positions, worst where baseline broad ---")
    print(f"  words:  baseline {st.mean([r['w_baseline'] for r in ok]):5.0f}   "
          f"scout {st.mean([r['w_scout'] for r in ok]):5.0f}   "
          f"<- injected answers are much LONGER, so raw Vendi is biased up")
    print(f"  RAW Vendi (length-biased, do NOT interpret): "
          f"baseline {st.mean([r['modes_baseline'] for r in ok]):.2f}  "
          f"scout {st.mean([r['modes_scout'] for r in ok]):.2f}  d {st.mean(dm):+.3f}")

    mk = [r for r in ok if "d_vendi_k" in r]
    print(f"\n  LENGTH-CONTROLLED (n={len(mk)}, Vendi at k=min units, {N_SUB} subsamples):")
    print(f"    Vendi@k   baseline {st.mean([r['vendi_k_baseline'] for r in mk]):.3f}   "
          f"scout {st.mean([r['vendi_k_scout'] for r in mk]):.3f}   "
          f"d {st.mean([r['d_vendi_k'] for r in mk]):+.3f}")
    print(f"    mean-pair-cos  baseline {st.mean([r['mpc_baseline'] for r in mk]):.3f}   "
          f"scout {st.mean([r['mpc_scout'] for r in mk]):.3f}   (higher = more alike)")
    print(f"    P3 {'HOLDS' if st.mean([r['d_vendi_k'] for r in mk]) < 0 else 'FAILS'}: "
          f"predicted Vendi@k drop < 0")

    bins = [("base < 0.3", lambda b: b < 0.3),
            ("0.3-0.8", lambda b: 0.3 <= b < 0.8),
            ("base >= 0.8", lambda b: b >= 0.8)]
    print(f"\n  {'bucket':<13}{'n':>4}{'d_cov':>9}{'attraction':>12}"
          f"{'d_Vendi@k':>11}{'mpc_base':>10}{'mpc_scout':>11}")
    for name, pred in bins:
        g = [r for r in mk if pred(r["base_cov"])]
        if not g:
            continue
        print(f"  {name:<13}{len(g):>4}{st.mean([r['delta'] for r in g]):>+9.3f}"
              f"{st.mean([r['attraction'] for r in g]):>+12.4f}"
              f"{st.mean([r['d_vendi_k'] for r in g]):>+11.3f}"
              f"{st.mean([r['mpc_baseline'] for r in g]):>10.3f}"
              f"{st.mean([r['mpc_scout'] for r in g]):>11.3f}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        cols = ["qid", "base_cov", "delta", "attraction", "n_poles", "k",
                "pole_baseline", "pole_scout", "pole_div_only",
                "vendi_k_baseline", "vendi_k_scout", "d_vendi_k",
                "mpc_baseline", "mpc_scout", "w_baseline", "w_scout",
                "modes_baseline", "modes_scout", "d_modes"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(ok)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
