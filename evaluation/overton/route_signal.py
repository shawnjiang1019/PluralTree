"""Which graph signal predicts when scout injection HELPS coverage?

The v5 traces showed injection helps on contested questions (+0.31) and hurts on
consensus ones (-0.45); an oracle router (per-question pick better of
baseline/scout) scores 0.62 vs 0.50 always-baseline. But raw fork divergence W
barely tracks the help-delta (corr +0.19) because the scout SELECTS max-W forks,
so W is high even on consensus questions (divergent-but-tangential survey items).

This script tests, on the 60 OvertonBench questions, whether any *inference-time*
graph signal separates helped-from-hurt — BEFORE spending a generation run on a
router. For each question it runs the scout, extracts candidate signals, joins
with the known per-question (scout - baseline) coverage delta, and reports each
signal's correlation with the delta + the best single-threshold gate score.

Candidate signals:
  w_raw       best fork's Wasserstein (what the scout already ranks on)
  z_level     calibrated divergence: is the anchor's subgroup split ABOVE the
              same-depth chance null? (contestedness, not just magnitude)
  relevance   best fork's mean question-relevance (anti-correlated in v5)
  driver_sim  MiniLM cosine between the fork's actual survey question and the
              USER question (is the divergence ON-topic, not just near-topic?)

CPU-only (no vLLM); needs the embeddings + scores CSV. ~minutes.

Usage:
    python -m evaluation.overton.route_signal \
        --embeddings embeddings_opinionqa.pt --text_feat feats_opinionqa.pt \
        --scores overton_scores_v5.csv --dataset opinionqa
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _corr(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


def _best_gate(sig, delta, base_cov, scout_cov):
    """Best 'inject iff sig>thr' OvertonScore over all thresholds (upper bound;
    single-split overfit to these questions — read as a ceiling for this signal)."""
    n = len(sig)
    base_mean = sum(base_cov) / n
    thrs = [-1e9] + sorted(set(sig))
    best_score, best_thr = base_mean, None
    for t in thrs:
        s = sum(scout_cov[i] if sig[i] > t else base_cov[i] for i in range(n)) / n
        if s > best_score:
            best_score, best_thr = s, t
    return best_score, best_thr, base_mean


def main():
    ap = argparse.ArgumentParser(description="Validate scout-injection routing signals")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--text_feat", default=None)
    ap.add_argument("--scores", required=True, help="overton_scores_vN.csv (long form)")
    ap.add_argument("--dataset", choices=["globalopinionqa", "opinionqa"], default="opinionqa")
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--n_null", type=int, default=300)
    args = ap.parse_args()

    import torch
    from pluraltree.manifolds.poincare import PoincareBall
    from evaluation.intrinsic.branch_divergence import branch_divergence, _null_divergence
    from evaluation.overton.eval_overtonbench import load_questions
    from retrieval.scout import (ScoutConfig, embed_question, node_relevance,
                                 load_or_compute_text_feat, scout)

    # coverage deltas
    cov = defaultdict(dict)
    for r in csv.DictReader(open(args.scores, encoding="utf-8")):
        cov[int(r["question_id"])][r["condition"]] = float(r["coverage"])

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
    cfg = ScoutConfig(tau=args.tau, alpha=1.0)

    # same-depth chance null for divergence calibration (computed once)
    null_mu, null_sd = _null_divergence(h_all, graph.children_indices,
                                        manifold=manifold, n_samples=args.n_null,
                                        seed=args.seed, same_level=True)
    print(f"same-level null W: mean={null_mu:.3f} std={null_sd:.3f}")

    rows = []
    for qid, question in load_questions():
        if qid not in cov or "scout" not in cov[qid]:
            continue
        q_emb = embed_question(question)
        forks = scout(question, graph, h_all, text_feat, manifold, cfg=cfg, q_emb=q_emb)
        delta = cov[qid]["scout"] - cov[qid]["baseline"]
        rec = {"qid": qid, "delta": delta, "base": cov[qid]["baseline"],
               "scout": cov[qid]["scout"]}
        if not forks:
            rec.update(w_raw=0.0, z_level=0.0, relevance=0.0, driver_sim=0.0)
            rows.append(rec)
            continue
        top = forks[0]
        # calibrated divergence: anchor's max child-pair W vs same-level null
        bd = branch_divergence(top.anchor, h_all, graph.children_indices,
                               manifold=manifold)
        z = (bd["max"] - null_mu) / null_sd if null_sd and null_sd > 0 else 0.0
        # driver_sim: does the fork's survey question match the USER question?
        drv_ids = [top.branch_a, top.branch_b] + [a for a, _, _ in top.top_pairs]
        drv_txt = " ".join(getattr(graph, "entity_text", {}).get(i, "") for i in drv_ids)
        d_emb = embed_question(drv_txt) if drv_txt.strip() else None
        driver_sim = float(node_relevance(q_emb, d_emb.unsqueeze(0))[0]) if d_emb is not None else 0.0
        rec.update(w_raw=top.w, z_level=float(z), relevance=top.relevance,
                   driver_sim=driver_sim)
        rows.append(rec)

    n = len(rows)
    delta = [r["delta"] for r in rows]
    base_cov = [r["base"] for r in rows]
    scout_cov = [r["scout"] for r in rows]
    print(f"\nn={n}  always-baseline={sum(base_cov)/n:.4f}  "
          f"always-scout={sum(scout_cov)/n:.4f}  "
          f"oracle={sum(max(r['base'], r['scout']) for r in rows)/n:.4f}")

    print(f"\n{'signal':<12}{'corr(delta)':>13}{'best_gate':>11}{'thr':>10}"
          f"{'helped_mean':>13}{'hurt_mean':>11}")
    for sig_name in ("w_raw", "z_level", "relevance", "driver_sim"):
        sig = [r[sig_name] for r in rows]
        c = _corr(sig, delta)
        score, thr, base_mean = _best_gate(sig, delta, base_cov, scout_cov)
        helped = [r[sig_name] for r in rows if r["delta"] > 0.01]
        hurt = [r[sig_name] for r in rows if r["delta"] < -0.01]
        thr_s = f"{thr:.3f}" if thr is not None else "none"
        print(f"{sig_name:<12}{c:>+13.3f}{score:>11.4f}{thr_s:>10}"
              f"{st.mean(helped) if helped else float('nan'):>13.3f}"
              f"{st.mean(hurt) if hurt else float('nan'):>11.3f}")
    print("\n(best_gate is an overfit ceiling; a signal is only useful if its "
          "helped_mean and hurt_mean clearly separate.)")


if __name__ == "__main__":
    main()
