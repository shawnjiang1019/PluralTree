"""Model-free validation of the divergence embedding on GlobalOpinionQA.

Does the learned geometry actually rank opinion divergence, or is the Wasserstein
signal an artifact? We have a ground truth that never touches the encoder: the raw
answer distributions ``opinion_dist``. For two countries answering the SAME
question, their Jensen-Shannon divergence is an exact, model-free measure of how
differently they answer.

This script pairs same-question opinions across countries and correlates:

    oracle   = JS(dist_i, dist_j)          (raw distributions, no embedding)
    embedding = geodesic(h_i, h_j)         (the learned Poincare distance)

A high Spearman means the embedding faithfully ranks true divergence — the
branch_divergence signal is trustworthy. Near zero means the pipeline is losing
the divergence that demonstrably exists in the data (feature design or encoder),
not that the data lacks it. A shuffled-pair null is reported as the floor.

Usage:
    python -m evaluation.validate_divergence \
        --embeddings embeddings_goqa.pt --curvature 0.5
"""

from __future__ import annotations

import argparse
import math

import torch

from evaluation.structure_metrics import _spearman, _pearson


def _js_divergence(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon divergence in bits (base-2), bounded [0, 1]."""
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]

    def _kl(a, b):
        s = 0.0
        for ai, bi in zip(a, b):
            if ai > 0.0 and bi > 0.0:
                s += ai * math.log2(ai / bi)
        return s

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _question_of(name: str) -> str | None:
    """Opinion node id is ``op_{row}_{country}``; the row index keys the question."""
    parts = name.split("_", 2)
    return parts[1] if len(parts) >= 3 and parts[0] == "op" else None


def main():
    ap = argparse.ArgumentParser(description="JS-oracle validation of GOQA divergence")
    ap.add_argument("--embeddings", required=True, help=".pt of h_all on the ball")
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_pairs", type=int, default=20000,
                    help="cap same-question country pairs (0 = all)")
    ap.add_argument("--question", default=None,
                    help="row-index key of ONE question: print its per-country "
                         "divergence (embedding + JS) and exit")
    ap.add_argument("--responses", type=int, default=3,
                    help="in --question mode, also print the actual answer "
                         "distributions for the top-K most divergent country pairs")
    ap.add_argument("--rank_questions", type=int, default=0,
                    help="print the top-K most divergent questions (by mean JS) and exit")
    args = ap.parse_args()

    from pluraltree.manifolds.poincare import PoincareBall
    from data.globalopinionqa import load_globalopinionqa

    graph = load_globalopinionqa(split_seed=args.seed, leakage_safe=True)
    h_all = torch.load(args.embeddings, map_location="cpu")
    if not isinstance(h_all, torch.Tensor):
        h_all = h_all["h_all"]
    manifold = PoincareBall(c=args.curvature)

    # Group opinion nodes by question (same row index == same question + options).
    by_q: dict[str, list[int]] = {}
    for nid, etype in graph.entity_types.items():
        if etype != "opinion":
            continue
        q = _question_of(graph.id_to_entity[nid])
        if q is not None and nid in graph.opinion_dist:
            by_q.setdefault(q, []).append(nid)

    def canon_of(nid: int) -> str:
        parts = graph.id_to_entity[nid].split("_", 2)
        return parts[2] if len(parts) >= 3 else "?"

    import os

    def _prefix(nid: int) -> str:
        """Shared '{question} ' prefix of an opinion's option strings."""
        texts = graph.opinion_texts.get(nid, [])
        pref = os.path.commonprefix(texts)
        cut = pref.rfind(" ")
        return pref[:cut + 1] if cut > 0 else pref

    def qtext(nid: int) -> str:
        # opinion_texts keeps the FULL (untruncated) question; entity_text is capped.
        p = _prefix(nid).strip()
        return p or graph.entity_text.get(nid, "").rsplit(" [", 1)[0]

    def options_of(nid: int) -> list[str]:
        """Recover the answer options by stripping the shared question prefix."""
        pref = _prefix(nid)
        return [t[len(pref):] for t in graph.opinion_texts.get(nid, [])]

    def pairwise(oids):
        """All country-pair (JS, geodesic) for one question's opinion leaves."""
        js, geo, pr = [], [], []
        for a in range(len(oids)):
            for b in range(a + 1, len(oids)):
                i, j = oids[a], oids[b]
                di, dj = graph.opinion_dist[i], graph.opinion_dist[j]
                if len(di) != len(dj):
                    continue
                jd = _js_divergence(di, dj)
                gd = float(manifold.distance(h_all[i:i+1], h_all[j:j+1]).squeeze())
                js.append(jd); geo.append(gd); pr.append((i, j, jd, gd))
        return js, geo, pr

    import statistics as st

    # --- single-question mode: diversity of responses to ONE question --------
    if args.question is not None:
        oids = by_q.get(args.question)
        if not oids:
            print(f"question {args.question!r} not found "
                  f"(keys are row indices: {sorted(by_q)[:8]}...)")
            return
        print(f"Q[{args.question}] {qtext(oids[0])}")
        print(f"  {len(oids)} countries responding")
        js, geo, pr = pairwise(oids)
        if not js:
            print("  <2 comparable responses"); return
        print(f"  mean JS={st.mean(js):.4f}  max JS={max(js):.4f}  "
              f"mean geodesic={st.mean(geo):.4f}")
        pr.sort(key=lambda x: x[2], reverse=True)
        print("  most divergent country pairs (by JS):")
        for i, j, jd, gd in pr[:10]:
            print(f"    {canon_of(i):<16} vs {canon_of(j):<16}  JS={jd:.4f}  geodesic={gd:.4f}")

        # Actual answer distributions for the top-K pairs — eyeball the ranking.
        for i, j, jd, gd in pr[:max(0, args.responses)]:
            print(f"\n  responses: {canon_of(i)} vs {canon_of(j)}  "
                  f"(JS={jd:.4f}  geodesic={gd:.4f})")
            opts = options_of(i)
            pa, pb = graph.opinion_dist[i], graph.opinion_dist[j]
            print(f"    {canon_of(i)[:12]:>12} {canon_of(j)[:12]:>12}   option")
            for k, opt in enumerate(opts):
                print(f"    {pa[k]:>12.2f} {pb[k]:>12.2f}   {opt[:70]}")
        return

    # --- rank questions by how divergent the responses are -------------------
    if args.rank_questions:
        rows = []
        for q, oids in by_q.items():
            if len(oids) < 2:
                continue
            js, geo, _ = pairwise(oids)
            if js:
                rows.append((q, st.mean(js), st.mean(geo), len(oids), qtext(oids[0])))
        rows.sort(key=lambda x: x[1], reverse=True)
        print(f"top {args.rank_questions} most divergent questions (by mean JS):")
        for q, mj, mg, n, qt in rows[:args.rank_questions]:
            print(f"  JS={mj:.4f}  geo={mg:.4f}  n={n:>3}  Q[{q}] {qt[:66]}")
        return

    oracle, embed = [], []
    for q, oids in by_q.items():
        for a in range(len(oids)):
            for b in range(a + 1, len(oids)):
                i, j = oids[a], oids[b]
                di, dj = graph.opinion_dist[i], graph.opinion_dist[j]
                if len(di) != len(dj):
                    continue
                oracle.append(_js_divergence(di, dj))
                embed.append(float(manifold.distance(h_all[i:i+1], h_all[j:j+1]).squeeze()))
                if args.max_pairs and len(oracle) >= args.max_pairs:
                    break
            if args.max_pairs and len(oracle) >= args.max_pairs:
                break
        if args.max_pairs and len(oracle) >= args.max_pairs:
            break

    if len(oracle) < 2:
        print(f"Only {len(oracle)} same-question pairs found — nothing to correlate.")
        return

    o = torch.tensor(oracle)
    e = torch.tensor(embed)
    rho = _spearman(e, o)
    r = _pearson(e, o)

    # Shuffled-pair null: break the pairing, re-correlate.
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(o), generator=g)
    rho_null = _spearman(e, o[perm])

    print(f"same-question country pairs: n={len(oracle)}  "
          f"({len(by_q)} questions)")
    print(f"oracle JS   : mean={o.mean():.4f}  max={o.max():.4f}")
    print(f"embed geodesic: mean={e.mean():.4f}  max={e.max():.4f}")
    print(f"Spearman(geodesic, JS) = {rho:+.4f}   [shuffled null {rho_null:+.4f}]")
    print(f"Pearson (geodesic, JS) = {r:+.4f}")
    verdict = ("STRONG: embedding ranks true divergence" if rho > 0.4 else
               "MODERATE: partial" if rho > 0.2 else
               "WEAK: pipeline is losing the divergence that exists in the data")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
