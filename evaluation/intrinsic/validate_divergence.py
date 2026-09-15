"""Model-free validation of the divergence embedding: does geometry rank real disagreement?

Does the learned geometry actually rank opinion divergence, or is the Wasserstein
signal an artifact? We have a ground truth that never touches the encoder: the raw
answer distributions ``opinion_dist``. For two groups answering the SAME question,
their Jensen-Shannon divergence is an exact, model-free measure of how differently
they answer.

    oracle    = JS(dist_i, dist_j)          (raw distributions, no embedding)
    embedding = geodesic(h_i, h_j)          (the learned Poincare distance)
    input     = 1 - cos(feat_i, feat_j)     (--feats: the encoder's own INPUT,
                                             distribution-weighted option text)

Pairs:
  globalopinionqa   countries answering the same question
  opinionqa / issp  --pairs axis (default): subgroups on the same question AND
                    attribute -- the siblings the scout actually forks between.
                    --pairs question: any two subgroups on the same question.

WHY THE OPINIONQA RUN MATTERS. The encoder is trained on link prediction, a
structure-fidelity term and a sibling-separation FLOOR that pushes every pair of
siblings apart by the same margin whether or not they disagree. Nothing in that
objective asks subgroups that answer differently to be far apart. If the geometry
does not rank disagreement, the scout's max-divergence fork choice is close to
random -- which would explain why curvature, hierarchy and selection ablations tie.

Read the three numbers together:
  input high, embedding low   the encoder (its objective) discards disagreement
  both low                    the input features never carried it
  both high                   the geometry is faithful; the ablation nulls come
                              from further downstream (delivery, judge targets)

The WITHIN-GROUP Spearman is the one the scout depends on: it only ever compares
siblings of one axis, so a pooled correlation driven by between-question
differences would overstate what fork selection can use.

Usage:
    python -m evaluation.intrinsic.validate_divergence --dataset opinionqa \
        --embeddings embeddings_opinionqa_c0p5.pt --curvature 0.5 \
        --feats feats_opinionqa.pt
    python -m evaluation.intrinsic.validate_divergence \
        --embeddings embeddings_goqa.pt --curvature 0.5
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics as st

import torch

from evaluation.intrinsic.structure_metrics import _pearson, _spearman


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


def leaf_key(name: str, dataset: str, pairs: str = "axis") -> str | None:
    """Group key for an opinion leaf, or None if ``name`` is not one.

    GOQA ids are ``op_{row}_{country}``; OpinionQA/ISSP ids are
    ``op:{qkey}:{attr}:{group}`` (qkey and attr carry no colons; group may).
    """
    if dataset == "globalopinionqa":
        parts = name.split("_", 2)
        return parts[1] if len(parts) >= 3 and parts[0] == "op" else None
    if not name.startswith("op:"):
        return None
    parts = name[3:].split(":", 2)
    if len(parts) != 3:
        return None
    return f"{parts[0]}:{parts[1]}" if pairs == "axis" else parts[0]


def canon(name: str, dataset: str) -> str:
    """Short label for a leaf: the country, or attr=group."""
    if dataset == "globalopinionqa":
        parts = name.split("_", 2)
        return parts[2] if len(parts) >= 3 else "?"
    parts = name[3:].split(":", 2)
    return f"{parts[1]}={parts[2]}" if len(parts) == 3 else "?"


def within_group_spearman(groups: dict[str, list[tuple[float, float]]],
                          min_pairs: int = 3) -> tuple[float, int]:
    """Mean Spearman(x, y) computed separately inside each group with enough
    pairs, and the number of groups used."""
    rhos = []
    for prs in groups.values():
        if len(prs) < min_pairs:
            continue
        x = torch.tensor([p[0] for p in prs], dtype=torch.float)
        y = torch.tensor([p[1] for p in prs], dtype=torch.float)
        if x.std() == 0 or y.std() == 0:
            continue
        r = float(_spearman(x, y))
        if r == r:
            rhos.append(r)
    return (st.mean(rhos) if rhos else float("nan")), len(rhos)


def _verdict(rho: float) -> str:
    return ("STRONG" if rho > 0.4 else "MODERATE" if rho > 0.2 else "WEAK")


def main():
    ap = argparse.ArgumentParser(description="JS-oracle validation of learned divergence")
    ap.add_argument("--embeddings", required=True, help=".pt of h_all on the ball")
    ap.add_argument("--dataset", choices=["globalopinionqa", "opinionqa", "issp"],
                    default="globalopinionqa")
    ap.add_argument("--pairs", choices=["axis", "question"], default="axis",
                    help="opinionqa/issp: pair siblings of one attribute (what the "
                         "scout compares) or any two groups on one question")
    ap.add_argument("--feats", default=None,
                    help="node-feature cache (e.g. feats_opinionqa.pt): also "
                         "correlate the encoder's INPUT distance with JS")
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42,
                    help="graph split seed; MUST match train.py --seed (42) or "
                         "node ids and embedding rows disagree")
    ap.add_argument("--max_pairs", type=int, default=20000,
                    help="cap pairs, drawn from groups in shuffled order (0 = all)")
    ap.add_argument("--question", default=None,
                    help="one group key: print its per-pair divergence and exit")
    ap.add_argument("--responses", type=int, default=3,
                    help="in --question mode, also print the answer distributions "
                         "for the top-K most divergent pairs")
    ap.add_argument("--rank_questions", type=int, default=0,
                    help="print the top-K most divergent groups (by mean JS) and exit")
    args = ap.parse_args()

    from data.loaders.graphs import load_graph
    from pluraltree.manifolds.poincare import PoincareBall

    graph = load_graph(args.dataset, split_seed=args.seed, leakage_safe=True)
    h_all = torch.load(args.embeddings, map_location="cpu")
    if not isinstance(h_all, torch.Tensor):
        h_all = h_all["h_all"]
    n_nodes = len(graph.id_to_entity)
    if h_all.shape[0] != n_nodes:
        raise SystemExit(f"{args.embeddings} has {h_all.shape[0]} rows but the "
                         f"{args.dataset} graph (seed {args.seed}) has {n_nodes} nodes")
    feats = None
    if args.feats:
        feats = torch.load(args.feats, map_location="cpu")
        if feats.shape[0] != n_nodes:
            raise SystemExit(f"{args.feats} has {feats.shape[0]} rows, graph has {n_nodes}")
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
    manifold = PoincareBall(c=args.curvature)

    by_q: dict[str, list[int]] = {}
    for nid, etype in graph.entity_types.items():
        if etype != "opinion" or nid not in graph.opinion_dist:
            continue
        k = leaf_key(graph.id_to_entity[nid], args.dataset, args.pairs)
        if k is not None:
            by_q.setdefault(k, []).append(nid)

    def label(nid: int) -> str:
        return canon(graph.id_to_entity[nid], args.dataset)

    def _prefix(nid: int) -> str:
        """Shared '{question} ' prefix of an opinion's option strings."""
        texts = graph.opinion_texts.get(nid, [])
        pref = os.path.commonprefix(texts)
        cut = pref.rfind(" ")
        return pref[:cut + 1] if cut > 0 else pref

    def qtext(nid: int) -> str:
        p = _prefix(nid).strip()
        return p or graph.entity_text.get(nid, "").rsplit(" [", 1)[0]

    def options_of(nid: int) -> list[str]:
        pref = _prefix(nid)
        return [t[len(pref):] for t in graph.opinion_texts.get(nid, [])]

    def pair_stats(i: int, j: int):
        di, dj = graph.opinion_dist[i], graph.opinion_dist[j]
        if len(di) != len(dj):
            return None
        gd = float(manifold.distance(h_all[i:i + 1], h_all[j:j + 1]).squeeze())
        fd = float(1.0 - feats[i] @ feats[j]) if feats is not None else float("nan")
        return _js_divergence(di, dj), gd, fd

    def pairwise(oids):
        out = []
        for a in range(len(oids)):
            for b in range(a + 1, len(oids)):
                s = pair_stats(oids[a], oids[b])
                if s is not None:
                    out.append((oids[a], oids[b], *s))
        return out

    # --- single-group mode ----------------------------------------------------
    if args.question is not None:
        oids = by_q.get(args.question)
        if not oids:
            print(f"group {args.question!r} not found (keys look like: "
                  f"{sorted(by_q)[:5]})")
            return
        print(f"[{args.question}] {qtext(oids[0])}")
        print(f"  {len(oids)} groups responding")
        pr = pairwise(oids)
        if not pr:
            print("  <2 comparable responses"); return
        print(f"  mean JS={st.mean(p[2] for p in pr):.4f}  "
              f"max JS={max(p[2] for p in pr):.4f}  "
              f"mean geodesic={st.mean(p[3] for p in pr):.4f}")
        pr.sort(key=lambda x: x[2], reverse=True)
        print("  most divergent pairs (by JS):")
        for i, j, jd, gd, _ in pr[:10]:
            print(f"    {label(i):<24} vs {label(j):<24}  JS={jd:.4f}  geodesic={gd:.4f}")
        for i, j, jd, gd, _ in pr[:max(0, args.responses)]:
            print(f"\n  responses: {label(i)} vs {label(j)}  (JS={jd:.4f}  geodesic={gd:.4f})")
            pa, pb = graph.opinion_dist[i], graph.opinion_dist[j]
            for k, opt in enumerate(options_of(i)):
                print(f"    {pa[k]:>6.2f} {pb[k]:>6.2f}   {opt[:70]}")
        return

    # --- rank groups by divergence ---------------------------------------------
    if args.rank_questions:
        rows = []
        for q, oids in by_q.items():
            pr = pairwise(oids) if len(oids) >= 2 else []
            if pr:
                rows.append((q, st.mean(p[2] for p in pr), st.mean(p[3] for p in pr),
                             len(oids), qtext(oids[0])))
        rows.sort(key=lambda x: x[1], reverse=True)
        print(f"top {args.rank_questions} most divergent groups (by mean JS):")
        for q, mj, mg, n, qt in rows[:args.rank_questions]:
            print(f"  JS={mj:.4f}  geo={mg:.4f}  n={n:>3}  [{q}] {qt[:60]}")
        return

    # --- correlation ------------------------------------------------------------
    keys = sorted(by_q)
    random.Random(args.seed).shuffle(keys)      # cap without favouring early questions
    oracle, embed, inp = [], [], []
    groups_geo: dict[str, list] = {}
    groups_inp: dict[str, list] = {}
    for q in keys:
        for i, j, jd, gd, fd in pairwise(by_q[q]):
            oracle.append(jd); embed.append(gd); inp.append(fd)
            groups_geo.setdefault(q, []).append((gd, jd))
            groups_inp.setdefault(q, []).append((fd, jd))
        if args.max_pairs and len(oracle) >= args.max_pairs:
            break

    if len(oracle) < 2:
        print(f"Only {len(oracle)} comparable pairs found -- nothing to correlate.")
        return

    o, e = torch.tensor(oracle), torch.tensor(embed)
    rho, r = _spearman(e, o), _pearson(e, o)
    perm = torch.randperm(len(o), generator=torch.Generator().manual_seed(args.seed))
    rho_null = _spearman(e, o[perm])
    w_geo, n_geo = within_group_spearman(groups_geo)

    unit = "country pairs" if args.dataset == "globalopinionqa" else f"{args.pairs} pairs"
    print(f"{args.dataset}: {unit} n={len(oracle)}  ({len(groups_geo)} groups)")
    print(f"oracle JS      : mean={o.mean():.4f}  max={o.max():.4f}")
    print(f"embed geodesic : mean={e.mean():.4f}  max={e.max():.4f}  "
          f"sd={e.std():.4f}")
    print(f"Spearman(geodesic, JS) pooled       = {rho:+.4f}   [shuffled null {rho_null:+.4f}]")
    print(f"Spearman(geodesic, JS) within-group = {w_geo:+.4f}   ({n_geo} groups with >=3 pairs)")
    print(f"Pearson (geodesic, JS) pooled       = {r:+.4f}")

    if feats is not None:
        f = torch.tensor(inp)
        rho_f = _spearman(f, o)
        w_inp, n_inp = within_group_spearman(groups_inp)
        print(f"Spearman(input dist, JS) pooled       = {rho_f:+.4f}")
        print(f"Spearman(input dist, JS) within-group = {w_inp:+.4f}   ({n_inp} groups)")
        print("\nreading (within-group, the scale fork selection uses):")
        print(f"  input    {_verdict(w_inp):<8} {w_inp:+.3f}")
        print(f"  geometry {_verdict(w_geo):<8} {w_geo:+.3f}")
        if w_inp > 0.2 and w_geo < w_inp - 0.15:
            print("  -> the input carries disagreement and the ENCODER discards it: "
                  "the training objective is the suspect")
        elif w_inp <= 0.2:
            print("  -> the input features barely carry disagreement: the feature "
                  "design, not only the objective, limits what geometry can rank")
        else:
            print("  -> the geometry keeps what the input carries: look downstream "
                  "(delivery, judge targets) for the ablation nulls")
    else:
        print(f"verdict (within-group): {_verdict(w_geo)}  -- pass --feats to tell an "
              f"encoder loss apart from a feature-design loss")


def _selftest() -> None:
    assert leaf_key("op:ABORT_W32:AGE:18-29", "opinionqa") == "ABORT_W32:AGE"
    assert leaf_key("op:ABORT_W32:AGE:18-29", "opinionqa", "question") == "ABORT_W32"
    assert leaf_key("op:q1:party:a:b", "issp") == "q1:party", "colons in group"
    assert leaf_key("ax:q1:AGE", "opinionqa") is None
    assert leaf_key("op_12_Canada", "globalopinionqa") == "12"
    assert canon("op:ABORT_W32:AGE:18-29", "opinionqa") == "AGE=18-29"
    assert canon("op_12_Canada", "globalopinionqa") == "Canada"
    g = {"a": [(1, 1), (2, 2), (3, 3)], "b": [(1, 3), (2, 2), (3, 1)], "c": [(1, 1)]}
    m, n = within_group_spearman(g)
    assert n == 2 and abs(m) < 1e-6, (m, n)
    print("validate_divergence self-test OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
