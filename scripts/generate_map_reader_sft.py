"""Generate Phase-1 *Map Reader* SFT data — fully deterministic, no Claude.

The Map Reader teaches the LM to *read a latent*: given the injected Poincaré
embedding (and nothing else for the structural fact), decode the node's geometry.
Targets are derived directly from the tree + ``h_all``, so there is no distiller
and no API cost.

Two task kinds (see pluraltree/sft/plan.md, Phase 1):
    1A  decode   : input = latent only            -> target = full structural caption
    1B  qa       : input = latent + templated Q   -> target = the structural answer

Unlike Phase-2 (Reasoner) data, the caption here is the FULL, geometry-bearing
one (exact depth / rho / branching) — Phase 1 *teaches* latent-reading, so the
geometry must be the supervision target. Caption-starving applies to Phase 2 only.

Each record stores ``node_id`` so the matching latent is looked up and injected at
train time (the prompt text never contains the geometry for 1B — that would leak).

Usage:
    python scripts/generate_map_reader_sft.py \
        --dataset wn18rr --data_dir data/wn18rr \
        --embeddings runs/h_all.pt --out data/map_reader_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from pluraltree.manifolds.poincare import PoincareBall
from evaluation.structure_metrics import (
    _parents_from_children,
    _depths,
    _level_k_ancestors,
)


def _role(node_id: int, children_indices, parents) -> str:
    if not parents[node_id]:
        return "root"
    if not children_indices[node_id]:
        return "leaf"
    return "junction"


def _abstraction(rho: float) -> str:
    return "specific" if rho > 0.66 else "broad" if rho < 0.33 else "intermediate"


def _branching(node_id: int, children_indices, depth, levels: int = 2) -> list[int]:
    """Node counts at each relative depth 1..levels below ``node_id`` (BFS-capped)."""
    counts = [0] * levels
    queue = [(node_id, 0)]
    seen = 0
    while queue and seen < 256:
        v, d = queue.pop(0)
        seen += 1
        for c in children_indices[v]:
            rd = depth[c] - depth[node_id]
            if 1 <= rd <= levels:
                counts[rd - 1] += 1
            if rd < levels:
                queue.append((c, d + 1))
    return counts


def _rho(h_node: torch.Tensor, manifold: PoincareBall) -> float:
    return float(manifold.c.sqrt() * h_node.norm())


def full_caption(node_id, graph, h_all, manifold, children_indices, parents, depth, lvl1):
    name = graph.id_to_entity[node_id]
    rho = _rho(h_all[node_id], manifold)
    d = depth[node_id]
    nch = len(children_indices[node_id])
    role = _role(node_id, children_indices, parents)
    branch = _branching(node_id, children_indices, depth)
    dom_ids = sorted(lvl1[node_id])
    domain = graph.id_to_entity[dom_ids[0]] if dom_ids else name
    return (
        f"Depth: {d}, rho: {rho:.2f}, branching: {branch}, role: {role}, "
        f"|children|: {nch}, domain: {domain}."
    )


def qa_pairs(node_id, graph, h_all, manifold, children_indices, parents, depth):
    """List of (question, answer) structural probes. Geometry stays in the ANSWER."""
    rho = _rho(h_all[node_id], manifold)
    d = depth[node_id]
    nch = len(children_indices[node_id])
    role = _role(node_id, children_indices, parents)
    abts = _abstraction(rho)
    return [
        ("How deep in the hierarchy is this node?", f"depth {d}"),
        ("Is this node a root, a junction, or a leaf?", role),
        ("How many direct children does this node have?", str(nch)),
        ("Is this concept broad, intermediate, or specific?", abts),
        ("What is the node's approximate hyperbolic radius?", f"{rho:.2f}"),
    ]


def main():
    p = argparse.ArgumentParser(description="Generate deterministic Map Reader SFT data")
    p.add_argument("--dataset", choices=["culturalbench", "wn18rr"], default="wn18rr")
    p.add_argument("--data_dir", default="data/wn18rr")
    p.add_argument("--embeddings", required=True, help=".pt of trained h_all on the ball")
    p.add_argument("--out", required=True, help="output JSONL")
    p.add_argument("--curvature", type=float, default=1.0)
    p.add_argument("--max_nodes", type=int, default=0, help="cap #nodes (0 = all)")
    p.add_argument("--qa_per_node", type=int, default=3,
                   help="how many 1B QA probes to sample per node (<=5)")
    p.add_argument("--no_decode", action="store_true", help="skip 1A caption records")
    p.add_argument("--no_qa", action="store_true", help="skip 1B QA records")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    if args.dataset == "wn18rr":
        from data.wordnet import load_wn18rr
        graph = load_wn18rr(data_dir=args.data_dir, split_seed=args.seed, leakage_safe=True)
    else:
        from data.culturalbench import load_culturalbench
        graph = load_culturalbench(split_seed=args.seed, leakage_safe=True)

    h_all = torch.load(args.embeddings, map_location="cpu")
    if not isinstance(h_all, torch.Tensor):
        h_all = h_all["h_all"]
    manifold = PoincareBall(c=args.curvature)

    ci = graph.children_indices
    parents = _parents_from_children(ci)
    depth = _depths(ci, parents)
    lvl1 = _level_k_ancestors(ci, parents, depth, 1)

    nodes = list(range(len(ci)))
    if args.max_nodes:
        random.shuffle(nodes)
        nodes = nodes[: args.max_nodes]

    n_dec, n_qa = 0, 0
    with open(args.out, "w", encoding="utf-8") as out:
        for node_id in nodes:
            if not args.no_decode:
                cap = full_caption(node_id, graph, h_all, manifold, ci, parents, depth, lvl1)
                out.write(json.dumps({
                    "node_id": node_id,
                    "kind": "decode",
                    "prompt": "Describe the structural position of the injected node.",
                    "target": cap,
                }, ensure_ascii=False) + "\n")
                n_dec += 1
            if not args.no_qa:
                pairs = qa_pairs(node_id, graph, h_all, manifold, ci, parents, depth)
                random.shuffle(pairs)
                for q, a in pairs[: max(0, args.qa_per_node)]:
                    out.write(json.dumps({
                        "node_id": node_id,
                        "kind": "qa",
                        "prompt": q,
                        "target": a,
                    }, ensure_ascii=False) + "\n")
                    n_qa += 1

    print(f"Done. Wrote {n_dec} decode + {n_qa} QA records to {args.out} "
          f"({len(nodes)} nodes).")


if __name__ == "__main__":
    main()
