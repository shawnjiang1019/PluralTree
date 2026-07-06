"""One-shot scout: retrieve forks that are both RELEVANT and DIVERGENT.

Given a question, find the parents in the knowledge graph whose child branches
(a) matter to the question and (b) genuinely disagree, and package them for
injection into an LLM prompt. See docs/scout_design.txt.

The Wasserstein scout alone optimizes divergence, not relevance — the most
divergent forks may be off-topic. Three guards are wired in:

  1. relevance GATE: subtree clouds are pruned to nodes whose MiniLM-cosine
     with the question exceeds ``tau`` before any OT runs;
  2. question-conditioned OT: the transport mass is ``softmax(rel / temp)``,
     so W is dominated by on-topic nodes;
  3. combined score ``rel^alpha * W`` instead of W alone.

Two geometries, on purpose: relevance is cosine in MiniLM TEXT space (the
question does not live on the ball); divergence is Wasserstein under the
LEARNED Poincare geodesic over the frozen ``h_all``.

Usage:
    python -m retrieval.scout --embeddings embeddings_goqa.pt \
        --dataset globalopinionqa --curvature 0.5 --text_feat feats_goqa.pt \
        --question "How much of a danger is North Korea?"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from evaluation.branch_divergence import (
    subtree_nodes,
    wasserstein,
    _ground_cost,
    _transport_plan,
)

MINILM = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Config / result containers
# ---------------------------------------------------------------------------
@dataclass
class ScoutConfig:
    tau: float = 0.25          # relevance gate (cosine); 0 disables the gate
    alpha: float = 1.0         # relevance exponent in the score; 0 = pure divergence
    temp: float = 0.1          # softmax temperature for OT mass; large = uniform
    max_anchors: int = 4       # anchors examined per question
    max_children: int = 8      # children per anchor (bounds O(c^2) pairs)
    max_nodes: int = 32        # subtree cloud cap (bounds OT size)
    min_keep: int = 2          # gate floor: never shrink a cloud below this
    top_k: int = 5             # forks returned
    top_pairs: int = 3         # transport pairs reported per fork


@dataclass
class ScoredFork:
    """One (anchor, branch_a, branch_b) fork with its relevance and divergence."""

    anchor: int
    branch_a: int
    branch_b: int
    w: float                   # Wasserstein between the gated clouds
    relevance: float           # mean question-relevance of the two clouds
    score: float               # max(relevance, 0)^alpha * w
    nodes_a: list[int] = field(default_factory=list)   # gated cloud (root first)
    nodes_b: list[int] = field(default_factory=list)
    top_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    #                          ^ (node_a, node_b, gamma*C) — drivers of the split


# ---------------------------------------------------------------------------
# Relevance (MiniLM text space)
# ---------------------------------------------------------------------------
_ENCODER_CACHE: dict = {}


def embed_question(question: str, model_name: str = MINILM,
                   device: str = "cpu") -> Tensor:
    """(d,) MiniLM embedding of the question — same space as the node features.

    CPU by default: this often runs on a node whose GPUs are filled by a
    colocated vLLM server, and MiniLM on CPU is milliseconds per question.
    The model is cached across calls (it was being re-loaded per question).
    """
    key = (model_name, device)
    if key not in _ENCODER_CACHE:
        from sentence_transformers import SentenceTransformer
        _ENCODER_CACHE[key] = SentenceTransformer(model_name, device=device)
    enc = _ENCODER_CACHE[key].encode(question, convert_to_numpy=True)
    return torch.as_tensor(enc).float()


def node_relevance(q_emb: Tensor, text_feat: Tensor) -> Tensor:
    """(N,) cosine similarity between the question and every node's text feature."""
    q = q_emb / q_emb.norm().clamp_min(1e-9)
    f = text_feat / text_feat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return f @ q


# ---------------------------------------------------------------------------
# Anchor extraction
# ---------------------------------------------------------------------------
def lexical_anchors(
    question: str,
    graph,
    rel: Tensor,
    max_anchors: int = 4,
) -> list[int]:
    """Internal nodes whose entity name appears in the question (hashmap lookup).

    Longest names match first (so "South Korea" beats "Korea"). A matched LEAF
    is replaced by its parent — forks live at internal nodes. If nothing
    matches lexically, fall back to the ``max_anchors`` most question-relevant
    internal nodes, so the scout never returns empty for want of exact wording.

    This is the slot an LLM entity extractor plugs into (see ``scout``'s
    ``anchor_fn``): the LLM emits entity strings, the same hashmap resolves them.
    """
    ql = question.lower()
    internal = {v for v in range(len(graph.children_indices))
                if graph.children_indices[v]}
    parents: dict[int, int] = {}
    for p in internal:
        for c in graph.children_indices[p]:
            parents[c] = p

    hits: list[int] = []
    names = sorted(enumerate(graph.id_to_entity), key=lambda t: -len(t[1]))
    for nid, name in names:
        if len(name) < 3 or name.lower() not in ql:
            continue
        a = nid if nid in internal else parents.get(nid)
        if a is not None and a not in hits:
            hits.append(a)
        if len(hits) >= max_anchors:
            return hits
    if hits:
        return hits

    # Lexical miss: most relevant internal nodes by MiniLM cosine.
    order = torch.argsort(rel, descending=True).tolist()
    return [v for v in order if v in internal][:max_anchors]


# ---------------------------------------------------------------------------
# Relevance-gated clouds
# ---------------------------------------------------------------------------
def gated_cloud(nodes: list[int], rel: Tensor, cfg: ScoutConfig) -> list[int]:
    """Prune a subtree cloud to on-topic nodes (root always kept, floor min_keep)."""
    kept = [nodes[0]] + [v for v in nodes[1:] if float(rel[v]) >= cfg.tau]
    if len(kept) < cfg.min_keep:
        rest = sorted(nodes[1:], key=lambda v: float(rel[v]), reverse=True)
        for v in rest:
            if v not in kept:
                kept.append(v)
            if len(kept) >= cfg.min_keep:
                break
    return kept


def _cloud_mass(nodes: list[int], rel: Tensor, temp: float) -> Tensor:
    """OT mass over a cloud: softmax(rel / temp) — on-topic nodes carry the weight."""
    r = rel[torch.tensor(nodes, dtype=torch.long)]
    return torch.softmax(r / max(temp, 1e-6), dim=0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_anchor(
    anchor: int,
    graph,
    h_all: Tensor,
    rel: Tensor,
    manifold,
    cfg: ScoutConfig,
) -> list[ScoredFork]:
    """All child-pair forks of one anchor, relevance-gated and scored."""
    kids = graph.children_indices[anchor][: cfg.max_children]
    clouds: list[tuple[int, list[int]]] = []
    for c in kids:
        nodes = gated_cloud(subtree_nodes(c, graph.children_indices, cfg.max_nodes),
                            rel, cfg)
        mean_rel = float(rel[torch.tensor(nodes, dtype=torch.long)].mean())
        if mean_rel >= cfg.tau:                      # branch-level relevance gate
            clouds.append((c, nodes))
    if len(clouds) < 2:
        return []

    forks: list[ScoredFork] = []
    for i in range(len(clouds)):
        for j in range(i + 1, len(clouds)):
            ca, na = clouds[i]
            cb, nb = clouds[j]
            P = h_all[torch.tensor(na, dtype=torch.long)]
            Q = h_all[torch.tensor(nb, dtype=torch.long)]
            a = _cloud_mass(na, rel, cfg.temp)
            b = _cloud_mass(nb, rel, cfg.temp)
            w = wasserstein(P, Q, manifold, weights=(a, b))
            r_pair = float((rel[torch.tensor(na, dtype=torch.long)].mean()
                            + rel[torch.tensor(nb, dtype=torch.long)].mean()) / 2)
            score = max(r_pair, 0.0) ** cfg.alpha * w
            forks.append(ScoredFork(anchor=anchor, branch_a=ca, branch_b=cb,
                                    w=w, relevance=r_pair, score=score,
                                    nodes_a=na, nodes_b=nb))
    return forks


def _fill_top_pairs(fork: ScoredFork, h_all: Tensor, rel: Tensor,
                    manifold, cfg: ScoutConfig) -> None:
    """Attach the transport pairs (gamma*C) that drive this fork's divergence."""
    P = h_all[torch.tensor(fork.nodes_a, dtype=torch.long)]
    Q = h_all[torch.tensor(fork.nodes_b, dtype=torch.long)]
    a = _cloud_mass(fork.nodes_a, rel, cfg.temp)
    b = _cloud_mass(fork.nodes_b, rel, cfg.temp)
    C = _ground_cost(P, Q, manifold)
    gamma = _transport_plan(a, b, C)
    cost = gamma * C
    flat = cost.flatten()
    k = min(cfg.top_pairs, flat.numel())
    top = torch.topk(flat, k)
    m = C.shape[1]
    fork.top_pairs = [
        (fork.nodes_a[int(ix) // m], fork.nodes_b[int(ix) % m], float(v))
        for v, ix in zip(top.values, top.indices)
    ]


def scout(
    question: str,
    graph,
    h_all: Tensor,
    text_feat: Tensor,
    manifold,
    cfg: ScoutConfig | None = None,
    anchor_fn=None,
    q_emb: Tensor | None = None,
) -> list[ScoredFork]:
    """One-shot pipeline: question -> top-k relevant AND divergent forks.

    ``anchor_fn(question, graph, rel, max_anchors) -> list[int]`` is the LLM
    entity-extractor hook; defaults to ``lexical_anchors``. Pass ``q_emb`` to
    skip loading MiniLM (e.g. batch evaluation).
    """
    cfg = cfg or ScoutConfig()
    if q_emb is None:
        q_emb = embed_question(question)
    rel = node_relevance(q_emb, text_feat)

    anchors = (anchor_fn or lexical_anchors)(question, graph, rel, cfg.max_anchors)
    forks: list[ScoredFork] = []
    for a in anchors:
        forks.extend(score_anchor(a, graph, h_all, rel, manifold, cfg))
    forks.sort(key=lambda f: f.score, reverse=True)
    forks = forks[: cfg.top_k]
    for f in forks:
        _fill_top_pairs(f, h_all, rel, manifold, cfg)
    return forks


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _opinion_prefix(texts: list[str]) -> str:
    """Shared '{question} ' prefix of an opinion's option strings."""
    import os

    pref = os.path.commonprefix(texts)
    cut = pref.rfind(" ")
    return pref[: cut + 1] if cut > 0 else pref


def describe_node(graph, nid: int, width: int = 120, show_q: bool = False) -> str:
    """Human-readable node line; opinion leaves get their answer distribution.

    ``width=0`` disables truncation. ``show_q=True`` prefixes opinion leaves
    with their survey question (drivers can come from different questions).
    """
    texts = getattr(graph, "opinion_texts", {}).get(nid)
    dist = getattr(graph, "opinion_dist", {}).get(nid)
    if texts and dist:
        pref = _opinion_prefix(texts)
        name = graph.id_to_entity[nid]
        if name.startswith("op:"):                   # opinionqa: op:{qkey}:{attr}:{group}
            _, attr, group = name.rsplit(":", 2)
            who = f"{group} ({attr})"
        else:                                        # goqa: op_{row}_{country}
            who = name.split("_", 2)[-1]
        opts = ", ".join(f"\"{t[len(pref):].strip()}\" {p:.0%}"
                         for t, p in zip(texts, dist) if p >= 0.05)
        head = f"{who} re \"{pref.strip()}\"" if show_q else f"{who} answered"
        return f"{head}: {opts}"
    s = graph.entity_text.get(nid) or graph.id_to_entity[nid]
    if width and len(s) > width:
        return s[: width - 1] + "…"
    return s


# ---------------------------------------------------------------------------
# Prompt injection
# ---------------------------------------------------------------------------
def format_for_prompt(forks: list[ScoredFork], graph, width: int = 90) -> str:
    """Render forks as plain text for injection into an LLM prompt."""

    def label(nid: int) -> str:
        s = graph.entity_text.get(nid) or graph.id_to_entity[nid]
        return s if len(s) <= width else s[: width - 1] + "…"

    lines = ["Divergent perspectives found in the knowledge graph:"]
    for k, f in enumerate(forks, 1):
        lines.append(f"\n[{k}] At '{label(f.anchor)}', two branches disagree "
                     f"(divergence={f.w:.2f}, relevance={f.relevance:.2f}):")
        lines.append(f"    A: {label(f.branch_a)}")
        lines.append(f"    B: {label(f.branch_b)}")
        for na, nb, c in f.top_pairs:
            lines.append(f"    A::{label(na)}  <->  B::{label(nb)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Text-feature cache
# ---------------------------------------------------------------------------
def load_or_compute_text_feat(graph, dataset: str, path: str | None) -> Tensor:
    """Node features in MiniLM space, cached to ``path`` (they are slow to build)."""
    import os

    if path and os.path.exists(path):
        return torch.load(path, map_location="cpu")
    if dataset in ("globalopinionqa", "opinionqa"):
        from data.globalopinionqa import compute_features
        feat = compute_features(graph).cpu()
    else:
        from data.culturalbench import compute_text_embeddings
        feat = compute_text_embeddings(graph).cpu()
    if path:
        torch.save(feat, path)
    return feat


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main():
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ap = argparse.ArgumentParser(description="One-shot relevant+divergent fork scout")
    ap.add_argument("--embeddings", required=True, help=".pt of h_all on the ball")
    ap.add_argument("--dataset", choices=["wn18rr", "culturalbench",
                                          "globalopinionqa", "grailqa", "opinionqa"],
                    default="globalopinionqa")
    ap.add_argument("--data_dir", default="data/wn18rr")
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--question", required=True)
    ap.add_argument("--text_feat", default=None,
                    help="cache .pt for MiniLM node features (computed if missing)")
    ap.add_argument("--anchors", default=None,
                    help="comma-separated entity names: bypass extraction")
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--prompt", action="store_true",
                    help="also print the format_for_prompt injection block")
    args = ap.parse_args()

    from pluraltree.manifolds.poincare import PoincareBall
    if args.dataset == "wn18rr":
        from data.wordnet import load_wn18rr
        graph = load_wn18rr(data_dir=args.data_dir, split_seed=args.seed,
                            leakage_safe=True)
    elif args.dataset == "globalopinionqa":
        from data.globalopinionqa import load_globalopinionqa
        graph = load_globalopinionqa(split_seed=args.seed, leakage_safe=True)
    elif args.dataset == "grailqa":
        from data.grailqa import load_grailqa
        graph = load_grailqa(split_seed=args.seed, leakage_safe=True)
    elif args.dataset == "opinionqa":
        from data.opinionqa import load_opinionqa
        graph = load_opinionqa(split_seed=args.seed, leakage_safe=True)
    else:
        from data.culturalbench import load_culturalbench
        graph = load_culturalbench(split_seed=args.seed, leakage_safe=True)

    h_all = torch.load(args.embeddings, map_location="cpu")
    if not isinstance(h_all, torch.Tensor):
        h_all = h_all["h_all"]
    manifold = PoincareBall(c=args.curvature)
    text_feat = load_or_compute_text_feat(graph, args.dataset, args.text_feat)

    cfg = ScoutConfig(tau=args.tau, alpha=args.alpha, temp=args.temp,
                      top_k=args.top)

    anchor_fn = None
    if args.anchors:
        wanted = [s.strip() for s in args.anchors.split(",") if s.strip()]
        name_to_id = {n.lower(): i for i, n in enumerate(graph.id_to_entity)}

        def anchor_fn(question, g, rel, max_anchors, _w=wanted, _m=name_to_id):
            ids = [_m[w.lower()] for w in _w if w.lower() in _m]
            missing = [w for w in _w if w.lower() not in _m]
            if missing:
                print(f"  (unresolved anchors: {missing})")
            return ids[:max_anchors]

    forks = scout(args.question, graph, h_all, text_feat, manifold,
                  cfg=cfg, anchor_fn=anchor_fn)

    print(f"Q: {args.question}")
    if not forks:
        print("no forks passed the relevance gate — lower --tau or add --anchors")
        return
    print(f"top {len(forks)} forks (score = rel^{cfg.alpha} * W, tau={cfg.tau}):")
    for f in forks:
        print(f"\n[{f.anchor}] anchor: {describe_node(graph, f.anchor, 0)}")
        print(f"  branch A: {describe_node(graph, f.branch_a, 0)}")
        print(f"  branch B: {describe_node(graph, f.branch_b, 0)}")
        print(f"  score={f.score:.3f}  W={f.w:.3f}  rel={f.relevance:.3f}")
        if f.top_pairs:
            print("  drivers:")
            for na, nb, c in f.top_pairs:
                print(f"    A: {describe_node(graph, na, 0, show_q=True)}")
                print(f"    B: {describe_node(graph, nb, 0, show_q=True)}")
    if args.prompt:
        print("\n" + format_for_prompt(forks, graph))


if __name__ == "__main__":
    _main()
