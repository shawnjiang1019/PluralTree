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

from dataclasses import dataclass, field, replace

import torch
from torch import Tensor

from evaluation.intrinsic.branch_divergence import (
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
    pair_select: str = "maxw"  # "maxw" | "random" -- which child pairs survive
    #   The ONE knob of the selection ablation (merge_v2_divrand). "maxw" keeps
    #   the highest rel^alpha * W pairs (the divergence scout); "random" keeps a
    #   deterministic uniform sample of the SAME candidate pool, so the arms
    #   differ only in which pairs are chosen -- not in the graph, the anchors,
    #   the relevance gate, the rendering, or the fork COUNT (both truncate the
    #   identical pool at top_k). merge_v2_rand's failure was exactly the
    #   opposite: 1.00 forks/row vs merge_v2's 4.85, so volume moved with
    #   content and the arm measured nothing.
    pair_seed: int = 0         # seed folded into the per-(question, anchor) key


def _rand_rank(question: str, fork: "ScoredFork", seed: int) -> float:
    """Deterministic uniform in [0, 1) for one (question, anchor, pair).

    Hashed, not drawn from an RNG stream: a stream's output depends on the order
    in which pairs are enumerated, so a candidate-set change anywhere would
    reshuffle every later pick. Python's ``hash`` is salted per process
    (PYTHONHASHSEED), so blake2b -- runs must reproduce across processes.
    """
    import hashlib

    key = f"{question}|{fork.anchor}|{fork.branch_a}|{fork.branch_b}|{seed}"
    d = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "big") / 2 ** 64


def rank_key(question: str, cfg: ScoutConfig):
    """The sort key that decides which forks survive -- the ablated component.

    Both branches return a scalar the caller sorts descending and truncates at
    ``top_k``, so the SELECTION path is byte-identical between arms and only the
    ordering differs. Any other placement (sampling inside ``score_anchor``,
    say) would change how many forks each anchor contributes and break the
    volume match.
    """
    if cfg.pair_select == "random":
        return lambda f: _rand_rank(question, f, cfg.pair_seed)
    if cfg.pair_select != "maxw":
        raise ValueError(f"unknown pair_select {cfg.pair_select!r}")
    return lambda f: f.score


def selection_stats(forks: list["ScoredFork"], mode: str,
                    extra: dict | None = None) -> dict:
    """What the pair-selection ablation actually did, per row.

    The manipulation check for merge_v2_divrand, and it is NOT the one
    check_random_fork.py runs: that script was written for merge_v2_rand, which
    jumps to an UNRELATED anchor and must therefore be LESS relevant. divrand
    keeps the anchor pool and swaps only which child pair wins, so relevance
    should come out roughly UNCHANGED and the moving quantity is ``mean_w``:
    maxw takes the argmax, so a random draw over the same pool must sit below
    it. Equal mean_w means the ablation did nothing and the arm is void.

    ``anchors`` (of the SELECTED forks) and ``anchor_pool`` (every anchor
    examined) are both emitted because they answer different questions. The pool
    is identical between arms by construction; the selected set is not -- maxw
    concentrates on whichever anchor holds the widest pairs (measured on the
    fixture: 1 anchor vs random's 3), and reading that difference as "different
    anchors were searched" would be wrong.
    """
    # None, not NaN, for the undefined cases: these land in a JSONL row and
    # json.dumps writes a bare `NaN`, which is not valid JSON and breaks any
    # reader stricter than Python's own.
    n = len(forks)
    out = {"mode": mode, "n_forks": n,
           "mean_w": (sum(f.w for f in forks) / n) if n else None,
           "mean_relevance": (sum(f.relevance for f in forks) / n) if n else None,
           "anchors": sorted({f.anchor for f in forks})}
    out.update(extra or {})
    return out


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
    spectrum: list[int] = field(default_factory=list)
    #   Subgroup leaves for the full-distribution render, when they cannot be
    #   read off the anchor's children. Empty for scout forks (answer.py derives
    #   them from the hierarchy); populated by ``flat_forks``, whose whole point
    #   is that there is no hierarchy to read. Without it merge_v2_flat's third
    #   draft would silently fall back to the 2-pole block while merge_v2's
    #   rendered the spectrum — a render confound on top of the one being tested.


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
    stats: dict | None = None,
) -> list[ScoredFork]:
    """One-shot pipeline: question -> top-k relevant AND divergent forks.

    ``anchor_fn(question, graph, rel, max_anchors) -> list[int]`` is the LLM
    entity-extractor hook; defaults to ``lexical_anchors``. Pass ``q_emb`` to
    skip loading MiniLM (e.g. batch evaluation).

    ``cfg.pair_select="random"`` keeps the pipeline and swaps only the ranking
    (the selection ablation) — see ``rank_key``.

    ``stats``, if given, is filled with the anchor pool and the candidate-pair
    count. Both are already computed here; recomputing them in the caller would
    re-run anchor extraction per row, and leaving them out would make the
    divrand manipulation check unanswerable from the responses file.
    """
    cfg = cfg or ScoutConfig()
    if q_emb is None:
        q_emb = embed_question(question)
    rel = node_relevance(q_emb, text_feat)

    anchors = (anchor_fn or lexical_anchors)(question, graph, rel, cfg.max_anchors)
    forks: list[ScoredFork] = []
    for a in anchors:
        forks.extend(score_anchor(a, graph, h_all, rel, manifold, cfg))
    if stats is not None:
        stats.update(anchor_pool=list(anchors), n_candidates=len(forks),
                     pool_mean_w=(sum(f.w for f in forks) / len(forks))
                     if forks else None)
    forks.sort(key=rank_key(question, cfg), reverse=True)
    forks = forks[: cfg.top_k]
    for f in forks:
        _fill_top_pairs(f, h_all, rel, manifold, cfg)
    return forks


# ---------------------------------------------------------------------------
# Flat retrieval (hierarchy ablation)
# ---------------------------------------------------------------------------
def opinion_leaf_ids(graph) -> list[int]:
    """Every opinion leaf in the graph, WITHOUT consulting the tree structure.

    ``opinion_dist`` is a flat node->distribution map, so the first branch reads
    no hierarchy at all. The childless-node fallback does touch
    ``children_indices``, and that is the one unavoidable use in this condition:
    it identifies which nodes ARE opinion leaves, never which nodes belong
    together. Selection below ranks the resulting pool by question similarity
    only -- no parent, no subtree, no descent.
    """
    dist = getattr(graph, "opinion_dist", None)
    if dist:
        return sorted(dist.keys())
    kids = graph.children_indices
    return [v for v, cs in enumerate(kids) if not cs]


def flat_forks(
    question: str,
    graph,
    h_all: Tensor,
    text_feat: Tensor,
    manifold,
    cfg: ScoutConfig | None = None,
    q_emb: Tensor | None = None,
    stats: dict | None = None,
) -> list[ScoredFork]:
    """Pseudo-forks from a FLAT similarity search over opinion leaves.

    The hierarchy ablation (merge_v2_flat): no anchor descent, no child
    subtrees, no branch-level gate -- embed the question, rank every opinion
    leaf by MiniLM cosine, and pair the top ones. If merge_v2's gain survives
    this, the tree is not what produced it.

    SHAPE PARITY IS THE EXPERIMENT, same lesson as the random-fork control. The
    returned objects carry every field ``fork_context``/``fork_context_full``
    read (anchor, two branches, ``top_pairs``, ``spectrum``), so the injected
    text has merge_v2's structure and only its CONTENT SOURCE differs. Volume is
    matched the same way: ``top_k`` forks of ``top_pairs`` driver lines each,
    drawn from a pool of ``top_k * 2 * (1 + top_pairs)`` leaves.

    ``w`` is the geodesic between the two paired leaves (a 1-vs-1 Wasserstein).
    It only fills the rendered ``divergence=`` slot, and it is the geometry, not
    the hierarchy, so computing it honestly keeps the ablation single-factor.

    ``anchor`` is the top-ranked leaf of the pair: the header slot needs a node,
    and naming the parent would be reading the tree. It renders one distribution
    line where merge_v2 renders the survey-question node -- the only shape
    difference, and it costs the flat block a few characters, not a structure.
    """
    cfg = cfg or ScoutConfig()
    if q_emb is None:
        q_emb = embed_question(question)
    rel = node_relevance(q_emb, text_feat)

    per_fork = 2 * (1 + cfg.top_pairs)               # 2 poles + 2 per driver line
    leaves = opinion_leaf_ids(graph)
    # THE SAME RELEVANCE GATE `scout` APPLIES, and it is not optional. scout
    # gates branches at cfg.tau and returns [] when nothing clears it, so on a
    # question the graph cannot reach, merge_v2 falls back to plain-only drafts.
    # Ungated, flat would inject its top-40 leaves on that same question no
    # matter how irrelevant, and the arms would then differ in WHICH QUESTIONS
    # GET INJECTED rather than in hierarchy alone -- a confound between the
    # thing under test and abstention. Same class of error as merge_v2_rand's
    # volume mismatch, one level subtler.
    order = sorted((v for v in leaves if float(rel[v]) >= cfg.tau),
                   key=lambda v: float(rel[v]), reverse=True)
    pool = order[: cfg.top_k * per_fork]
    if len(pool) < 2:                                # nothing resolved -> abstain
        if stats is not None:
            stats.update(anchor_pool=[], n_candidates=len(pool), pool_mean_w=None)
        return []

    forks: list[ScoredFork] = []
    for start in range(0, len(pool), per_fork):
        chunk = pool[start:start + per_fork]
        if len(chunk) < 2:
            break                                    # a fork needs two positions
        a, b = chunk[0], chunk[1]
        idx = torch.tensor([a, b], dtype=torch.long)
        w = float(wasserstein(h_all[torch.tensor([a], dtype=torch.long)],
                              h_all[torch.tensor([b], dtype=torch.long)],
                              manifold))
        r = float(rel[idx].mean())
        rest = chunk[2:]
        drivers = [(rest[2 * i], rest[2 * i + 1], 0.0)
                   for i in range(len(rest) // 2)][: cfg.top_pairs]
        if not drivers:                              # thin pool: still render the
            drivers = [(a, b, 0.0)]                  # pair, so the block is intact
        forks.append(ScoredFork(
            anchor=a, branch_a=a, branch_b=b, w=w, relevance=r,
            score=max(r, 0.0) ** cfg.alpha * w,
            nodes_a=[a] + [na for na, _, _ in drivers],
            nodes_b=[b] + [nb for _, nb, _ in drivers],
            top_pairs=drivers, spectrum=list(chunk)))
        if len(forks) >= cfg.top_k:
            break
    # Same stats contract as ``scout`` so the caller stays retrieval-agnostic.
    # There is no anchor pool here -- that absence IS the condition.
    if stats is not None:
        stats.update(anchor_pool=[], n_candidates=len(pool), pool_mean_w=None)
    return forks


# ---------------------------------------------------------------------------
# Random-fork control (docs/random_fork_control.md)
# ---------------------------------------------------------------------------
_DEPTH_CACHE: dict = {}


def _node_depths(children_indices: list[list[int]]) -> list[int]:
    """Depth of every node, BFS from the root (entity 0). -1 if unreachable."""
    key = id(children_indices)
    if key in _DEPTH_CACHE:
        return _DEPTH_CACHE[key]
    depth = [-1] * len(children_indices)
    if depth:
        depth[0] = 0
        queue = [0]
        while queue:
            nxt = []
            for v in queue:
                for c in children_indices[v]:
                    if depth[c] < 0:
                        depth[c] = depth[v] + 1
                        nxt.append(c)
            queue = nxt
    _DEPTH_CACHE[key] = depth
    return depth


def _related(a: int, b: int, children_indices: list[list[int]],
             max_nodes: int) -> bool:
    """True if either anchor sits in the other's subtree -- i.e. not unrelated."""
    return (b in subtree_nodes(a, children_indices, max_nodes)
            or a in subtree_nodes(b, children_indices, max_nodes))


def random_forks(
    real: ScoredFork,
    graph,
    h_all: Tensor,
    rel: Tensor,
    manifold,
    cfg: ScoutConfig,
    n_candidates: int = 12,
    seed: int = 0,
) -> list[ScoredFork]:
    """Structurally comparable forks from UNRELATED parts of the graph.

    The control arm for "does the graph supply CONTENT, or just variance?"
    (docs/random_fork_control.md). Every current result is consistent with the
    graph acting as a randomizer: `scout` 0.3927 and `distributional` 0.3941 both
    LOSE to baseline 0.4967, and they differ from each other by +0.0014 against a
    0.027 noise floor -- two very different payloads, indistinguishable outcomes.

    MATCHING IS THE EXPERIMENT. A random fork that is shorter, shallower, or has
    fewer branches changes prompt length and position count at the same time as
    relevance, and then the arm measures nothing. Candidates are therefore drawn
    at the SAME DEPTH with a comparable child count, and the caller picks among
    them on rendered length (`answer._matched_random_fork`).

    Relevance and W are recomputed honestly for each candidate rather than copied
    from ``real`` -- the logged `g_relevance` should show these are irrelevant,
    and that is the manipulation check.

    Returns [] if the graph offers no comparable unrelated anchor, which the
    caller must treat as "skip this question", never as "inject nothing".
    """
    import random as _random

    kids = graph.children_indices
    depth = _node_depths(kids)
    d_real = depth[real.anchor]
    n_kids_real = len(kids[real.anchor])

    pool = [v for v in range(len(kids))
            if len(kids[v]) >= 2 and depth[v] == d_real and v != real.anchor]
    # Same depth AND a comparable branching factor, so the rendered fork has a
    # similar number of positions to choose between.
    tight = [v for v in pool if abs(len(kids[v]) - n_kids_real) <= 1]
    pool = tight or pool
    if not pool:
        return []

    rng = _random.Random(seed)
    rng.shuffle(pool)
    out: list[ScoredFork] = []
    for a in pool:
        if len(out) >= n_candidates:
            break
        if _related(a, real.anchor, kids, cfg.max_nodes):
            continue
        # score_anchor applies the SAME relevance gate; a genuinely unrelated
        # anchor will usually fail it, so score the pair directly instead.
        cand = _unscored_fork(a, graph, h_all, rel, manifold, cfg)
        if cand is not None:
            out.append(cand)
    return out


def _unscored_fork(anchor: int, graph, h_all: Tensor, rel: Tensor, manifold,
                   cfg: ScoutConfig) -> ScoredFork | None:
    """The most divergent child pair of ``anchor``, WITHOUT the relevance gate.

    `score_anchor` drops branches below `cfg.tau`, which by construction removes
    every unrelated anchor -- exactly the ones this control needs. The clouds are
    still capped and mass-weighted the same way, so the only difference from the
    real path is that irrelevance does not disqualify.
    """
    kids = graph.children_indices[anchor][: cfg.max_children]
    clouds = [(c, subtree_nodes(c, graph.children_indices, cfg.max_nodes))
              for c in kids]
    clouds = [(c, n) for c, n in clouds if len(n) >= 1]
    if len(clouds) < 2:
        return None
    best = None
    for i in range(len(clouds)):
        for j in range(i + 1, len(clouds)):
            ca, na = clouds[i]
            cb, nb = clouds[j]
            P = h_all[torch.tensor(na, dtype=torch.long)]
            Q = h_all[torch.tensor(nb, dtype=torch.long)]
            w = wasserstein(P, Q, manifold,
                            weights=(_cloud_mass(na, rel, cfg.temp),
                                     _cloud_mass(nb, rel, cfg.temp)))
            if w != w:                                   # NaN guard
                continue
            if best is None or w > best[0]:
                best = (w, ca, cb, na, nb)
    if best is None:
        return None
    w, ca, cb, na, nb = best
    idx = torch.tensor(na + nb, dtype=torch.long)
    r = float(rel[idx].mean())
    fork = ScoredFork(anchor=anchor, branch_a=ca, branch_b=cb, w=float(w),
                      relevance=r, score=max(r, 0.0) ** cfg.alpha * float(w),
                      nodes_a=na, nodes_b=nb)
    _fill_top_pairs(fork, h_all, rel, manifold, cfg)
    return fork


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
        from data.loaders.globalopinionqa import compute_features
        feat = compute_features(graph).cpu()
    else:
        from data.loaders.culturalbench import compute_text_embeddings
        feat = compute_text_embeddings(graph).cpu()
    if path:
        torch.save(feat, path)
    return feat


# ---------------------------------------------------------------------------
# Self-test (no endpoint, no MiniLM: q_emb and text_feat are supplied)
# ---------------------------------------------------------------------------
def _toy_graph(n_anchors: int = 2, n_kids: int = 4):
    """Two anchors x n_kids opinion leaves -- enough pairs to exceed top_k.

    The pool must be LARGER than top_k or the count assertion is vacuous: any
    selection rule returns the whole pool when the pool is short, which is
    exactly how a broken ablation would still look volume-matched.
    """
    import types

    n = 1 + n_anchors + n_anchors * n_kids
    kids: list[list[int]] = [[] for _ in range(n)]
    kids[0] = list(range(1, n_anchors + 1))
    leaves = []
    nxt = n_anchors + 1
    for a in kids[0]:
        kids[a] = list(range(nxt, nxt + n_kids))
        leaves.extend(kids[a])
        nxt += n_kids

    feat = torch.zeros(n, 4)
    feat[:, 0] = 1.0
    feat[:, 1] = torch.linspace(0.05, 0.5, n)     # distinct but all on-topic
    torch.manual_seed(0)
    g = types.SimpleNamespace(
        children_indices=kids,
        id_to_entity=[f"n{i}" for i in range(n)],
        entity_text={i: f"node {i}" for i in range(n)},
        opinion_texts={v: ["Q yes", "Q no"] for v in leaves},
        opinion_dist={v: [0.5, 0.5] for v in leaves})
    return g, torch.randn(n, 8) * 0.1, feat, torch.tensor([1.0, 0.0, 0.0, 0.0]), leaves


def _selftest() -> None:
    """The two ablations must differ from merge_v2 in ONE thing each.

    Volume and reproducibility are asserted, not assumed: merge_v2_rand shipped
    for four run versions producing 1.00 forks/row against merge_v2's 4.85, and
    every comparison drawn from it was uninterpretable.
    """
    g, h, feat, q, leaves = _toy_graph()

    def anchors(question, graph, rel, max_anchors):
        return [1, 2]

    q_text = "does the geometry help?"
    base = ScoutConfig(top_k=5, top_pairs=3)
    rnd = ScoutConfig(top_k=5, top_pairs=3, pair_select="random")

    f_max = scout(q_text, g, h, feat, None, cfg=base, anchor_fn=anchors, q_emb=q)
    f_rnd = scout(q_text, g, h, feat, None, cfg=rnd, anchor_fn=anchors, q_emb=q)
    pool = score_anchor(1, g, h, node_relevance(q, feat), None, base) + \
        score_anchor(2, g, h, node_relevance(q, feat), None, base)
    assert len(pool) > base.top_k, len(pool)          # the count test must bite
    assert len(f_max) == len(f_rnd) == base.top_k, (len(f_max), len(f_rnd))

    key = lambda fs: [(f.anchor, f.branch_a, f.branch_b) for f in fs]
    assert key(f_max) != key(f_rnd), "random selection is inert -- same pairs"
    assert sorted(key(f_max)) != sorted(key(f_rnd)), "same SET, only reordered"
    assert [f.score for f in f_max] == sorted((f.score for f in f_max),
                                              reverse=True), "maxw path changed"

    # Reproducibility: same seed -> same pairs, different seed -> different ones.
    again = scout(q_text, g, h, feat, None, cfg=rnd, anchor_fn=anchors, q_emb=q)
    assert key(again) == key(f_rnd), "random selection is not reproducible"
    other = scout(q_text, g, h, feat, None, anchor_fn=anchors, q_emb=q,
                  cfg=ScoutConfig(top_k=5, top_pairs=3, pair_select="random",
                                  pair_seed=1))
    assert key(other) != key(f_rnd), "pair_seed does not move the sample"
    # Per QUESTION as well as per anchor -- a key that ignored the question would
    # inject the identical fork set into every row of the arm.
    other_q = scout("a different question", g, h, feat, None, cfg=rnd,
                    anchor_fn=anchors, q_emb=q)
    assert key(other_q) != key(f_rnd), "selection ignores the question"

    try:
        scout(q_text, g, h, feat, None, anchor_fn=anchors, q_emb=q,
              cfg=ScoutConfig(pair_select="nope"))
        raise AssertionError("unknown pair_select must fail loudly")
    except ValueError:
        pass

    # Flat retrieval: volume-matched, hierarchy-free, renderable.
    fcfg = ScoutConfig(top_k=2, top_pairs=1)
    flat = flat_forks(q_text, g, h, feat, None, cfg=fcfg, q_emb=q)
    assert len(flat) == fcfg.top_k, len(flat)
    assert all(f.branch_a != f.branch_b for f in flat), "degenerate pair"
    assert all(len(f.top_pairs) == fcfg.top_pairs for f in flat), "driver count"
    assert all(f.spectrum for f in flat), "no spectrum -> full_dist would differ"
    picked = [v for f in flat for v in f.spectrum]
    assert len(set(picked)) == len(picked), "a leaf was injected twice"
    rel = node_relevance(q, feat)
    top = sorted(leaves, key=lambda v: float(rel[v]), reverse=True)[:len(picked)]
    assert set(picked) == set(top), "flat retrieval is not similarity-ranked"
    assert all(v in leaves for v in picked), "flat retrieval left the leaf set"

    # The arms must ABSTAIN TOGETHER. scout gates branches at cfg.tau, so on an
    # unreachable question merge_v2 falls back to plain-only drafts. An ungated
    # flat would still inject its top-k leaves there, and the two arms would
    # then differ in which questions got injected at all -- confounding the
    # hierarchy test with abstention. This asserts the gate is really applied.
    shut = replace(fcfg, tau=1.01)                   # above any cosine
    assert scout("which fork helps?", g, h, feat, None, cfg=shut, q_emb=q) == []
    assert flat_forks("which fork helps?", g, h, feat, None,
                      cfg=shut, q_emb=q) == [], "flat ignored the relevance gate"

    print("scout self-test OK")
    print(f"  pair_select: maxw {key(f_max)} vs random {key(f_rnd)} "
          f"-- {len(f_max)} forks either way from a pool of {len(pool)}")
    print(f"  flat       : {len(flat)} forks, {fcfg.top_pairs} driver pair(s) "
          f"each, {len(picked)} leaves by similarity alone (no descent)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main():
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ap = argparse.ArgumentParser(description="One-shot relevant+divergent fork scout")
    ap.add_argument("--selftest", action="store_true",
                    help="check the selection/hierarchy ablations offline")
    ap.add_argument("--embeddings", help=".pt of h_all on the ball")
    ap.add_argument("--dataset", choices=["wn18rr", "culturalbench",
                                          "globalopinionqa", "grailqa", "opinionqa"],
                    default="globalopinionqa")
    ap.add_argument("--data_dir", default="data/wn18rr")
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--question", default=None)
    ap.add_argument("--text_feat", default=None,
                    help="cache .pt for MiniLM node features (computed if missing)")
    ap.add_argument("--anchors", default=None,
                    help="comma-separated entity names: bypass extraction")
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--pair_select", choices=["maxw", "random"], default="maxw",
                    help="random = the selection ablation (merge_v2_divrand)")
    ap.add_argument("--flat", action="store_true",
                    help="flat similarity retrieval, no hierarchy (merge_v2_flat)")
    ap.add_argument("--prompt", action="store_true",
                    help="also print the format_for_prompt injection block")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    for req in ("embeddings", "question"):
        if not getattr(args, req):
            ap.error(f"--{req} is required (or pass --selftest)")

    from pluraltree.manifolds.poincare import PoincareBall
    if args.dataset == "wn18rr":
        from data.loaders.wordnet import load_wn18rr
        graph = load_wn18rr(data_dir=args.data_dir, split_seed=args.seed,
                            leakage_safe=True)
    elif args.dataset == "globalopinionqa":
        from data.loaders.globalopinionqa import load_globalopinionqa
        graph = load_globalopinionqa(split_seed=args.seed, leakage_safe=True)
    elif args.dataset == "grailqa":
        from data.loaders.grailqa import load_grailqa
        graph = load_grailqa(split_seed=args.seed, leakage_safe=True)
    elif args.dataset == "opinionqa":
        from data.loaders.opinionqa import load_opinionqa
        graph = load_opinionqa(split_seed=args.seed, leakage_safe=True)
    else:
        from data.loaders.culturalbench import load_culturalbench
        graph = load_culturalbench(split_seed=args.seed, leakage_safe=True)

    h_all = torch.load(args.embeddings, map_location="cpu")
    if not isinstance(h_all, torch.Tensor):
        h_all = h_all["h_all"]
    manifold = PoincareBall(c=args.curvature)
    text_feat = load_or_compute_text_feat(graph, args.dataset, args.text_feat)

    cfg = ScoutConfig(tau=args.tau, alpha=args.alpha, temp=args.temp,
                      top_k=args.top, pair_select=args.pair_select)

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

    if args.flat:
        forks = flat_forks(args.question, graph, h_all, text_feat, manifold, cfg=cfg)
    else:
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
