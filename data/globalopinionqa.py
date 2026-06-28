"""GlobalOpinionQA dataset loader.

Builds a knowledge graph from Anthropic/llm_global_opinions (Durmus et al.,
"Towards Measuring the Representation of Subjective Global Opinions in LMs").

Unlike WN18RR (which ships as triples), GlobalOpinionQA is a flat survey: each
row is a question + answer options + a per-country distribution over those
options. We *impose* the geographic hierarchy and put the answer distribution
into the geometry, so that branch divergence can actually fire:

    World -> Region -> Country -> Opinion (leaf)

The key difference from CulturalBench:
  * Country is an INTERNAL node whose subtree is its cloud of opinion vectors
    (its "opinion profile") -- not a singleton-leaf grouping.
  * The Opinion leaf's feature is the *distribution-weighted* option embedding
    (see ``compute_features``), so two countries that answer the same question
    differently land at different points. That distributional signal is what a
    fork detector needs; CulturalBench had none (single-answer benchmark).

Relations:
    answered_by : opinion -> country
    located_in  : country -> region
    part_of     : region  -> world
"""

from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass, field

import torch
from torch import Tensor

from data.culturalbench import (
    CulturalGraph,
    REGION_TO_COUNTRIES,
    COUNTRY_TO_REGION,
    COUNTRY_ALIASES,
)

RELATION_VOCAB: dict[str, int] = {
    "answered_by": 0,
    "located_in":  1,
    "part_of":     2,
}
NUM_RELATIONS = len(RELATION_VOCAB)


# ---------------------------------------------------------------------------
# Country-name normalization (Pew/WVS names -> our canonical country set)
# ---------------------------------------------------------------------------
_ALIAS_TO_CANON: dict[str, str] = {}
for _canon, _aliases in COUNTRY_ALIASES.items():
    for _a in _aliases:
        _ALIAS_TO_CANON[_a.lower()] = _canon
_CANON_BY_LOWER: dict[str, str] = {c.lower(): c for c in COUNTRY_TO_REGION}

# GlobalOpinionQA uses Pew survey shorthands that the alias table misses.
_PEW_FIXES: dict[str, str] = {
    "s. korea":      "South Korea",
    "czech rep.":    "Czech Republic",
    "u.s.":          "United States",
    "great britain": "United Kingdom",
}


def _resolve_country(name: str) -> str | None:
    """Map a survey country label to our canonical name, or None if unknown."""
    if not name:
        return None
    k = name.strip().lower()
    if k in _CANON_BY_LOWER:
        return _CANON_BY_LOWER[k]
    if k in _ALIAS_TO_CANON:
        return _ALIAS_TO_CANON[k]
    return _PEW_FIXES.get(k)


def _parse_options(raw) -> list[str]:
    """Parse the ``options`` field (a stringified Python list) into a list."""
    if isinstance(raw, (list, tuple)):
        return [str(o) for o in raw]
    try:
        val = ast.literal_eval(str(raw))
        return [str(o) for o in val] if isinstance(val, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def _parse_selections(raw) -> dict[str, list[float]]:
    """Parse the ``selections`` field (often ``defaultdict(<class 'list'>, {...})``)."""
    if isinstance(raw, dict):
        return raw
    m = re.search(r"\{.*\}", str(raw), re.DOTALL)
    if not m:
        return {}
    try:
        val = ast.literal_eval(m.group(0))
        return val if isinstance(val, dict) else {}
    except (ValueError, SyntaxError):
        return {}


# ---------------------------------------------------------------------------
# Data container (CulturalGraph + the per-opinion distribution needed for features)
# ---------------------------------------------------------------------------
@dataclass
class GlobalOpinionGraph(CulturalGraph):
    # opinion node id -> ["{question} {option}" for each option]
    opinion_texts: dict[int, list[str]] = field(default_factory=dict)
    # opinion node id -> normalized probability over those options
    opinion_dist:  dict[int, list[float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Builder (network-free; takes already-parsed rows so it is unit-testable)
# ---------------------------------------------------------------------------
def _build_graph(
    rows,
    *,
    split_seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    leakage_safe: bool = True,
) -> GlobalOpinionGraph:
    entity_vocab: dict[str, int] = {}
    id_to_entity: list[str] = []
    entity_text:  dict[int, str] = {}
    entity_types: dict[int, str] = {}
    opinion_texts: dict[int, list[str]] = {}
    opinion_dist:  dict[int, list[float]] = {}

    def add_entity(name: str, text: str, etype: str) -> int:
        if name not in entity_vocab:
            eid = len(id_to_entity)
            entity_vocab[name] = eid
            id_to_entity.append(name)
            entity_text[eid] = text
            entity_types[eid] = etype
        return entity_vocab[name]

    world_id = add_entity("World", "World — the global community of all nations and cultures", "world")

    # --- parse rows into opinion cells; add region/country/opinion on demand ---
    opinion_cells: list[tuple[int, int]] = []     # (opinion_id, country_id)
    skipped_countries: set[str] = set()
    n_rows_used = 0
    for ri, row in enumerate(rows):
        options = _parse_options(row.get("options"))
        sel = _parse_selections(row.get("selections"))
        q = (row.get("question") or "").strip()
        if not options or not sel or not q:
            continue
        used = False
        for cname, probs in sel.items():
            canon = _resolve_country(cname)
            if canon is None:
                skipped_countries.add(str(cname))
                continue
            if not isinstance(probs, (list, tuple)) or len(probs) != len(options):
                continue
            total = float(sum(probs))
            if total <= 0:
                continue
            probs = [float(p) / total for p in probs]

            region = COUNTRY_TO_REGION[canon]
            add_entity(region, f"{region.replace('_', ' ')} — geographic and cultural region", "region")
            country_id = add_entity(canon, f"{canon} — country in {region.replace('_', ' ')}", "country")

            oid = add_entity(f"op_{ri}_{canon}", f"{q} [{canon}]"[:200], "opinion")
            opinion_texts[oid] = [f"{q} {opt}" for opt in options]
            opinion_dist[oid] = probs
            opinion_cells.append((oid, country_id))
            used = True
        if used:
            n_rows_used += 1

    r_ans  = RELATION_VOCAB["answered_by"]
    r_loc  = RELATION_VOCAB["located_in"]
    r_part = RELATION_VOCAB["part_of"]

    # --- structural triples (region->world, country->region) ---
    structural_triples: list[tuple[int, int, int]] = []
    for nid, etype in entity_types.items():
        if etype == "region":
            structural_triples.append((nid, r_part, world_id))
        elif etype == "country":
            region_id = entity_vocab[COUNTRY_TO_REGION[id_to_entity[nid]]]
            structural_triples.append((nid, r_loc, region_id))

    # --- opinion triples + train/val/test split (on opinions only) ---
    opinion_triples = [(oid, r_ans, cid) for oid, cid in opinion_cells]
    rng = random.Random(split_seed)
    shuffled = opinion_triples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    train_op = shuffled[:n_train]
    val_op   = shuffled[n_train:n_train + n_val]
    test_op  = shuffled[n_train + n_val:]

    if leakage_safe:
        # Structural links are trivial -> train-only. Tree is built from TRAIN
        # opinions only, so a held-out opinion never leaks into its country's
        # aggregated embedding (it stays an isolated, inductively-encoded leaf).
        train_triples = structural_triples + train_op
        val_triples   = val_op
        test_triples  = test_op
        tree_op = train_op
    else:
        train_triples = structural_triples + train_op
        val_triples   = structural_triples + val_op
        test_triples  = structural_triples + test_op
        tree_op = opinion_triples
    all_triples = structural_triples + shuffled

    # --- tree (World -> Region -> Country -> Opinion) ---
    n_entities = len(id_to_entity)
    children_indices: list[list[int]] = [[] for _ in range(n_entities)]
    for nid, etype in entity_types.items():
        if etype == "region":
            children_indices[world_id].append(nid)
        elif etype == "country":
            region_id = entity_vocab[COUNTRY_TO_REGION[id_to_entity[nid]]]
            children_indices[region_id].append(nid)
    for oid, _, cid in tree_op:
        children_indices[cid].append(oid)

    from pluraltree.utils.tree_utils import topological_sort
    topo_order = topological_sort(children_indices)

    # --- type constraints for negative sampling ---
    country_ids = [nid for nid, t in entity_types.items() if t == "country"]
    region_ids  = [nid for nid, t in entity_types.items() if t == "region"]
    type_constraints = {
        r_ans:  country_ids,
        r_loc:  region_ids,
        r_part: [world_id],
    }

    n_op = sum(1 for t in entity_types.values() if t == "opinion")
    print(f"  GlobalOpinionQA: {n_entities} entities "
          f"({len(country_ids)} countries, {len(region_ids)} regions, {n_op} opinions) "
          f"from {n_rows_used} questions")
    if skipped_countries:
        print(f"  Skipped {len(skipped_countries)} unmapped country labels "
              f"(e.g. {sorted(skipped_countries)[:5]})")
    print(f"  Opinion triples — train {len(train_op)} | val {len(val_op)} | test {len(test_op)} "
          f"| leakage_safe={leakage_safe}")

    return GlobalOpinionGraph(
        entity_vocab=entity_vocab,
        id_to_entity=id_to_entity,
        relation_vocab=RELATION_VOCAB,
        all_triples=all_triples,
        train_triples=train_triples,
        val_triples=val_triples,
        test_triples=test_triples,
        entity_text=entity_text,
        children_indices=children_indices,
        topo_order=topo_order,
        type_constraints=type_constraints,
        entity_types=entity_types,
        opinion_texts=opinion_texts,
        opinion_dist=opinion_dist,
    )


def load_globalopinionqa(
    split_seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    leakage_safe: bool = True,
    config: str | None = None,
) -> GlobalOpinionGraph:
    """Download Anthropic/llm_global_opinions and build the opinion graph."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install huggingface datasets: pip install datasets")

    ds = load_dataset("Anthropic/llm_global_opinions") if config is None \
        else load_dataset("Anthropic/llm_global_opinions", config)
    split = "train" if "train" in ds else list(ds.keys())[0]
    rows = [dict(r) for r in ds[split]]
    return _build_graph(rows, split_seed=split_seed, train_frac=train_frac,
                        val_frac=val_frac, leakage_safe=leakage_safe)


# ---------------------------------------------------------------------------
# Node features: distribution-weighted option embeddings for opinions
# ---------------------------------------------------------------------------
def compute_features(graph: GlobalOpinionGraph, model_name: str = "all-MiniLM-L6-v2") -> Tensor:
    """(N, d) node features.

    Opinion node  -> sum_i p_i * MiniLM("{question} {option_i}")  (its country's
                     distribution-weighted answer, in option-text space).
    Other nodes   -> MiniLM(entity_text).

    Two countries that answer a question differently get different opinion
    vectors, and their geodesic distance reflects how differently — the signal
    branch divergence measures.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers")

    model = SentenceTransformer(model_name)
    n = len(graph.id_to_entity)

    # Encode every unique string once (option-grounded opinion strings + templates).
    unique: set[str] = set()
    for nid in range(n):
        if nid in graph.opinion_texts:
            unique.update(graph.opinion_texts[nid])
        else:
            unique.add(graph.entity_text[nid])
    texts = sorted(unique)
    emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=True).clone()
    idx = {t: i for i, t in enumerate(texts)}

    out = torch.zeros(n, emb.shape[1])
    for nid in range(n):
        if nid in graph.opinion_texts:
            ts = graph.opinion_texts[nid]
            w = torch.tensor(graph.opinion_dist[nid]).unsqueeze(1)   # (k, 1)
            vecs = emb[[idx[t] for t in ts]]                         # (k, d)
            out[nid] = (w * vecs).sum(0)
        else:
            out[nid] = emb[idx[graph.entity_text[nid]]]
    return out
