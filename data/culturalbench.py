"""CulturalBench dataset loader.

Loads kellycyy/CulturalBench (Easy split) from HuggingFace and builds a
knowledge graph with geographic hierarchy:

    World → Region → Country → Cultural Practice (leaf)

Relations:
    practiced_in  : practice → country
    located_in    : country  → region
    part_of       : region   → world
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Static geographic hierarchy
# ---------------------------------------------------------------------------

REGION_TO_COUNTRIES: dict[str, list[str]] = {
    "East_Asia":       ["China", "Hong Kong", "Japan", "South Korea", "Taiwan"],
    "Southeast_Asia":  ["Indonesia", "Malaysia", "Philippines", "Singapore", "Thailand", "Vietnam"],
    "South_Asia":      ["Bangladesh", "India", "Nepal", "Pakistan"],
    "Eastern_Europe":  ["Czech Republic", "Poland", "Romania", "Russia", "Ukraine"],
    "Western_Europe":  ["France", "Germany", "Netherlands", "United Kingdom"],
    "Southern_Europe": ["Italy", "Spain"],
    "Middle_East":     ["Iran", "Israel", "Lebanon", "Saudi Arabia", "Turkey"],
    "Africa":          ["Egypt", "Morocco", "Nigeria", "South Africa", "Zimbabwe"],
    "South_America":   ["Argentina", "Brazil", "Chile", "Mexico", "Peru"],
    "North_America":   ["Canada", "United States"],
    "Oceania":         ["Australia", "New Zealand"],
}

COUNTRY_TO_REGION: dict[str, str] = {
    country: region
    for region, countries in REGION_TO_COUNTRIES.items()
    for country in countries
}

RELATION_VOCAB: dict[str, int] = {
    "practiced_in": 0,
    "located_in":   1,
    "part_of":      2,
}
NUM_RELATIONS = len(RELATION_VOCAB)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class CulturalGraph:
    """Complete knowledge graph built from CulturalBench."""

    # Entity vocabulary
    entity_vocab:  dict[str, int]   # name → id
    id_to_entity:  list[str]        # id  → name

    # Relations
    relation_vocab: dict[str, int]  # name → id

    # All triples (s_id, r_id, o_id)
    all_triples:   list[tuple[int, int, int]]
    train_triples: list[tuple[int, int, int]]
    val_triples:   list[tuple[int, int, int]]
    test_triples:  list[tuple[int, int, int]]

    # Raw text for embedding (entity_id → text)
    entity_text: dict[int, str]

    # Tree structure (full hierarchy, single rooted tree)
    children_indices: list[list[int]]   # children_indices[i] = [child ids of node i]
    topo_order:       list[int]         # leaves first, root last

    # Type constraints for negative sampling
    # Maps relation_id → list of valid object entity ids
    type_constraints: dict[int, list[int]]

    # Entity type labels for analysis
    entity_types: dict[int, str]  # id → "world" | "region" | "country" | "practice"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def load_culturalbench(
    split_seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> CulturalGraph:
    """Load CulturalBench and build the knowledge graph.

    Downloads kellycyy/CulturalBench (CulturalBench-Easy) from HuggingFace,
    deduplicates by question_idx, builds the geographic hierarchy tree, and
    creates train/val/test triple splits.

    Args:
        split_seed: random seed for reproducible splits
        train_frac: fraction of practice triples for training
        val_frac: fraction for validation (remainder goes to test)

    Returns:
        CulturalGraph with all structure needed for training
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install huggingface datasets: pip install datasets")

    ds = load_dataset("kellycyy/CulturalBench", "CulturalBench-Easy")

    # Use the test split (only split available) — we create our own train/val/test
    rows = ds["test"]

    # Deduplicate by question_idx (easy split: one row per question)
    seen_qidx: set[int] = set()
    questions: list[dict] = []
    for row in rows:
        qidx = row["question_idx"]
        if qidx not in seen_qidx:
            seen_qidx.add(qidx)
            questions.append(dict(row))

    # Filter to known countries only
    questions = [q for q in questions if q["country"] in COUNTRY_TO_REGION]

    # ------------------------------------------------------------------
    # Build entity vocabulary
    # ------------------------------------------------------------------
    entity_vocab: dict[str, int] = {}
    id_to_entity: list[str] = []
    entity_text:  dict[int, str] = {}
    entity_types: dict[int, str] = {}

    def add_entity(name: str, text: str, etype: str) -> int:
        if name not in entity_vocab:
            eid = len(id_to_entity)
            entity_vocab[name] = eid
            id_to_entity.append(name)
            entity_text[eid] = text
            entity_types[eid] = etype
        return entity_vocab[name]

    # World (root)
    add_entity("World", "World — the global community of all nations and cultures", "world")

    # Regions
    for region in REGION_TO_COUNTRIES:
        readable = region.replace("_", " ")
        add_entity(region, f"{readable} — geographic and cultural region", "region")

    # Countries
    for region, countries in REGION_TO_COUNTRIES.items():
        for country in countries:
            add_entity(country, f"{country} — country in {region.replace('_', ' ')}", "country")

    # Cultural practices (one per question)
    practice_questions: list[dict] = []
    for q in questions:
        name = f"practice_{q['question_idx']}"
        text = q["prompt_question"]
        add_entity(name, text, "practice")
        practice_questions.append(q)

    # ------------------------------------------------------------------
    # Build triples
    # ------------------------------------------------------------------
    r_practiced_in = RELATION_VOCAB["practiced_in"]
    r_located_in   = RELATION_VOCAB["located_in"]
    r_part_of      = RELATION_VOCAB["part_of"]

    structural_triples: list[tuple[int, int, int]] = []
    practice_triples:   list[tuple[int, int, int]] = []

    world_id = entity_vocab["World"]

    # Region → World
    for region in REGION_TO_COUNTRIES:
        structural_triples.append((entity_vocab[region], r_part_of, world_id))

    # Country → Region
    for region, countries in REGION_TO_COUNTRIES.items():
        region_id = entity_vocab[region]
        for country in countries:
            country_id = entity_vocab[country]
            structural_triples.append((country_id, r_located_in, region_id))

    # Practice → Country
    for q in practice_questions:
        practice_name = f"practice_{q['question_idx']}"
        practice_id   = entity_vocab[practice_name]
        country_id    = entity_vocab[q["country"]]
        practice_triples.append((practice_id, r_practiced_in, country_id))

    # ------------------------------------------------------------------
    # Train / val / test split on practice triples only
    # Structural triples (country→region, region→world) are always visible
    # ------------------------------------------------------------------
    rng = random.Random(split_seed)
    shuffled = practice_triples[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_triples = structural_triples + shuffled[:n_train]
    val_triples   = structural_triples + shuffled[n_train : n_train + n_val]
    test_triples  = structural_triples + shuffled[n_train + n_val :]
    all_triples   = structural_triples + shuffled

    # ------------------------------------------------------------------
    # Tree structure (full hierarchy)
    # ------------------------------------------------------------------
    n_entities = len(id_to_entity)
    children_indices: list[list[int]] = [[] for _ in range(n_entities)]

    # World → Regions
    for region in REGION_TO_COUNTRIES:
        children_indices[world_id].append(entity_vocab[region])

    # Region → Countries
    for region, countries in REGION_TO_COUNTRIES.items():
        region_id = entity_vocab[region]
        for country in countries:
            children_indices[region_id].append(entity_vocab[country])

    # Country → Practices (using ALL practice triples to build the tree)
    for s_id, r_id, o_id in practice_triples:
        if r_id == r_practiced_in:
            children_indices[o_id].append(s_id)  # country → practice

    # Topological sort (leaves first)
    from pluraltree.utils.tree_utils import topological_sort
    topo_order = topological_sort(children_indices)

    # ------------------------------------------------------------------
    # Type constraints for negative sampling
    # ------------------------------------------------------------------
    country_ids  = [entity_vocab[c] for countries in REGION_TO_COUNTRIES.values() for c in countries]
    region_ids   = [entity_vocab[r] for r in REGION_TO_COUNTRIES]
    practice_ids = [entity_vocab[f"practice_{q['question_idx']}"] for q in practice_questions]

    type_constraints = {
        r_practiced_in: country_ids,   # corrupt with another country
        r_located_in:   region_ids,    # corrupt with another region
        r_part_of:      [world_id],    # only one world
    }

    return CulturalGraph(
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
    )


def compute_text_embeddings(graph: CulturalGraph, model_name: str = "all-MiniLM-L6-v2") -> Tensor:
    """Compute sentence-transformer embeddings for all entities.

    Args:
        graph: the CulturalGraph
        model_name: sentence-transformers model name

    Returns:
        (N, embed_dim) tensor of embeddings, indexed by entity id
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers")

    model = SentenceTransformer(model_name)
    n = len(graph.id_to_entity)
    texts = [graph.entity_text[i] for i in range(n)]
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    # Clone to exit inference_mode context set by sentence-transformers
    return embeddings.clone()  # (N, 384) for MiniLM
