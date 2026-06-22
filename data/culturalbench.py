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
import re
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
# Country name + demonym aliases (for label masking)
# ---------------------------------------------------------------------------
# ~89% of CulturalBench question texts literally name the country or its demonym
# (e.g. "In Japanese culture...", "...for Spanish people?"). Embedding that text
# verbatim leaks the answer into the input features, so practice->country link
# prediction reduces to string matching (test MRR ~0.94 with no graph signal).
# These aliases are stripped from question text when mask_country=True so the
# model must infer the country from the cultural *content*, not a literal mention.
COUNTRY_ALIASES: dict[str, list[str]] = {
    "China":          ["China", "Chinese"],
    "Hong Kong":      ["Hong Kong", "Hongkong", "Hongkonger"],
    "Japan":          ["Japan", "Japanese"],
    "South Korea":    ["South Korea", "South Korean", "Korea", "Korean"],
    "Taiwan":         ["Taiwan", "Taiwanese"],
    "Indonesia":      ["Indonesia", "Indonesian"],
    "Malaysia":       ["Malaysia", "Malaysian"],
    "Philippines":    ["Philippines", "Philippine", "Filipino", "Filipina", "Filipinos"],
    "Singapore":      ["Singapore", "Singaporean"],
    "Thailand":       ["Thailand", "Thai"],
    "Vietnam":        ["Vietnam", "Vietnamese"],
    "Bangladesh":     ["Bangladesh", "Bangladeshi"],
    "India":          ["India", "Indian"],
    "Nepal":          ["Nepal", "Nepali", "Nepalese"],
    "Pakistan":       ["Pakistan", "Pakistani"],
    "Czech Republic": ["Czech Republic", "Czechia", "Czech"],
    "Poland":         ["Poland", "Polish"],
    "Romania":        ["Romania", "Romanian"],
    "Russia":         ["Russia", "Russian"],
    "Ukraine":        ["Ukraine", "Ukrainian"],
    "France":         ["France", "French"],
    "Germany":        ["Germany", "German"],
    "Netherlands":    ["Netherlands", "Holland", "Dutch"],
    "United Kingdom": ["United Kingdom", "Great Britain", "Britain", "British",
                       "England", "English", "UK"],
    "Italy":          ["Italy", "Italian"],
    "Spain":          ["Spain", "Spanish", "Spaniard"],
    "Iran":           ["Iran", "Iranian", "Persian"],
    "Israel":         ["Israel", "Israeli"],
    "Lebanon":        ["Lebanon", "Lebanese"],
    "Saudi Arabia":   ["Saudi Arabia", "Saudi Arabian", "Saudi"],
    "Turkey":         ["Turkey", "Turkish"],
    "Egypt":          ["Egypt", "Egyptian"],
    "Morocco":        ["Morocco", "Moroccan"],
    "Nigeria":        ["Nigeria", "Nigerian"],
    "South Africa":   ["South Africa", "South African"],
    "Zimbabwe":       ["Zimbabwe", "Zimbabwean"],
    "Argentina":      ["Argentina", "Argentine", "Argentinian"],
    "Brazil":         ["Brazil", "Brazilian"],
    "Chile":          ["Chile", "Chilean"],
    "Mexico":         ["Mexico", "Mexican"],
    "Peru":           ["Peru", "Peruvian"],
    "Canada":         ["Canada", "Canadian"],
    "United States":  ["United States of America", "United States", "USA",
                       "America", "American"],
    "Australia":      ["Australia", "Australian"],
    "New Zealand":    ["New Zealand", "New Zealander", "Kiwi"],
}

MASK_TOKEN = "[COUNTRY]"

# One word-boundary alternation over every alias, longest first so multi-word
# names ("South Korea") match before their substrings ("Korea"). Case-insensitive.
_ALL_ALIASES = sorted(
    {a for aliases in COUNTRY_ALIASES.values() for a in aliases},
    key=len, reverse=True,
)
_COUNTRY_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _ALL_ALIASES) + r")\b",
    re.IGNORECASE,
)


def mask_country_text(text: str) -> tuple[str, bool]:
    """Replace any country name/demonym mention with MASK_TOKEN.

    Returns (masked_text, changed) where changed is True if any mention was
    found. Collapses runs of mask tokens and extra whitespace so the result
    reads cleanly (e.g. "In Japanese culture" -> "In [COUNTRY] culture").
    """
    masked, n = _COUNTRY_RE.subn(MASK_TOKEN, text)
    if n == 0:
        return text, False
    masked = re.sub(r"(?:" + re.escape(MASK_TOKEN) + r"\s*){2,}",
                    MASK_TOKEN + " ", masked)
    masked = re.sub(r"\s{2,}", " ", masked).strip()
    return masked, True


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

    # Inductive holdout (optional). When a holdout is requested, these hold the
    # entities whose triples were removed from train/val/test (so the model never
    # trains on their links) and the held-out triples to evaluate inductively
    # (non-hierarchy relations of those entities). Empty otherwise.
    inductive_test:     list[tuple[int, int, int]] = field(default_factory=list)
    holdout_entity_ids: set[int]                    = field(default_factory=set)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def load_culturalbench(
    split_seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    leakage_safe: bool = True,
    mask_country: bool = True,
) -> CulturalGraph:
    """Load CulturalBench and build the knowledge graph.

    Downloads kellycyy/CulturalBench (CulturalBench-Easy) from HuggingFace,
    deduplicates by question_idx, builds the geographic hierarchy tree, and
    creates train/val/test triple splits.

    Args:
        split_seed: random seed for reproducible splits
        train_frac: fraction of practice triples for training
        val_frac: fraction for validation (remainder goes to test)
        leakage_safe: if True (default), avoid the two leakage sources:
            (1) structural triples (country->region, region->world) are trivial
                and memorized — they are kept in TRAIN only, never scored in
                val/test, so evaluation reflects real practice prediction;
            (2) the Country->Practice tree edges are built from TRAIN practices
                ONLY, so the encoder never aggregates a held-out practice into
                its country's embedding (the leaked structural answer).
            Set False to reproduce the original (leaky) behavior.
        mask_country: if True (default), strip country names and demonyms from
            each practice's question text before it becomes the node feature
            (e.g. "In Japanese culture..." -> "In [COUNTRY] culture..."). Without
            this, ~89% of questions name the answer country in the input, so
            practice->country prediction is trivial string matching rather than
            cultural inference. Set False to reproduce the leaky text.

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
    n_masked = 0
    for q in questions:
        name = f"practice_{q['question_idx']}"
        text = q["prompt_question"]
        if mask_country:
            text, changed = mask_country_text(text)
            n_masked += int(changed)
        add_entity(name, text, "practice")
        practice_questions.append(q)
    if mask_country:
        print(f"  Masked country mention in {n_masked}/{len(practice_questions)} "
              f"practice texts")

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

    train_practices = shuffled[:n_train]
    val_practices   = shuffled[n_train : n_train + n_val]
    test_practices  = shuffled[n_train + n_val :]

    if leakage_safe:
        # Source 1 fix: structural triples are trivial (part_of has 1 candidate;
        # located_in is fully memorized from train) — keep them in TRAIN only so
        # val/test measure genuine practice prediction, not memorized structure.
        train_triples = structural_triples + train_practices
        val_triples   = val_practices
        test_triples  = test_practices
        # Source 2 fix: build the Country->Practice tree from TRAIN practices only.
        tree_practice_triples = train_practices
    else:
        # Legacy (leaky) behavior, kept for reproducing prior numbers.
        train_triples = structural_triples + train_practices
        val_triples   = structural_triples + val_practices
        test_triples  = structural_triples + test_practices
        tree_practice_triples = practice_triples

    # all_triples = every known-true triple, used ONLY as the filter set for
    # filtered ranking (excludes other true triples from competing). This is
    # standard filtered-setting practice and does not leak into embeddings.
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

    # Country → Practices.
    # leakage_safe: train practices only (held-out practices stay isolated leaves,
    # encoded from their own text alone — the inductive, leakage-free setup).
    for s_id, r_id, o_id in tree_practice_triples:
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
    return embeddings.clone()  # (N, embed_dim): 384 for MiniLM, 768 for mpnet
