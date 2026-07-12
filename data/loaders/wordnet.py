"""WN18RR dataset loader — produces the same CulturalGraph interface.

WN18RR is the hierarchy-heavy WordNet link-prediction benchmark (≈40,943
entities, 11 relations). Unlike CulturalBench's single shallow rooted tree, the
WordNet hypernym structure is a deep *forest of DAGs*. We adapt it to the
PluralTree encoder without changing the architecture:

  * The encoding tree is built from the ``_hypernym`` / ``_instance_hypernym``
    edges (tail = hypernym = parent, head = hyponym = child).
  * A single virtual ROOT is added as the parent of every entity that has no
    hypernym parent, so the whole graph is one connected, single-rooted DAG that
    the bottom-up encoder can traverse. (``topological_sort`` already supports
    DAGs / multi-parent nodes.)
  * Leakage-safe by default: the tree uses TRAIN hypernym edges only, so a
    held-out (val/test) hypernym link is never baked into the structure that
    produces its endpoints' embeddings.

Link prediction uses the standard WN18RR setting: rank the true object against
ALL entities (no type constraint). Pair this loader with the vectorized
``evaluate_link_prediction`` — the per-candidate Python loop is far too slow at
~40K candidates.

Data layout (place the standard WN18RR release under ``data_dir``):
    data/wn18rr/train.txt          head \t relation \t tail   (one per line)
    data/wn18rr/valid.txt
    data/wn18rr/test.txt
    data/wn18rr/entity2text.txt    (optional) entity_id \t short text
    data/wn18rr/entity2textlong.txt (optional) entity_id \t definition

Entity text (the node feature source) is resolved in this order:
    1. entity2textlong.txt / entity2text.txt if present (recommended),
    2. nltk WordNet glosses if the ids look like synset offsets and nltk is
       installed,
    3. a cleaned form of the raw id (fallback — embeddings will be weak).
"""

from __future__ import annotations

import os
import random
from collections import defaultdict

from data.loaders.culturalbench import CulturalGraph
from pluraltree.utils.tree_utils import topological_sort

ROOT_NAME = "__ROOT__"
HIERARCHY_RELATIONS = ("_hypernym", "_instance_hypernym")


# ---------------------------------------------------------------------------
# Triple / text file readers
# ---------------------------------------------------------------------------
def _read_triples(path: str) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            triples.append((h, r, t))
    return triples


def _read_entity_text(data_dir: str) -> dict[str, str]:
    """Load an entity_id -> text map from the optional KG-BERT text files."""
    for fname in ("entity2textlong.txt", "entity2text.txt"):
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            mapping: dict[str, str] = {}
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t", 1)
                    if len(parts) == 2:
                        mapping[parts[0]] = parts[1].strip()
            if mapping:
                print(f"  Loaded entity text from {fname} ({len(mapping)} entries)")
                return mapping
    return {}


def _nltk_gloss(entity_id: str) -> str | None:
    """Best-effort WordNet gloss for a synset-offset id (needs nltk data).

    Offsets absent from nltk's WordNet version emit a UserWarning and return None;
    we suppress those (they're expected) and fall back to the raw id upstream.
    """
    import warnings
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return None
    digits = entity_id.lstrip("0") or "0"
    if not digits.isdigit():
        return None
    offset = int(digits)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for pos in ("n", "v", "a", "r", "s"):
            try:
                syn = wn.synset_from_pos_and_offset(pos, offset)
            except Exception:
                continue
            if syn is not None:
                name = syn.lemma_names()[0].replace("_", " ")
                return f"{name} — {syn.definition()}"
    return None


def _fallback_text(entity_id: str) -> str:
    return entity_id.replace("_", " ").strip()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_wn18rr(
    data_dir: str = "data/wn18rr",
    split_seed: int = 42,          # accepted for a uniform interface; splits are fixed
    leakage_safe: bool = True,
    hierarchy_relations: tuple[str, ...] = HIERARCHY_RELATIONS,
    holdout_entities: float = 0.0,
) -> CulturalGraph:
    """Load WN18RR and build a CulturalGraph for the Tree-GRU encoder.

    Args:
        data_dir: directory with train.txt / valid.txt / test.txt (+ optional
            entity2text files).
        split_seed: unused for splitting (WN18RR ships fixed splits); kept so the
            call signature matches load_culturalbench.
        leakage_safe: if True (default), the encoding tree is built from TRAIN
            hypernym edges only, so held-out hypernym links are not encoded into
            their endpoints. Set False to build the tree from all splits' edges.
        hierarchy_relations: relation names treated as parent/child edges.

    Returns:
        CulturalGraph with the same fields used by the trainer/encoder.
    """
    for fname in ("train.txt", "valid.txt", "test.txt"):
        if not os.path.exists(os.path.join(data_dir, fname)):
            raise FileNotFoundError(
                f"Missing {fname} in {data_dir!r}. Download the WN18RR release "
                f"(train/valid/test.txt) into that directory. See data/loaders/wordnet.py "
                f"header for the expected layout."
            )

    train_raw = _read_triples(os.path.join(data_dir, "train.txt"))
    val_raw   = _read_triples(os.path.join(data_dir, "valid.txt"))
    test_raw  = _read_triples(os.path.join(data_dir, "test.txt"))

    # ------------------------------------------------------------------
    # Vocabularies
    # ------------------------------------------------------------------
    entity_vocab: dict[str, int] = {}
    id_to_entity: list[str] = []
    relation_vocab: dict[str, int] = {}

    def add_entity(name: str) -> int:
        if name not in entity_vocab:
            entity_vocab[name] = len(id_to_entity)
            id_to_entity.append(name)
        return entity_vocab[name]

    def add_relation(name: str) -> int:
        if name not in relation_vocab:
            relation_vocab[name] = len(relation_vocab)
        return relation_vocab[name]

    for h, r, t in train_raw + val_raw + test_raw:
        add_entity(h)
        add_entity(t)
        add_relation(r)
    root_id = add_entity(ROOT_NAME)

    def encode(triples_raw):
        return [
            (entity_vocab[h], relation_vocab[r], entity_vocab[t])
            for h, r, t in triples_raw
        ]

    train_triples = encode(train_raw)
    val_triples   = encode(val_raw)
    test_triples  = encode(test_raw)
    all_triples   = train_triples + val_triples + test_triples

    # ------------------------------------------------------------------
    # Entity text (node features)
    # ------------------------------------------------------------------
    text_map = _read_entity_text(data_dir)
    entity_text: dict[int, str] = {}
    n_nltk = n_fallback = 0
    for eid, name in enumerate(id_to_entity):
        if name == ROOT_NAME:
            entity_text[eid] = "root — the most general concept, top of the hierarchy"
            continue
        if name in text_map:
            entity_text[eid] = text_map[name]
            continue
        gloss = _nltk_gloss(name)
        if gloss is not None:
            entity_text[eid] = gloss
            n_nltk += 1
        else:
            entity_text[eid] = _fallback_text(name)
            n_fallback += 1
    if n_nltk:
        print(f"  Resolved {n_nltk} entity texts via nltk WordNet glosses")
    if n_fallback:
        print(f"  WARNING: {n_fallback} entities fell back to raw-id text "
              f"(provide entity2text.txt for meaningful features)")

    # ------------------------------------------------------------------
    # Build the encoding tree from hypernym edges
    # tail = hypernym = parent; head = hyponym = child.
    # ------------------------------------------------------------------
    hier_rel_ids = {relation_vocab[r] for r in hierarchy_relations if r in relation_vocab}
    tree_source = train_triples if leakage_safe else all_triples

    n_entities = len(id_to_entity)
    child_sets: list[set[int]] = [set() for _ in range(n_entities)]
    for s_id, r_id, o_id in tree_source:
        if r_id in hier_rel_ids:
            child_sets[o_id].add(s_id)          # parent o_id gets child s_id

    # Attach every parentless entity (except ROOT) under the virtual ROOT so the
    # whole graph is one connected, single-rooted DAG.
    has_parent = set()
    for parent, kids in enumerate(child_sets):
        has_parent.update(kids)
    for eid in range(n_entities):
        if eid == root_id:
            continue
        if eid not in has_parent:
            child_sets[root_id].add(eid)

    children_indices: list[list[int]] = [sorted(s) for s in child_sets]
    topo_order = topological_sort(children_indices)

    # ------------------------------------------------------------------
    # Optional inductive holdout. Pick a fraction of LEAF entities, remove all
    # their triples from train/val/test (the model never trains on them, and they
    # are not in the transductive eval), and collect their non-hierarchy triples
    # as an inductive test set. Their hierarchy position stays in the tree (it was
    # built above), so they are embedded from text + structure alone — the test of
    # whether the *computed* embeddings generalise to never-trained nodes.
    # ------------------------------------------------------------------
    inductive_test: list[tuple[int, int, int]] = []
    holdout_ids: set[int] = set()
    if holdout_entities > 0.0:
        hier_rel_ids = {relation_vocab[r] for r in hierarchy_relations if r in relation_vocab}
        leaves = [eid for eid in range(n_entities)
                  if eid != root_id and not children_indices[eid]]
        rng = random.Random(split_seed)
        rng.shuffle(leaves)
        k = int(round(holdout_entities * len(leaves)))
        holdout_ids = set(leaves[:k])

        def _touches(t):
            return t[0] in holdout_ids or t[2] in holdout_ids

        inductive_test = [t for t in all_triples
                          if _touches(t) and t[1] not in hier_rel_ids]
        train_triples = [t for t in train_triples if not _touches(t)]
        val_triples   = [t for t in val_triples   if not _touches(t)]
        test_triples  = [t for t in test_triples  if not _touches(t)]
        print(f"  Inductive holdout: {len(holdout_ids)} leaf entities "
              f"({holdout_entities:.0%} of {len(leaves)} leaves) | "
              f"{len(inductive_test)} inductive test triples | "
              f"train now {len(train_triples)}")

    # ------------------------------------------------------------------
    # Negative sampling / ranking candidates: all entities except ROOT.
    # Every relation shares the same candidate list (no WordNet type system).
    # ------------------------------------------------------------------
    candidate_ids = [eid for eid in range(n_entities) if eid != root_id]
    type_constraints = {r_id: candidate_ids for r_id in relation_vocab.values()}

    entity_types = {eid: ("root" if name == ROOT_NAME else "synset")
                    for eid, name in enumerate(id_to_entity)}

    n_tree_edges = sum(len(c) for c in children_indices)
    print(f"  WN18RR: {n_entities} entities (incl. ROOT), "
          f"{len(relation_vocab)} relations")
    print(f"  Triples — train {len(train_triples)} | val {len(val_triples)} | "
          f"test {len(test_triples)}")
    print(f"  Tree edges: {n_tree_edges} "
          f"(ROOT children: {len(children_indices[root_id])}) | "
          f"leakage_safe={leakage_safe}")

    return CulturalGraph(
        entity_vocab=entity_vocab,
        id_to_entity=id_to_entity,
        relation_vocab=relation_vocab,
        all_triples=all_triples,
        train_triples=train_triples,
        val_triples=val_triples,
        test_triples=test_triples,
        entity_text=entity_text,
        children_indices=children_indices,
        topo_order=topo_order,
        type_constraints=type_constraints,
        entity_types=entity_types,
        inductive_test=inductive_test,
        holdout_entity_ids=holdout_ids,
    )
