"""Loader for the GraIL inductive WN18RR benchmark (Teru et al., ICML 2020).

Produces two CulturalGraphs with DISJOINT entities but a SHARED relation vocabulary
(relations transfer; entities do not):

    train_graph : the training graph (WN18RR_v{k}/{train,valid,test}.txt)
    ind_graph   : the inductive graph of new entities (WN18RR_v{k}_ind), whose
                  train.txt is the support structure and test.txt are the held-out
                  facts to predict.

Train the model on ``train_graph``, then encode ``ind_graph`` (its own tree, its own
text features) and rank its ``test_triples`` — never-seen entities embedded from text
+ structure. This is the standard inductive protocol, so results are comparable to
GraIL / NBFNet / NodePiece etc.
"""

from __future__ import annotations

import os

from data.loaders.culturalbench import CulturalGraph
from data.loaders.wordnet import (
    _read_triples, _read_entity_text, _nltk_gloss, _fallback_text,
    ROOT_NAME, HIERARCHY_RELATIONS,
)
from pluraltree.utils.tree_utils import topological_sort


def _build_graph(rel_vocab, text_map, train_raw, val_raw, test_raw) -> CulturalGraph:
    """Build one CulturalGraph with entities local to this graph, sharing rel_vocab.

    The tree is built from this graph's TRAIN hypernym edges only (leakage-safe).
    """
    entity_vocab: dict[str, int] = {}
    id_to_entity: list[str] = []

    def add_e(name: str) -> int:
        if name not in entity_vocab:
            entity_vocab[name] = len(id_to_entity)
            id_to_entity.append(name)
        return entity_vocab[name]

    for h, r, t in train_raw + val_raw + test_raw:
        add_e(h); add_e(t)
    root_id = add_e(ROOT_NAME)

    def enc(raws):
        return [(entity_vocab[h], rel_vocab[r], entity_vocab[t]) for h, r, t in raws]

    train = enc(train_raw)
    val   = enc(val_raw)
    test  = enc(test_raw)
    all_t = train + val + test
    n = len(id_to_entity)

    # Entity text (node features): text file -> nltk gloss -> raw-id fallback.
    entity_text: dict[int, str] = {}
    for eid, name in enumerate(id_to_entity):
        if name == ROOT_NAME:
            entity_text[eid] = "root — the most general concept, top of the hierarchy"
            continue
        if name in text_map:
            entity_text[eid] = text_map[name]
            continue
        gloss = _nltk_gloss(name)
        entity_text[eid] = gloss if gloss is not None else _fallback_text(name)

    # Tree from this graph's train hypernym edges; parentless -> ROOT.
    hier = {rel_vocab[r] for r in HIERARCHY_RELATIONS if r in rel_vocab}
    child_sets: list[set[int]] = [set() for _ in range(n)]
    for s, r, o in train:
        if r in hier:
            child_sets[o].add(s)
    has_parent: set[int] = set()
    for _, kids in enumerate(child_sets):
        has_parent.update(kids)
    for eid in range(n):
        if eid != root_id and eid not in has_parent:
            child_sets[root_id].add(eid)
    children = [sorted(s) for s in child_sets]
    topo = topological_sort(children)

    cand = [eid for eid in range(n) if eid != root_id]
    type_constraints = {r_id: cand for r_id in rel_vocab.values()}
    entity_types = {eid: ("root" if id_to_entity[eid] == ROOT_NAME else "synset")
                    for eid in range(n)}

    return CulturalGraph(
        entity_vocab=entity_vocab,
        id_to_entity=id_to_entity,
        relation_vocab=rel_vocab,
        all_triples=all_t,
        train_triples=train,
        val_triples=val,
        test_triples=test,
        entity_text=entity_text,
        children_indices=children,
        topo_order=topo,
        type_constraints=type_constraints,
        entity_types=entity_types,
    )


def load_grail_wn18rr(version: str = "v1", data_dir: str = "data/grail",
                      text_dir: str = "data/wn18rr"):
    """Load one GraIL inductive WN18RR version.

    Entity text (node features) is resolved in this order: an entity2text file in
    the GraIL dir, then the standard WN18RR ``entity2text.txt`` under ``text_dir``
    (curated descriptions for all 40,943 synsets — avoids nltk WordNet-version
    offset mismatches), then nltk glosses, then a raw-id fallback.

    Returns:
        (train_graph, ind_graph) — both CulturalGraphs sharing the relation vocab.
        ind_graph.test_triples are the held-out facts to predict inductively;
        ind_graph.train_triples are its support structure (the tree).
    """
    base = os.path.join(data_dir, f"WN18RR_{version}")
    ind  = base + "_ind"

    def rd(d, f):
        p = os.path.join(d, f)
        return _read_triples(p) if os.path.exists(p) else []

    tr  = rd(base, "train.txt"); va = rd(base, "valid.txt"); te = rd(base, "test.txt")
    itr = rd(ind, "train.txt");  ite = rd(ind, "test.txt")
    if not tr or not itr or not ite:
        raise FileNotFoundError(
            f"GraIL {version} data missing under {data_dir!r}. "
            f"Run scripts/get_grail_wn18rr.py on the login node."
        )

    # Shared relation vocabulary across both graphs (relations transfer).
    rel_vocab: dict[str, int] = {}
    for h, r, t in tr + va + te + itr + ite:
        if r not in rel_vocab:
            rel_vocab[r] = len(rel_vocab)

    # Prefer curated WN18RR descriptions (no nltk version mismatch); fall back to
    # the GraIL dirs, then (inside _build_graph) to nltk glosses / raw id.
    text_map = (_read_entity_text(text_dir)
                or _read_entity_text(base) or _read_entity_text(ind))

    train_graph = _build_graph(rel_vocab, text_map, tr, va, te)
    ind_graph   = _build_graph(rel_vocab, text_map, itr, [], ite)

    print(f"  GraIL WN18RR {version}: "
          f"train graph {len(train_graph.id_to_entity)} ent / {len(tr)} edges | "
          f"ind graph {len(ind_graph.id_to_entity)} ent / {len(itr)} support / "
          f"{len(ite)} test | {len(rel_vocab)} relations (shared)")
    return train_graph, ind_graph
