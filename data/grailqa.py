from __future__ import annotations

import random
from dataclasses import dataclass

from data.culturalbench import CulturalGraph

# Hierarchy relations occupy the first ids; Freebase relations are appended after.
HIERARCHY_RELATIONS: dict[str, int] = {
    "instance_of": 0,   # entity -> type (class)
    "subtype_of":  1,   # type   -> domain
    "in_domain":   2,   # domain -> root
}

ROOT_NAME = "__FREEBASE_ROOT__"


# ---------------------------------------------------------------------------
# HF nested-field normalization
# ---------------------------------------------------------------------------
def _as_records(seq) -> list[dict]:
    """Normalize a HF nested field into a list of dicts.

    ``datasets`` may return a Sequence-of-struct either as a list of dicts or,
    more often, as a struct-of-lists (parallel arrays). Handle both.
    """
    if seq is None:
        return []
    if isinstance(seq, dict):
        keys = list(seq.keys())
        if not keys:
            return []
        n = len(seq[keys[0]])
        return [{k: seq[k][i] for k in keys} for i in range(n)]
    return [dict(r) for r in seq]


def _humanize(s: str) -> str:
    """Turn a Freebase identifier into readable text: people.person -> people person."""
    return s.replace("_", " ").replace(".", " ").strip()


def _domain_of(class_str: str) -> str:
    """First namespace segment of a Freebase class: people.person -> people."""
    return class_str.split(".")[0] if class_str else ""


# ---------------------------------------------------------------------------
# Builder (network-free; takes already-parsed rows so it is unit-testable)
# ---------------------------------------------------------------------------
def _build_graph(
    rows,
    *,
    split_seed: int = 42,
    val_frac_of_dev: float = 0.5,
    leakage_safe: bool = True,
) -> CulturalGraph:
    """Build one CulturalGraph from GrailQA rows.

    Each row is a dict with ``split`` ("train"|"dev") and ``graph_query``
    (dict with ``nodes`` and ``edges``; either list-of-dicts or struct-of-lists).
    """
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

    root_id = add_entity(ROOT_NAME, "the root of the Freebase type hierarchy", "root")

    # Keys for the imposed hierarchy nodes (prefixed so they never collide with
    # Freebase mids, which are "m.xxx" / "g.xxx").
    def dom_key(d: str) -> str:  return f"dom:{d}"
    def cls_key(c: str) -> str:  return f"cls:{c}"

    type_friendly: dict[str, str] = {}     # class string -> best friendly name seen
    node_class: dict[str, str] = {}        # global node id (mid/class) -> its class

    # --- pass 1: collect nodes (entities + the types/domains they belong to) ---
    # Each row's nodes use a row-local nid; the global identity is node["id"].
    fb_edges: list[tuple[str, str, str, str]] = []   # (head_id, relation, tail_id, origin)
    skipped_literal = 0
    for row in rows:
        gq = row.get("graph_query") or {}
        nodes = _as_records(gq.get("nodes"))
        edges = _as_records(gq.get("edges"))
        origin = row.get("split", "train")

        # Map each row-local nid to the VOCAB KEY of its global node, so a Freebase
        # edge that lands on a class node (e.g. the question node "film.film")
        # resolves to that type's node rather than being dropped.
        local_key: dict[int, str] = {}
        for nd in nodes:
            ntype = (nd.get("node_type") or "").lower()
            gid = str(nd.get("id") or "").strip()
            if not gid or ntype == "literal":
                if ntype == "literal":
                    skipped_literal += 1
                continue
            cls = str(nd.get("class") or "").strip()
            fname = str(nd.get("friendly_name") or "").strip()

            if ntype == "class":
                # The node *is* a type; remember its friendly name.
                cls = cls or gid
                if fname:
                    type_friendly.setdefault(cls, fname)
                node_class[gid] = cls
                local_key[int(nd["nid"])] = cls_key(cls)      # -> type node
            else:  # entity (or anything non-class, non-literal)
                add_entity(gid, fname or _humanize(gid), "entity")
                if cls:
                    node_class[gid] = cls
                local_key[int(nd["nid"])] = gid                # -> entity node

        for ed in edges:
            try:
                h = local_key.get(int(ed["start"]))
                t = local_key.get(int(ed["end"]))
            except (KeyError, TypeError, ValueError):
                continue
            rel = str(ed.get("relation") or "").strip()
            if h and t and rel:
                fb_edges.append((h, rel, t, origin))

    # --- materialize the type + domain hierarchy nodes ---
    all_classes = set(type_friendly) | set(node_class.values())
    all_classes.discard("")
    for cls in sorted(all_classes):
        ctext = type_friendly.get(cls) or _humanize(cls)
        add_entity(cls_key(cls), f"{ctext} — a Freebase type", "type")
        dom = _domain_of(cls)
        if dom:
            add_entity(dom_key(dom), f"{_humanize(dom)} — a Freebase domain", "domain")

    # --- relation vocabulary: hierarchy first, then Freebase relations ---
    relation_vocab: dict[str, int] = dict(HIERARCHY_RELATIONS)
    for _, rel, _, _ in fb_edges:
        if rel not in relation_vocab:
            relation_vocab[rel] = len(relation_vocab)

    r_inst = relation_vocab["instance_of"]
    r_sub  = relation_vocab["subtype_of"]
    r_dom  = relation_vocab["in_domain"]

    # --- hierarchy triples (always-visible structure) + the tree ---
    n0 = len(id_to_entity)
    children_indices: list[list[int]] = [[] for _ in range(n0)]
    hierarchy_triples: list[tuple[int, int, int]] = []

    # domain -> root
    for name, eid in list(entity_vocab.items()):
        if entity_types[eid] == "domain":
            hierarchy_triples.append((eid, r_dom, root_id))
            children_indices[root_id].append(eid)

    # type -> domain
    for cls in sorted(all_classes):
        tid = entity_vocab[cls_key(cls)]
        dom = _domain_of(cls)
        parent = entity_vocab[dom_key(dom)] if dom and dom_key(dom) in entity_vocab else root_id
        hierarchy_triples.append((tid, r_sub, parent))
        children_indices[parent].append(tid)

    # entity -> type (or root if its class is unknown)
    for gid, cls in node_class.items():
        if gid not in entity_vocab:               # gid was a class/literal, not an entity
            continue
        eid = entity_vocab[gid]
        if entity_types[eid] != "entity":
            continue
        parent = entity_vocab[cls_key(cls)] if cls and cls_key(cls) in entity_vocab else root_id
        hierarchy_triples.append((eid, r_inst, parent))
        children_indices[parent].append(eid)
    # entities with no class at all -> attach to root so they still get encoded
    classed = {entity_vocab[g] for g in node_class if g in entity_vocab}
    for eid in range(n0):
        if entity_types[eid] == "entity" and eid not in classed:
            hierarchy_triples.append((eid, r_inst, root_id))
            children_indices[root_id].append(eid)

    # --- Freebase relation triples, split by example origin (leakage-safe) ---
    seen: dict[tuple[int, int, int], str] = {}
    for h, rel, t, origin in fb_edges:
        if h not in entity_vocab or t not in entity_vocab:
            continue
        trip = (entity_vocab[h], relation_vocab[rel], entity_vocab[t])
        # A triple seen in train origin stays train (never demote to test).
        if seen.get(trip) != "train":
            seen[trip] = "train" if origin == "train" else seen.get(trip, origin)
    train_fb = [t for t, o in seen.items() if o == "train"]
    dev_fb   = [t for t, o in seen.items() if o != "train"]

    rng = random.Random(split_seed)
    rng.shuffle(dev_fb)
    n_val = int(len(dev_fb) * val_frac_of_dev)
    val_fb  = dev_fb[:n_val]
    test_fb = dev_fb[n_val:]

    if leakage_safe:
        train_triples = hierarchy_triples + train_fb
        val_triples   = val_fb
        test_triples  = test_fb
    else:
        train_triples = hierarchy_triples + train_fb
        val_triples   = hierarchy_triples + val_fb
        test_triples  = hierarchy_triples + test_fb
    all_triples = hierarchy_triples + train_fb + val_fb + test_fb

    from pluraltree.utils.tree_utils import topological_sort
    topo_order = topological_sort(children_indices)

    # --- type constraints for negative sampling (all non-root entities) ---
    cand = [eid for eid in range(len(id_to_entity)) if eid != root_id]
    type_constraints = {rid: cand for rid in relation_vocab.values()}

    n_ent  = sum(1 for t in entity_types.values() if t == "entity")
    n_type = sum(1 for t in entity_types.values() if t == "type")
    n_dom  = sum(1 for t in entity_types.values() if t == "domain")
    print(f"  GrailQA: {len(id_to_entity)} nodes "
          f"({n_ent} entities, {n_type} types, {n_dom} domains) | "
          f"{len(relation_vocab)} relations "
          f"({len(relation_vocab) - len(HIERARCHY_RELATIONS)} Freebase + 3 hierarchy)")
    print(f"  Freebase triples — train {len(train_fb)} | val {len(val_fb)} | "
          f"test {len(test_fb)} | leakage_safe={leakage_safe}")
    if skipped_literal:
        print(f"  Skipped {skipped_literal} literal nodes (dates/numbers/comparatives)")

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
    )


def load_grailqa(
    data_dir: str = "data/grailqa",
    split_seed: int = 42,
    val_frac_of_dev: float = 0.5,
    leakage_safe: bool = True,
) -> CulturalGraph:
    """Load GrailQA (train + dev) from local JSON files and build the graph.

    The HF ``dki-lab/grail_qa`` dataset ships as a loader *script*, which modern
    ``datasets`` refuses to run, so we read the official release JSON directly.
    Expected files under ``data_dir`` (fetched by ``scripts/get_grailqa.py``):

        grailqa_v1.0_train.json
        grailqa_v1.0_dev.json

    The hidden ``test_public`` split has no annotations, so link-prediction
    triples come from train (train edges) and dev (val/test edges).
    """
    import json
    import os

    def _read(fname: str) -> list[dict]:
        for cand in (os.path.join(data_dir, fname),
                     os.path.join(data_dir, "GrailQA_v1.0", fname)):
            if os.path.exists(cand):
                with open(cand, encoding="utf-8") as f:
                    return json.load(f)
        return []

    train_rows = _read("grailqa_v1.0_train.json")
    dev_rows   = _read("grailqa_v1.0_dev.json")
    if not train_rows or not dev_rows:
        raise FileNotFoundError(
            f"GrailQA JSON not found under {data_dir!r}. Fetch it once with "
            f"`python scripts/get_grailqa.py --out {data_dir}` (needs network)."
        )

    rows = [{"split": "train", "graph_query": s.get("graph_query")} for s in train_rows]
    rows += [{"split": "dev", "graph_query": s.get("graph_query")} for s in dev_rows]
    return _build_graph(rows, split_seed=split_seed,
                        val_frac_of_dev=val_frac_of_dev, leakage_safe=leakage_safe)


# Node features are plain entity text -> reuse the CulturalBench encoder.
from data.culturalbench import compute_text_embeddings  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Offline self-test (no network): a tiny 2-question synthetic GrailQA graph
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = [
        {
            "split": "train",
            "graph_query": {
                "nodes": [
                    {"nid": 0, "node_type": "class", "id": "film.film",
                     "class": "film.film", "friendly_name": "Film"},
                    {"nid": 1, "node_type": "entity", "id": "m.0dir",
                     "class": "film.director", "friendly_name": "Some Director"},
                ],
                "edges": [
                    {"start": 0, "end": 1, "relation": "film.film.directed_by",
                     "friendly_name": "directed by"},
                ],
            },
        },
        {
            "split": "dev",
            "graph_query": {
                # struct-of-lists form (the other way HF can return it)
                "nodes": {
                    "nid": [0, 1],
                    "node_type": ["class", "entity"],
                    "id": ["people.person", "m.0nat"],
                    "class": ["people.person", "location.country"],
                    "friendly_name": ["Person", "France"],
                },
                "edges": {
                    "start": [0], "end": [1],
                    "relation": ["people.person.nationality"],
                    "friendly_name": ["nationality"],
                },
            },
        },
    ]
    g = _build_graph(sample, leakage_safe=True)
    print("entities:", g.id_to_entity)
    print("relations:", g.relation_vocab)
    print("train:", g.train_triples)
    print("val:", g.val_triples, "test:", g.test_triples)
    print("children_indices:", g.children_indices)
    print("topo_order:", g.topo_order)
    # sanity: every non-root node reaches the root via the tree
    assert len(g.topo_order) == len(g.id_to_entity), "topo order must cover all nodes"
    print("OK")
