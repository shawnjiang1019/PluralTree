"""OpinionQA/SubPOP dataset loader — US demographic opinion graph.

Builds a knowledge graph from SubPOP (Suh et al., ACL 2025, HF jjssuh/subpop):
~3.2k Pew American Trends Panel questions with post-stratification-weighted
answer distributions for 22 US subpopulations across 8 demographic attributes
(political ideology, party, religion, income, education, race, gender, region).

NOTE: the HF dataset is GATED (CC-BY-NC-SA). Accept the terms on the dataset
page with your HF account, then ``huggingface-cli login`` before first load.

GOQA's World->Region->Country tree is only depth 4; here we mint topic layers
by clustering question embeddings so the hierarchy is deep enough for the
geometry (rho/abstraction) and for multi-level forks:

    US Public -> topic -> subtopic -> question -> attribute axis -> opinion

  * axis node   = one demographic attribute's split on one question. Forks at
    the QUESTION level compare axes ("is the ideology divide bigger than the
    age divide here?"); forks at the AXIS level compare subgroups (liberal vs
    conservative) — the OvertonBench-relevant fork type.
  * opinion leaf = one (question, attribute, group) with its distribution;
    feature = distribution-weighted MiniLM barycenter (see GOQA's
    ``compute_features``, reused unchanged via subclassing).

Relations:
    answered_by : opinion  -> axis      (link-prediction signal, split)
    splits      : axis     -> question  (train-only structural)
    about       : question -> subtopic
    within      : subtopic -> topic
    part_of     : topic    -> root
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import torch

from data.loaders.globalopinionqa import GlobalOpinionGraph

RELATION_VOCAB: dict[str, int] = {
    "answered_by": 0,
    "splits":      1,
    "about":       2,
    "within":      3,
    "part_of":     4,
}


@dataclass
class OpinionQAGraph(GlobalOpinionGraph):
    """Same container; opinion_texts/opinion_dist inherited (features reuse)."""


# ---------------------------------------------------------------------------
# Row parsing (tolerant: SubPOP JSONL field names may drift between versions)
# ---------------------------------------------------------------------------
def _field(row: dict, *names, default=None):
    for n in names:
        v = row.get(n)
        if v is not None:
            return v
    return default


def parse_records(rows) -> list[dict]:
    """Normalize raw rows to {qkey, question, options, attribute, group, dist}."""
    out, skipped = [], 0
    for row in rows:
        q = str(_field(row, "question", "question_text", "question_refined",
                       "question_raw", default="")).strip()
        options = _field(row, "options", "answers", "choices")
        attr = _field(row, "attribute", "attr", "subpopulation_attribute")
        group = _field(row, "group", "subpopulation", "subgroup")
        resp = _field(row, "responses", "distribution", "response_distribution",
                      "output_distribution")
        if not q or not options or attr is None or group is None or resp is None:
            skipped += 1
            continue
        options = [str(o) for o in options]
        if isinstance(resp, dict):                      # option -> prob form
            probs = [float(resp.get(o, 0.0)) for o in options]
        else:
            probs = [float(p) for p in resp]
        if len(probs) != len(options) or sum(probs) <= 0:
            skipped += 1
            continue
        total = sum(probs)
        probs = [p / total for p in probs]
        qkey = str(_field(row, "qkey", "question_id", "key",
                          default=hashlib.md5(q.encode()).hexdigest()[:10]))
        out.append({"qkey": qkey, "question": q, "options": options,
                    "attribute": str(attr), "group": str(group), "dist": probs})
    if skipped:
        print(f"  OpinionQA: skipped {skipped} malformed rows")
    return out


# ---------------------------------------------------------------------------
# Topic layers: seeded k-means over question embeddings (no sklearn dep)
# ---------------------------------------------------------------------------
def _kmeans(X: torch.Tensor, k: int, seed: int, iters: int = 50) -> torch.Tensor:
    """(n,) cluster assignment; deterministic given seed."""
    k = min(k, X.shape[0])
    gen = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(X.shape[0], generator=gen)[:k]].clone()
    assign = torch.zeros(X.shape[0], dtype=torch.long)
    for _ in range(iters):
        assign = torch.cdist(X, C).argmin(dim=1)
        for j in range(k):
            m = assign == j
            if m.any():
                C[j] = X[m].mean(0)
    return assign


def cluster_questions(
    q_texts: list[str],
    *,
    k_topics: int = 12,
    k_subtopics: int = 4,
    seed: int = 0,
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[list[tuple[int, int]], dict[int, str], dict[tuple[int, int], str]]:
    """Assign each question a (topic, subtopic); label clusters by central questions.

    Returns (assignments, topic_text, sub_text). Labels are the 3 questions
    nearest each centroid — good enough for MiniLM node features.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    X = model.encode(q_texts, convert_to_tensor=True,
                     show_progress_bar=True).cpu().float()

    topics = _kmeans(X, k_topics, seed)
    assignments: list[tuple[int, int]] = [(0, 0)] * len(q_texts)
    topic_text: dict[int, str] = {}
    sub_text: dict[tuple[int, int], str] = {}

    def label(idx: torch.Tensor) -> str:
        sub = X[idx]
        d = torch.cdist(sub, sub.mean(0, keepdim=True)).squeeze(1)
        near = idx[d.argsort()[:3]]
        return "; ".join(" ".join(q_texts[i].split()[:12]) for i in near.tolist())

    for t in range(int(topics.max()) + 1):
        members = (topics == t).nonzero().squeeze(1)
        if members.numel() == 0:
            continue
        topic_text[t] = label(members)
        subs = _kmeans(X[members], k_subtopics, seed + t + 1)
        for s in range(int(subs.max()) + 1):
            sm = members[subs == s]
            if sm.numel() == 0:
                continue
            sub_text[(t, s)] = label(sm)
            for i in sm.tolist():
                assignments[i] = (t, s)
    return assignments, topic_text, sub_text


# ---------------------------------------------------------------------------
# Builder (network-free: records + cluster assignments in, graph out)
# ---------------------------------------------------------------------------
def _build_graph(
    records: list[dict],
    topic_of: dict[str, tuple[int, int]],
    topic_text: dict[int, str],
    sub_text: dict[tuple[int, int], str],
    *,
    split_seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    leakage_safe: bool = True,
) -> OpinionQAGraph:
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

    root_id = add_entity("US_Public", "The United States public — all demographic "
                         "groups and their opinions", "root")

    r_ans, r_spl, r_abt, r_win, r_prt = (RELATION_VOCAB[k] for k in
                                         ("answered_by", "splits", "about",
                                          "within", "part_of"))

    structural: list[tuple[int, int, int]] = []
    opinion_cells: list[tuple[int, int]] = []        # (opinion_id, axis_id)
    seen_q: set[str] = set()

    for rec in records:
        qkey, q = rec["qkey"], rec["question"]
        t, s = topic_of[qkey]
        tid = add_entity(f"topic:{t}", f"Survey topic: {topic_text[t]}", "topic")
        sid = add_entity(f"sub:{t}.{s}", f"Survey subtopic: {sub_text[(t, s)]}",
                         "subtopic")
        qid = add_entity(f"q:{qkey}", q[:300], "question")
        if qkey not in seen_q:
            seen_q.add(qkey)
            structural += [(tid, r_prt, root_id), (sid, r_win, tid),
                           (qid, r_abt, sid)]

        attr, group = rec["attribute"], rec["group"]
        ax_name = f"ax:{qkey}:{attr}"
        new_axis = ax_name not in entity_vocab
        aid = add_entity(ax_name, f"How {attr} divides opinion on: {q[:200]}",
                         "axis")
        if new_axis:
            structural.append((aid, r_spl, qid))

        oid = add_entity(f"op:{qkey}:{attr}:{group}",
                         f"{q[:160]} [{attr}={group}]", "opinion")
        opinion_texts[oid] = [f"{q} {opt}" for opt in rec["options"]]
        opinion_dist[oid] = rec["dist"]
        opinion_cells.append((oid, aid))

    # Deduplicate structural triples (topic/sub re-added per record).
    structural = sorted(set(structural))

    # --- opinion triples + split (opinions only, same scheme as GOQA) --------
    opinion_triples = [(oid, r_ans, aid) for oid, aid in opinion_cells]
    rng = random.Random(split_seed)
    shuffled = opinion_triples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_op = shuffled[:n_train]
    val_op = shuffled[n_train:n_train + n_val]
    test_op = shuffled[n_train + n_val:]

    train_triples = structural + train_op
    if leakage_safe:
        val_triples, test_triples = val_op, test_op
        tree_op = train_op
    else:
        val_triples = structural + val_op
        test_triples = structural + test_op
        tree_op = opinion_triples
    all_triples = structural + shuffled

    # --- tree -----------------------------------------------------------------
    n_entities = len(id_to_entity)
    children_indices: list[list[int]] = [[] for _ in range(n_entities)]
    for h, r, tl in structural:                      # child -> parent edges
        children_indices[tl].append(h)
    for oid, _, aid in tree_op:
        children_indices[aid].append(oid)

    from pluraltree.utils.tree_utils import topological_sort
    topo_order = topological_sort(children_indices)

    by_type = lambda t: [nid for nid, ty in entity_types.items() if ty == t]
    type_constraints = {
        r_ans: by_type("axis"),
        r_spl: by_type("question"),
        r_abt: by_type("subtopic"),
        r_win: by_type("topic"),
        r_prt: [root_id],
    }

    print(f"  OpinionQA/SubPOP: {n_entities} entities "
          f"({len(by_type('topic'))} topics, {len(by_type('subtopic'))} subtopics, "
          f"{len(seen_q)} questions, {len(by_type('axis'))} axes, "
          f"{len(by_type('opinion'))} opinions)")
    print(f"  Opinion triples — train {len(train_op)} | val {len(val_op)} | "
          f"test {len(test_op)} | leakage_safe={leakage_safe}")

    return OpinionQAGraph(
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


def load_opinionqa(
    split_seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    leakage_safe: bool = True,
    hf_split: str = "train",
    k_topics: int = 12,
    k_subtopics: int = 4,
    attributes: list[str] | None = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> OpinionQAGraph:
    """Load SubPOP from HF (gated — accept terms + huggingface-cli login first).

    ``attributes`` optionally restricts to a subset of demographic axes
    (e.g. ["POLIDEOLOGY", "POLPARTY"]).
    """
    from datasets import load_dataset

    ds = load_dataset("jjssuh/subpop", split=hf_split)
    records = parse_records(dict(r) for r in ds)
    if attributes:
        want = {a.lower() for a in attributes}
        records = [r for r in records if r["attribute"].lower() in want]
        print(f"  OpinionQA: kept {len(records)} records for attributes {sorted(want)}")

    # Cluster once per unique question, then map assignments back by qkey.
    qkeys, qtexts = [], []
    seen: set[str] = set()
    for r in records:
        if r["qkey"] not in seen:
            seen.add(r["qkey"])
            qkeys.append(r["qkey"])
            qtexts.append(r["question"])
    assign, topic_text, sub_text = cluster_questions(
        qtexts, k_topics=k_topics, k_subtopics=k_subtopics,
        seed=split_seed, model_name=model_name)
    topic_of = {k: a for k, a in zip(qkeys, assign)}

    return _build_graph(records, topic_of, topic_text, sub_text,
                        split_seed=split_seed, train_frac=train_frac,
                        val_frac=val_frac, leakage_safe=leakage_safe)
