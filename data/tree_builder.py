"""Convert a CulturalGraph into batched tree inputs for the encoder.

The full CulturalBench hierarchy is one large rooted tree:
    World → Regions → Countries → Practices

This module packages that tree (or subtrees) into the tensor format
expected by TreeEncoder / GKITreeEncoder.
"""

from __future__ import annotations

import torch
from torch import Tensor

from data.loaders.culturalbench import CulturalGraph


def build_full_tree_inputs(
    graph: CulturalGraph,
    node_embeddings: Tensor,
) -> dict:
    """Package the full hierarchy as encoder inputs.

    Args:
        graph: CulturalGraph from load_culturalbench()
        node_embeddings: (N, d_input) pre-computed entity embeddings

    Returns:
        dict with keys:
            node_features     : (N, d_input) Tensor
            node_ids          : (N,) LongTensor of entity ids
            children_indices  : list[list[int]]
            topo_order        : list[int]
    """
    n = len(graph.id_to_entity)
    node_ids = torch.arange(n, dtype=torch.long)

    return {
        "node_features":    node_embeddings,        # (N, d_input)
        "node_ids":         node_ids,               # (N,)
        "children_indices": graph.children_indices, # list[list[int]]
        "topo_order":       graph.topo_order,       # list[int]
    }


def build_subtree_inputs(
    graph: CulturalGraph,
    node_embeddings: Tensor,
    root_entity: str,
) -> dict:
    """Package a subtree rooted at a given entity as encoder inputs.

    Useful for training on individual country or region subtrees rather
    than the full world tree.

    Args:
        graph: CulturalGraph
        node_embeddings: (N, d_input) embeddings for ALL entities
        root_entity: entity name to use as subtree root (e.g. "Japan")

    Returns:
        Same dict format as build_full_tree_inputs, but for the subtree only.
        node_ids maps back to original entity ids for knowledge lookup.
    """
    root_id = graph.entity_vocab[root_entity]

    # BFS to collect all nodes in the subtree
    from collections import deque
    subtree_nodes: list[int] = []
    queue = deque([root_id])
    visited: set[int] = set()
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        subtree_nodes.append(node)
        for child in graph.children_indices[node]:
            queue.append(child)

    # Remap original ids to local subtree ids
    local_id: dict[int, int] = {orig: i for i, orig in enumerate(subtree_nodes)}
    n_local = len(subtree_nodes)

    local_children: list[list[int]] = [[] for _ in range(n_local)]
    for orig_id in subtree_nodes:
        local = local_id[orig_id]
        for child_orig in graph.children_indices[orig_id]:
            if child_orig in local_id:
                local_children[local].append(local_id[child_orig])

    from pluraltree.utils.tree_utils import topological_sort
    local_topo = topological_sort(local_children)

    # Gather embeddings and ids for subtree nodes
    orig_ids_tensor = torch.tensor(subtree_nodes, dtype=torch.long)
    subtree_features = node_embeddings[orig_ids_tensor]  # (n_local, d_input)

    return {
        "node_features":    subtree_features,
        "node_ids":         orig_ids_tensor,   # original ids for knowledge lookup
        "children_indices": local_children,
        "topo_order":       local_topo,
    }


def get_triple_node_ids(
    triples: list[tuple[int, int, int]],
) -> tuple[Tensor, Tensor, Tensor]:
    """Unpack a list of triples into subject, relation, object tensors.

    Args:
        triples: list of (s_id, r_id, o_id)

    Returns:
        s_ids: (N,) LongTensor
        r_ids: (N,) LongTensor
        o_ids: (N,) LongTensor
    """
    if not triples:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty, empty
    s, r, o = zip(*triples)
    return (
        torch.tensor(s, dtype=torch.long),
        torch.tensor(r, dtype=torch.long),
        torch.tensor(o, dtype=torch.long),
    )
