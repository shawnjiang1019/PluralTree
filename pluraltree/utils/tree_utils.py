"""Utilities for tree structure manipulation — topological sort, batching, padding."""

from collections import deque
from torch import Tensor


def topological_sort(children_indices: list[list[int]]) -> list[int]:
    """Compute topological order (leaves first, root last) for a rooted tree.

    Args:
        children_indices: children_indices[i] = list of child node indices for node i

    Returns:
        List of node indices in topological order (leaves processed first)
    """
    N = len(children_indices)
    # Count parents for each node
    parent_count = [0] * N
    for i, children in enumerate(children_indices):
        for c in children:
            parent_count[c] += 1

    # Find root(s) — nodes with no parent
    # Start BFS from leaves (nodes with no children)
    in_degree_from_children = [0] * N
    for i, children in enumerate(children_indices):
        in_degree_from_children[i] = len(children)

    queue = deque()
    for i in range(N):
        if in_degree_from_children[i] == 0:
            queue.append(i)

    # Build reverse adjacency (child -> parent)
    parent_of = [[] for _ in range(N)]
    for i, children in enumerate(children_indices):
        for c in children:
            parent_of[c].append(i)

    order = []
    processed = [False] * N

    while queue:
        node = queue.popleft()
        if processed[node]:
            continue
        order.append(node)
        processed[node] = True
        for parent in parent_of[node]:
            in_degree_from_children[parent] -= 1
            if in_degree_from_children[parent] == 0:
                queue.append(parent)

    return order


def find_root(children_indices: list[list[int]]) -> int:
    """Find the root node (the node that is no other node's child)."""
    N = len(children_indices)
    is_child = set()
    for children in children_indices:
        for c in children:
            is_child.add(c)
    roots = [i for i in range(N) if i not in is_child]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly 1 root, found {len(roots)}: {roots}")
    return roots[0]


def tree_depth(children_indices: list[list[int]], root: int | None = None) -> dict[int, int]:
    """Compute the depth of each node (root = 0)."""
    if root is None:
        root = find_root(children_indices)
    depths = {}
    queue = deque([(root, 0)])
    while queue:
        node, d = queue.popleft()
        depths[node] = d
        for c in children_indices[node]:
            queue.append((c, d + 1))
    return depths
