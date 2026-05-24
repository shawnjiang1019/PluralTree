"""Tests for the Hyperbolic Tree-GRU cell."""

import torch
import pytest

from pluraltree.manifolds.poincare import PoincareBall
from pluraltree.manifolds.math_utils import safe_norm
from pluraltree.tree_gru.cell import HyperbolicTreeGRUCell
from pluraltree.tree_gru.encoder import TreeEncoder


@pytest.fixture
def ball():
    return PoincareBall(c=1.0)


class TestTreeGRUCell:
    def test_output_shape(self, ball):
        cell = HyperbolicTreeGRUCell(16, 32, ball)
        x_v = torch.randn(4, 16)  # batch=4, d_input=16
        h_ch = torch.randn(3, 4, 32) * 0.3  # 3 children
        h_ch = ball.project(h_ch)
        mask = torch.ones(3, 4, dtype=torch.bool)
        h = cell(x_v, h_ch, mask)
        assert h.shape == (4, 32)

    def test_output_on_ball(self, ball):
        cell = HyperbolicTreeGRUCell(16, 32, ball)
        x_v = torch.randn(4, 16)
        h_ch = torch.randn(3, 4, 32) * 0.3
        h_ch = ball.project(h_ch)
        mask = torch.ones(3, 4, dtype=torch.bool)
        h = cell(x_v, h_ch, mask)
        norms = safe_norm(h, dim=-1).squeeze(-1)
        assert (norms < 1.0).all()

    def test_leaf_node(self, ball):
        """Leaf nodes with a single zero-child should still produce valid output."""
        cell = HyperbolicTreeGRUCell(16, 32, ball)
        x_v = torch.randn(1, 16)
        h_ch = torch.zeros(1, 1, 32)  # single zero child
        mask = torch.ones(1, 1, dtype=torch.bool)
        h = cell(x_v, h_ch, mask)
        assert torch.isfinite(h).all()


class TestTreeEncoder:
    def test_simple_tree(self, ball):
        """Test on a simple 3-node tree: root(0) -> child(1), child(2)."""
        encoder = TreeEncoder(8, 16, ball)
        features = torch.randn(3, 8)
        children = [[1, 2], [], []]  # node 0 has children 1, 2
        topo = [1, 2, 0]  # leaves first
        h = encoder(features, children, topo)
        assert h.shape == (3, 16)
        assert torch.isfinite(h).all()

    def test_deep_tree(self, ball):
        """Test on a chain: 0 -> 1 -> 2 -> 3."""
        encoder = TreeEncoder(8, 16, ball)
        features = torch.randn(4, 8)
        children = [[1], [2], [3], []]
        topo = [3, 2, 1, 0]
        h = encoder(features, children, topo)
        assert h.shape == (4, 16)
        assert torch.isfinite(h).all()
        # All outputs should be on the ball
        norms = safe_norm(h, dim=-1).squeeze(-1)
        assert (norms < 1.0).all()
