"""Tests for the combined GKI + Tree-GRU model."""

import torch
import pytest

from pluraltree.manifolds.poincare import PoincareBall
from pluraltree.manifolds.math_utils import safe_norm
from pluraltree.knowledge.kg_embedding import KGEmbeddingSource
from pluraltree.combined.gki_tree_gru_cell import GKITreeGRUCell, InjectionPoint
from pluraltree.combined.gki_tree_encoder import GKITreeEncoder
from pluraltree.combined.knowledge_schedule import KnowledgeSchedule


@pytest.fixture
def ball():
    return PoincareBall(c=1.0)


@pytest.fixture
def source():
    return KGEmbeddingSource(num_entities=100, embedding_dim=16, freeze=False)


class TestGKITreeGRUCell:
    @pytest.mark.parametrize("injection", list(InjectionPoint))
    def test_all_injection_points(self, ball, source, injection):
        cell = GKITreeGRUCell(16, 32, ball, [source], injection_point=injection)
        x_v = torch.randn(2, 16)
        h_ch = torch.randn(3, 2, 32) * 0.3
        h_ch = ball.project(h_ch)
        node_ids = torch.randint(0, 100, (2,))
        child_ids = torch.randint(0, 100, (3, 2))
        mask = torch.ones(3, 2, dtype=torch.bool)
        h = cell(x_v, h_ch, node_ids, child_ids, mask)
        assert h.shape == (2, 32)
        assert torch.isfinite(h).all()

    def test_output_on_ball(self, ball, source):
        cell = GKITreeGRUCell(16, 32, ball, [source])
        x_v = torch.randn(4, 16)
        h_ch = torch.randn(2, 4, 32) * 0.3
        h_ch = ball.project(h_ch)
        node_ids = torch.randint(0, 100, (4,))
        child_ids = torch.randint(0, 100, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.bool)
        h = cell(x_v, h_ch, node_ids, child_ids, mask)
        norms = safe_norm(h, dim=-1).squeeze(-1)
        assert (norms < 1.0).all()


class TestGKITreeEncoder:
    def test_simple_tree(self, ball, source):
        encoder = GKITreeEncoder(8, 16, ball, [source])
        features = torch.randn(3, 8)
        node_ids = torch.tensor([0, 1, 2])
        children = [[1, 2], [], []]
        topo = [1, 2, 0]
        h = encoder(features, node_ids, children, topo)
        assert h.shape == (3, 16)
        assert torch.isfinite(h).all()


class TestKnowledgeSchedule:
    def test_gate_bias_phases(self):
        sched = KnowledgeSchedule(warmup1=100, warmup2=200)
        assert sched.get_gate_bias(0) == -2.0
        assert sched.get_gate_bias(50) == -2.0
        assert sched.get_gate_bias(150) == pytest.approx(-1.0, abs=0.01)
        assert sched.get_gate_bias(300) == 0.0

    def test_active_sources_phases(self):
        sched = KnowledgeSchedule(warmup1=100, warmup2=200)
        assert sched.get_active_sources(0) == ["kg"]
        assert sched.get_active_sources(50) == ["kg"]
        assert sched.get_active_sources(150) == ["kg", "structured"]
        assert sched.get_active_sources(300) == ["kg", "structured", "retrieval"]
