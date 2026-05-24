"""Tests for GKI gate mechanisms."""

import torch
import pytest

from pluraltree.manifolds.poincare import PoincareBall
from pluraltree.gki.gate import EuclideanGate, HyperbolicGate
from pluraltree.combined.depth_aware_gate import DepthAwareHyperbolicGate


@pytest.fixture
def ball():
    return PoincareBall(c=1.0)


class TestEuclideanGate:
    def test_output_shape(self):
        gate = EuclideanGate(32, 32)
        h = torch.randn(8, 32)
        k = torch.randn(8, 32)
        g = gate(h, k)
        assert g.shape == (8, 32)

    def test_output_range(self):
        gate = EuclideanGate(32, 32)
        h = torch.randn(8, 32)
        k = torch.randn(8, 32)
        g = gate(h, k)
        assert (g >= 0).all() and (g <= 1).all()

    def test_negative_bias_starts_closed(self):
        gate = EuclideanGate(32, 32, bias_init=-5.0)
        h = torch.zeros(8, 32)
        k = torch.zeros(8, 32)
        g = gate(h, k)
        assert g.mean() < 0.1  # sigmoid(-5) ≈ 0.007


class TestHyperbolicGate:
    def test_output_shape(self, ball):
        gate = HyperbolicGate(32, 32, ball)
        h = torch.randn(8, 32) * 0.3
        k = torch.randn(8, 32) * 0.3
        h, k = ball.project(h), ball.project(k)
        g = gate(h, k)
        assert g.shape == (8, 32)

    def test_output_range(self, ball):
        gate = HyperbolicGate(32, 32, ball)
        h = torch.randn(8, 32) * 0.3
        k = torch.randn(8, 32) * 0.3
        h, k = ball.project(h), ball.project(k)
        g = gate(h, k)
        assert (g >= 0).all() and (g <= 1).all()


class TestDepthAwareGate:
    def test_output_shape(self, ball):
        gate = DepthAwareHyperbolicGate(32, 32, ball)
        h = torch.randn(8, 32) * 0.3
        k = torch.randn(8, 32) * 0.3
        h, k = ball.project(h), ball.project(k)
        g = gate(h, k)
        assert g.shape == (8, 32)

    def test_rho_range(self, ball):
        gate = DepthAwareHyperbolicGate(32, 32, ball)
        h = torch.randn(8, 32) * 0.5
        h = ball.project(h)
        rho = gate.compute_rho(h)
        assert (rho >= 0).all() and (rho < 1).all()

    def test_depth_affects_blending(self, ball):
        gate = DepthAwareHyperbolicGate(32, 32, ball)
        k_broad = torch.randn(1, 32) * 0.1
        k_precise = torch.randn(1, 32) * 0.1
        k_broad, k_precise = ball.project(k_broad), ball.project(k_precise)

        # Near-origin point (abstract)
        h_shallow = torch.randn(1, 32) * 0.05
        h_shallow = ball.project(h_shallow)
        rho_shallow = gate.compute_rho(h_shallow)

        # Near-boundary point (specific)
        h_deep = torch.randn(1, 32)
        h_deep = h_deep / h_deep.norm(dim=-1, keepdim=True) * 0.9
        h_deep = ball.project(h_deep)
        rho_deep = gate.compute_rho(h_deep)

        assert rho_deep > rho_shallow
