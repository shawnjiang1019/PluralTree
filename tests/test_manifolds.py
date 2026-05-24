"""Tests for Poincaré ball manifold operations."""

import torch
import pytest

from pluraltree.manifolds.poincare import PoincareBall
from pluraltree.manifolds.math_utils import safe_artanh, safe_norm


@pytest.fixture
def ball():
    return PoincareBall(c=1.0)


@pytest.fixture
def ball_c2():
    return PoincareBall(c=2.0)


class TestMathUtils:
    def test_safe_artanh_within_range(self):
        x = torch.tensor([0.0, 0.5, -0.5, 0.99])
        result = safe_artanh(x)
        assert torch.isfinite(result).all()

    def test_safe_artanh_at_boundary(self):
        x = torch.tensor([1.0, -1.0])
        result = safe_artanh(x)
        assert torch.isfinite(result).all()

    def test_safe_norm_nonzero(self):
        x = torch.zeros(3)
        result = safe_norm(x)
        assert result > 0


class TestExpLogRoundTrip:
    def test_exp_log_zero_roundtrip(self, ball):
        v = torch.randn(5, 8) * 0.3
        x = ball.exp_map_zero(v)
        v_rec = ball.log_map_zero(x)
        assert torch.allclose(v, v_rec, atol=1e-5)

    def test_log_exp_zero_roundtrip(self, ball):
        x = torch.randn(5, 8) * 0.3
        x = ball.project(x)
        v = ball.log_map_zero(x)
        x_rec = ball.exp_map_zero(v)
        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_roundtrip_different_curvature(self, ball_c2):
        v = torch.randn(5, 8) * 0.2
        x = ball_c2.exp_map_zero(v)
        v_rec = ball_c2.log_map_zero(x)
        assert torch.allclose(v, v_rec, atol=1e-5)


class TestMobiusAdd:
    def test_identity(self, ball):
        x = torch.randn(5, 8) * 0.3
        x = ball.project(x)
        zero = torch.zeros_like(x)
        result = ball.mobius_add(x, zero)
        assert torch.allclose(x, result, atol=1e-5)

    def test_inverse(self, ball):
        x = torch.randn(5, 8) * 0.3
        x = ball.project(x)
        result = ball.mobius_add(x, -x)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-4)


class TestDistance:
    def test_self_distance_zero(self, ball):
        x = torch.randn(5, 8) * 0.3
        x = ball.project(x)
        d = ball.distance(x, x)
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-5)

    def test_distance_symmetry(self, ball):
        # Use smaller scale to keep points away from boundary where precision degrades
        x = torch.randn(5, 8) * 0.1
        y = torch.randn(5, 8) * 0.1
        x, y = ball.project(x), ball.project(y)
        assert torch.allclose(ball.distance(x, y), ball.distance(y, x), atol=1e-5)

    def test_distance_positive(self, ball):
        x = torch.randn(5, 8) * 0.3
        y = torch.randn(5, 8) * 0.3
        x, y = ball.project(x), ball.project(y)
        assert (ball.distance(x, y) >= 0).all()


class TestMobiusMidpoint:
    def test_single_point_on_ball(self, ball):
        """Einstein midpoint of a single point stays on the ball."""
        x = torch.randn(1, 3, 8) * 0.3
        x = ball.project(x)
        w = torch.ones(1, 3, 1)
        mid = ball.mobius_midpoint(x, w)
        norms = safe_norm(mid, dim=-1)
        assert (norms < 1.0).all()

    def test_midpoint_between_points(self, ball):
        """Midpoint of two points lies between them (triangle inequality)."""
        x = torch.randn(2, 1, 8) * 0.2
        x = ball.project(x)
        w = torch.ones(2, 1, 1) * 0.5
        mid = ball.mobius_midpoint(x, w)
        d_total = ball.distance(x[0], x[1])
        d0 = ball.distance(mid, x[0])
        d1 = ball.distance(mid, x[1])
        # Midpoint should be closer to each point than the points are to each other
        assert (d0 < d_total).all()
        assert (d1 < d_total).all()

    def test_midpoint_on_ball(self, ball):
        x = torch.randn(3, 5, 8) * 0.3
        x = ball.project(x)
        w = torch.ones(3, 5, 1) / 3
        mid = ball.mobius_midpoint(x, w)
        norms = safe_norm(mid, dim=-1)
        assert (norms < 1.0).all()


class TestHyperboloidConversion:
    def test_roundtrip(self, ball):
        x = torch.randn(5, 8) * 0.3
        x = ball.project(x)
        h = ball.to_hyperboloid(x)
        x_rec = ball.from_hyperboloid(h)
        assert torch.allclose(x, x_rec, atol=1e-4)

    def test_hyperboloid_constraint(self, ball):
        """Points on the hyperboloid satisfy -t^2 + ||s||^2 = -1/c."""
        # Use float64 and moderate norms to avoid precision loss near boundary
        x = torch.randn(5, 8, dtype=torch.float64) * 0.1
        ball64 = PoincareBall(c=1.0)
        ball64 = ball64.double()
        x = ball64.project(x)
        h = ball64.to_hyperboloid(x)
        t = h[..., :1]
        s = h[..., 1:]
        constraint = -t ** 2 + torch.sum(s ** 2, dim=-1, keepdim=True)
        expected = -1.0 / ball64.c
        assert torch.allclose(constraint, expected * torch.ones_like(constraint), atol=1e-8)


class TestNumericalStability:
    def test_near_boundary(self, ball):
        x = torch.randn(5, 8)
        x = x / x.norm(dim=-1, keepdim=True) * 0.999
        x = ball.project(x)
        v = ball.log_map_zero(x)
        assert torch.isfinite(v).all()

    def test_very_small_vectors(self, ball):
        x = torch.randn(5, 8) * 1e-8
        v = ball.log_map_zero(ball.exp_map_zero(x))
        assert torch.isfinite(v).all()
