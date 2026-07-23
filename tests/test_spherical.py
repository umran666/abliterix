"""Tests for Spherical Steering (geodesic rotation on the activation hypersphere)."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from abliterix.core.steering import (
    _make_angular_hook,
    _make_spherical_hook,
    apply_steering,
)
from abliterix.settings import AbliterixConfig
from abliterix.types import SteeringProfile


class TestSphericalHook:
    """Tests for _make_spherical_hook."""

    def test_preserves_norm(self):
        """Spherical steering must preserve the activation norm."""
        torch.manual_seed(42)
        d = F.normalize(torch.randn(64), p=2, dim=0)
        hook = _make_spherical_hook(d, angle_degrees=90.0)

        h = torch.randn(2, 10, 64)  # (batch, seq, hidden)
        original_norms = h.norm(dim=-1)

        # Simulate hook call.
        h_new = hook(None, None, h)

        new_norms = h_new.norm(dim=-1)
        torch.testing.assert_close(new_norms, original_norms, atol=1e-4, rtol=1e-4)

    def test_moves_toward_removal_tangent(self):
        """Steering reduces each activation's directional projection."""
        torch.manual_seed(42)
        d = F.normalize(torch.randn(64), p=2, dim=0)
        hook = _make_spherical_hook(d, angle_degrees=45.0)

        h = torch.randn(1, 5, 64)
        proj_before = (F.normalize(h, p=2, dim=-1) @ d).abs()

        h_new = hook(None, None, h)
        proj_after = (F.normalize(h_new, p=2, dim=-1) @ d).abs()

        assert torch.all(proj_after < proj_before)

    def test_zero_angle_is_identity(self):
        """With angle=0, spherical steering should return the original activation."""
        torch.manual_seed(42)
        d = F.normalize(torch.randn(64), p=2, dim=0)
        hook = _make_spherical_hook(d, angle_degrees=0.0)

        h = torch.randn(1, 3, 64)
        h_new = hook(None, None, h)

        torch.testing.assert_close(h_new, h, atol=1e-5, rtol=1e-5)

    def test_zero_angle_is_exact_identity_for_zero_activation(self):
        d = F.normalize(torch.tensor([1.0, 2.0, 3.0]), p=2, dim=0)
        h = torch.zeros(1, 1, 3)
        hook = _make_spherical_hook(d, angle_degrees=0.0)

        h_new = hook(None, None, h)

        assert torch.equal(h_new, h)

    def test_zero_activation_remains_zero_at_maximum_angle(self):
        d = F.normalize(torch.tensor([1.0, 2.0, 3.0]), p=2, dim=0)
        h = torch.zeros(1, 1, 3)
        hook = _make_spherical_hook(d, angle_degrees=90.0)

        h_new = hook(None, None, h)

        assert torch.equal(h_new, h)

    def test_equivalent_to_angular_for_2d_rotation(self):
        """Spherical and angular produce equivalent results for 2D rotation.

        Both methods rotate in the plane spanned by h and d, which is
        mathematically equivalent to geodesic rotation on the hypersphere.
        This test validates that our spherical implementation is correct by
        checking consistency with the known-good angular implementation.
        """
        torch.manual_seed(42)
        d = F.normalize(torch.randn(64), p=2, dim=0)

        spherical_hook = _make_spherical_hook(d, angle_degrees=45.0)
        angular_hook = _make_angular_hook(d, angle_degrees=45.0, adaptive=False)

        h = torch.randn(1, 10, 64)
        h_spherical = spherical_hook(None, None, h.clone())
        h_angular = angular_hook(None, None, h.clone())

        # The two methods should produce (nearly) identical results.
        torch.testing.assert_close(h_spherical, h_angular, atol=1e-4, rtol=1e-4)

    def test_maximum_angle_removes_direction_without_crossing_it(self):
        """A 90-degree budget reaches the removal tangent, not ``-h``."""
        d = torch.tensor([1.0, 0.0, 0.0])
        h = torch.tensor([[[3.0, 4.0, 0.0]]])
        hook = _make_spherical_hook(d, angle_degrees=90.0)

        h_new = hook(None, None, h)

        torch.testing.assert_close(h_new @ d, torch.zeros(1, 1), atol=1e-6, rtol=0)
        torch.testing.assert_close(h_new.norm(dim=-1), h.norm(dim=-1))

    def test_profile_strength_is_fraction_of_full_removal(self):
        """A profile strength of 0.5 performs half the required geodesic."""
        layer = torch.nn.Identity()
        engine = SimpleNamespace(transformer_layers=[layer])
        config = AbliterixConfig(
            model={"model_id": "test/model"},
            steering={"steering_mode": "spherical"},
        )
        profile = SteeringProfile(
            max_weight=0.5,
            max_weight_position=0.0,
            min_weight=0.5,
            min_weight_distance=1.0,
        )
        direction = torch.tensor([1.0, 0.0, 0.0])
        vectors = torch.stack([direction, direction])
        h = torch.tensor([[[3.0, 4.0, 0.0]]])

        apply_steering(
            engine,
            steering_vectors=vectors,
            vector_index=None,
            profiles={"mlp.down_proj": profile},
            config=config,
        )
        h_new = layer(h)

        assert 0 < (h_new @ direction).item() < (h @ direction).item()

    def test_handles_tuple_output(self):
        """Hook should handle tuple outputs (as returned by some transformer layers)."""
        torch.manual_seed(42)
        d = F.normalize(torch.randn(64), p=2, dim=0)
        hook = _make_spherical_hook(d, angle_degrees=45.0)

        h = torch.randn(1, 3, 64)
        extra = torch.randn(1, 3, 64)  # e.g., attention weights
        output = (h, extra)

        result = hook(None, None, output)
        assert isinstance(result, tuple)
        assert len(result) == 2
        # The second element should be unchanged.
        torch.testing.assert_close(result[1], extra)

    def test_numerical_stability_near_parallel(self):
        """Hook should handle activations nearly parallel to direction."""
        d = F.normalize(torch.randn(64), p=2, dim=0)
        hook = _make_spherical_hook(d, angle_degrees=30.0)

        # Make h very close to d.
        h = d.unsqueeze(0).unsqueeze(0) * 2.5 + torch.randn(1, 1, 64) * 1e-6
        h_new = hook(None, None, h)

        # Should not produce NaN or Inf.
        assert not torch.isnan(h_new).any(), "NaN detected in spherical steering output"
        assert not torch.isinf(h_new).any(), "Inf detected in spherical steering output"

    def test_exactly_parallel_activation_keeps_norm_at_full_removal(self):
        """The degenerate great circle gets a stable orthogonal tangent."""
        d = F.normalize(torch.tensor([1.0, 2.0, 3.0]), p=2, dim=0)
        h = 2.5 * d.view(1, 1, -1)
        hook = _make_spherical_hook(d, angle_degrees=90.0)

        h_new = hook(None, None, h)

        torch.testing.assert_close(h_new.norm(dim=-1), h.norm(dim=-1))
        torch.testing.assert_close(h_new @ d, torch.zeros(1, 1), atol=1e-6, rtol=0)


class TestAngularHook:
    """Shared removal semantics for the angular runtime hook."""

    def test_maximum_angle_removes_direction_without_inverting_activation(self):
        d = torch.tensor([1.0, 0.0, 0.0])
        h = torch.tensor([[[3.0, 4.0, 0.0]]])
        hook = _make_angular_hook(d, angle_degrees=90.0)

        h_new = hook(None, None, h)

        torch.testing.assert_close(h_new @ d, torch.zeros(1, 1), atol=1e-6, rtol=0)
        torch.testing.assert_close(h_new.norm(dim=-1), h.norm(dim=-1))

    def test_exactly_parallel_activation_keeps_norm_at_full_removal(self):
        d = F.normalize(torch.tensor([1.0, 2.0, 3.0]), p=2, dim=0)
        h = 2.5 * d.view(1, 1, -1)
        hook = _make_angular_hook(d, angle_degrees=90.0)

        h_new = hook(None, None, h)

        torch.testing.assert_close(h_new.norm(dim=-1), h.norm(dim=-1))
        torch.testing.assert_close(h_new @ d, torch.zeros(1, 1), atol=1e-6, rtol=0)

    def test_zero_activation_remains_zero_at_maximum_angle(self):
        d = F.normalize(torch.tensor([1.0, 2.0, 3.0]), p=2, dim=0)
        h = torch.zeros(1, 1, 3)
        hook = _make_angular_hook(d, angle_degrees=90.0)

        h_new = hook(None, None, h)

        assert torch.equal(h_new, h)


class TestSharedRemovalSemantics:
    """Observable invariants shared by angular and spherical modes."""

    @staticmethod
    def _factories():
        return (
            lambda d, angle: _make_angular_hook(d, angle, adaptive=False),
            _make_spherical_hook,
        )

    def test_projection_decreases_monotonically_with_rotation_budget(self):
        d = torch.tensor([1.0, 0.0, 0.0])
        h = torch.tensor([[[3.0, 4.0, 0.0]]])

        for factory in self._factories():
            projections = [
                abs((factory(d, angle)(None, None, h) @ d).item())
                for angle in (0.0, 30.0, 60.0, 90.0)
            ]
            assert projections[0] > projections[1] > projections[2]
            assert projections[2] > projections[3] - 1e-6
            assert projections[3] < 1e-6

    def test_output_depends_on_removal_direction(self):
        h = torch.tensor([[[1.0, 2.0, 3.0]]])
        d1 = torch.tensor([1.0, 0.0, 0.0])
        d2 = torch.tensor([0.0, 1.0, 0.0])

        for factory in self._factories():
            along_d1 = factory(d1, 90.0)(None, None, h)
            along_d2 = factory(d2, 90.0)(None, None, h)
            assert not torch.allclose(along_d1, along_d2)

    def test_rotation_budget_saturates_at_removal_tangent(self):
        d = torch.tensor([1.0, 0.0, 0.0])
        h = torch.tensor([[[3.0, 4.0, 0.0]]])

        for factory in self._factories():
            at_max = factory(d, 90.0)(None, None, h)
            above_max = factory(d, 180.0)(None, None, h)
            torch.testing.assert_close(above_max, at_max)
