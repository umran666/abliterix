"""Tests for abliterix.vectors — steering vector computation.

All tests use small synthetic tensors (no GPU, no model).
"""

import torch
import torch.nn.functional as F

from abliterix.settings import AbliterixConfig
from abliterix.types import VectorMethod
from abliterix.vectors import (
    compute_configured_steering_vectors,
    compute_steering_vectors,
)


# ---------------------------------------------------------------------------
# MEAN method
# ---------------------------------------------------------------------------


def test_mean_output_shape(synthetic_states):
    benign, target = synthetic_states
    result = compute_steering_vectors(benign, target, VectorMethod.MEAN, False)
    # Input is (20, 8, 64), output should be (8, 64).
    assert result.shape == (8, 64)


def test_mean_unit_normalized(synthetic_states):
    benign, target = synthetic_states
    result = compute_steering_vectors(benign, target, VectorMethod.MEAN, False)
    norms = torch.linalg.vector_norm(result, dim=1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_mean_direction(synthetic_states):
    """MEAN should equal normalized(mean_target - mean_benign)."""
    benign, target = synthetic_states
    result = compute_steering_vectors(benign, target, VectorMethod.MEAN, False)
    expected = F.normalize(target.mean(dim=0) - benign.mean(dim=0), p=2, dim=1)
    assert torch.allclose(result, expected, atol=1e-5)


def test_multi_direction_keeps_mean_difference_as_primary_direction():
    """Rank-k extraction must retain the refusal mean shift as direction 0."""
    # Each layer has a constant mean shift plus stronger, zero-mean variation
    # along axes orthogonal to that shift.  Centring before SVD therefore
    # makes the old implementation select residual noise as direction 0.
    benign = torch.zeros(6, 2, 4)
    target = benign.clone()
    target[:, 0, 0] = 2.0
    target[:, 1, 1] = -3.0

    coefficients = torch.tensor([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
    target[:, 0, 1] += coefficients
    target[:, 1, 2] += coefficients * 2.0

    actual = compute_steering_vectors(
        benign,
        target,
        VectorMethod.MEAN,
        False,
        n_directions=3,
    )
    expected_primary = F.normalize(
        target.mean(dim=0) - benign.mean(dim=0),
        p=2,
        dim=1,
    )

    torch.testing.assert_close(actual[0], expected_primary)


def test_multi_direction_residuals_are_unit_and_mutually_orthogonal():
    benign = torch.zeros(8, 1, 5)
    target = benign.clone()
    target[:, 0, 0] = 2.0
    target[:, 0, 1] += 4.0 * torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    target[:, 0, 2] += 2.0 * torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])

    actual = compute_steering_vectors(
        benign,
        target,
        VectorMethod.MEAN,
        False,
        n_directions=3,
    )
    gram = actual[:, 0, :] @ actual[:, 0, :].T

    torch.testing.assert_close(gram, torch.eye(3), atol=1e-6, rtol=1e-6)


def test_multi_direction_does_not_spend_residual_rank_on_mean_axis():
    """Mean-axis residual variance must not hide a smaller independent axis."""
    benign = torch.zeros(4, 1, 4)
    target = benign.clone()
    mean_axis_coefficients = torch.tensor([1.0, -1.0, 0.0, 0.0])
    orthogonal_coefficients = torch.tensor([0.0, 0.0, 1.0, -1.0])
    target[:, 0, 0] = 10.0 + 8.0 * mean_axis_coefficients
    target[:, 0, 1] = orthogonal_coefficients

    actual = compute_steering_vectors(
        benign,
        target,
        VectorMethod.MEAN,
        False,
        n_directions=2,
    )

    assert torch.linalg.vector_norm(actual[1, 0]).item() == 1.0
    assert abs(actual[1, 0, 1].item()) == 1.0


def test_multi_direction_applies_winsorization_and_preserves_zero_directions():
    benign = torch.zeros(4, 1, 4)
    target = torch.tensor([100.0, 1.0, 1.0, 1.0]).repeat(4, 1, 1)

    actual = compute_steering_vectors(
        benign,
        target,
        VectorMethod.MEAN,
        False,
        n_directions=2,
        winsorize=True,
        winsorize_quantile=0.5,
    )

    torch.testing.assert_close(actual[0, 0], torch.full((4,), 0.5))
    torch.testing.assert_close(actual[1, 0], torch.zeros(4))


def test_multi_direction_reorthogonalizes_after_winsorization_and_projection():
    benign_mean = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    benign = benign_mean.repeat(8, 1, 1)
    target = benign.clone()
    target[:, 0, 0] += 2.0
    target[:, 0, 1] += 4.0 * torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    target[:, 0, 2] += 2.0 * torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])

    actual = compute_steering_vectors(
        benign,
        target,
        VectorMethod.MEAN,
        False,
        n_directions=3,
        winsorize=True,
        winsorize_quantile=0.9,
        projected_abliteration=True,
    )
    nonzero = torch.linalg.vector_norm(actual[:, 0, :], dim=1) > 1e-6
    basis = actual[nonzero, 0, :]

    assert basis.shape[0] == 3
    torch.testing.assert_close(
        basis @ basis.T,
        torch.eye(3),
        atol=1e-6,
        rtol=1e-6,
    )


def test_configured_vector_helper_uses_complete_recipe(synthetic_states):
    benign, target = synthetic_states
    config = AbliterixConfig(
        model={"model_id": "dummy/model"},
        steering={
            "vector_method": "mean",
            "orthogonal_projection": False,
            "winsorize_vectors": True,
            "winsorize_quantile": 0.9,
        },
    )

    actual = compute_configured_steering_vectors(benign, target, config)
    expected = compute_steering_vectors(
        benign,
        target,
        VectorMethod.MEAN,
        False,
        winsorize=True,
        winsorize_quantile=0.9,
    )

    torch.testing.assert_close(actual, expected)


# ---------------------------------------------------------------------------
# MEDIAN_OF_MEANS method
# ---------------------------------------------------------------------------


def test_median_of_means_output_shape(synthetic_states):
    benign, target = synthetic_states
    result = compute_steering_vectors(
        benign, target, VectorMethod.MEDIAN_OF_MEANS, False
    )
    assert result.shape == (8, 64)


def test_median_of_means_unit_normalized(synthetic_states):
    benign, target = synthetic_states
    result = compute_steering_vectors(
        benign, target, VectorMethod.MEDIAN_OF_MEANS, False
    )
    norms = torch.linalg.vector_norm(result, dim=1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_median_of_means_robust_to_outlier():
    """Median should be more robust than mean to a single outlier group."""
    torch.manual_seed(99)
    benign = torch.randn(25, 4, 32)
    target = benign + torch.randn(1, 4, 32) * 0.3

    # Inject a large outlier into the first 5 samples (one chunk).
    target[:5] += torch.randn(1, 4, 32) * 100.0

    mean_result = compute_steering_vectors(benign, target, VectorMethod.MEAN, False)
    median_result = compute_steering_vectors(
        benign, target, VectorMethod.MEDIAN_OF_MEANS, False
    )

    # Without the outlier, compute the "clean" mean direction.
    clean_target = target.clone()
    clean_target[:5] = benign[:5] + torch.randn(1, 4, 32) * 0.3
    clean = compute_steering_vectors(benign, clean_target, VectorMethod.MEAN, False)

    # Median should be closer to the clean direction than the mean is.
    cos_median = F.cosine_similarity(median_result, clean, dim=1).mean()
    cos_mean = F.cosine_similarity(mean_result, clean, dim=1).mean()
    assert cos_median > cos_mean


# ---------------------------------------------------------------------------
# PCA method
# ---------------------------------------------------------------------------


def test_pca_output_shape(synthetic_states):
    benign, target = synthetic_states
    result = compute_steering_vectors(benign, target, VectorMethod.PCA, False)
    assert result.shape == (8, 64)


def test_pca_unit_normalized(synthetic_states):
    benign, target = synthetic_states
    result = compute_steering_vectors(benign, target, VectorMethod.PCA, False)
    norms = torch.linalg.vector_norm(result, dim=1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_pca_captures_dominant_direction():
    """PCA should align with the dominant axis of variation."""
    torch.manual_seed(77)
    dominant = torch.randn(1, 1, 32)
    dominant = F.normalize(dominant, p=2, dim=2)

    benign = torch.randn(20, 1, 32) * 0.01
    # Target varies primarily along 'dominant'.
    target = benign + dominant * torch.randn(20, 1, 1) * 5.0

    result = compute_steering_vectors(benign, target, VectorMethod.PCA, False)
    # The result should be roughly aligned with dominant (or anti-aligned).
    cos = F.cosine_similarity(result, dominant.squeeze(0), dim=1).abs()
    assert cos.item() > 0.8


# ---------------------------------------------------------------------------
# Orthogonal projection
# ---------------------------------------------------------------------------


def test_orthogonal_projection_removes_benign(synthetic_states):
    """After orthogonal projection, dot with benign direction should be ~0."""
    benign, target = synthetic_states
    result = compute_steering_vectors(benign, target, VectorMethod.MEAN, True)
    benign_dir = F.normalize(benign.mean(dim=0), p=2, dim=1)
    dot = torch.sum(result * benign_dir, dim=1)
    assert torch.allclose(dot, torch.zeros(8), atol=1e-5)


def test_orthogonal_projection_still_normalized(synthetic_states):
    benign, target = synthetic_states
    result = compute_steering_vectors(benign, target, VectorMethod.MEAN, True)
    norms = torch.linalg.vector_norm(result, dim=1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_identical_inputs_zero_vector():
    """When benign == target, mean diff is zero; F.normalize returns zeros."""
    states = torch.randn(10, 4, 32)
    result = compute_steering_vectors(states, states.clone(), VectorMethod.MEAN, False)
    assert torch.allclose(result, torch.zeros(4, 32), atol=1e-6)
