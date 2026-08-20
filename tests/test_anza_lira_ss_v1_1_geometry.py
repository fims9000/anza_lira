"""Numerical contracts required before V1.1 B0-B3 training."""

from __future__ import annotations

import math

import numpy as np
import torch

from datasets.cracks import BLUE, WHITE
from structural_stability_v1_1.geometry_metric import GeometrySidecar, metric_from_axial, parameter_audit
from structural_stability_v1_1.geometry_targets import geometry_target
from structural_stability_v1_1.matrix_log import spd_matrix_exp, spd_matrix_log
from structural_stability_v1_1.metric_transport import (
    area_normalize_jacobian,
    forward_jacobian_xy,
    metric_equivariance_loss,
    output_to_input_jacobian_xy,
    transport_metric,
)


def _metric(theta: float, d: float, m: float = 0.0) -> torch.Tensor:
    c2 = torch.full((1, 1, 1), math.cos(2.0 * theta))
    s2 = torch.full((1, 1, 1), math.sin(2.0 * theta))
    return metric_from_axial(c2, s2, torch.full_like(c2, d), torch.full_like(c2, m))[0, :, :, 0, 0]


def test_b3_is_spd_and_determinant_one() -> None:
    head = GeometrySidecar(4, "B3").eval()
    output = head(torch.randn(2, 4, 9, 11))
    matrices = output.metric.permute(0, 3, 4, 1, 2)
    eigenvalues = torch.linalg.eigvalsh(matrices)
    assert torch.all(eigenvalues > 0)
    torch.testing.assert_close(torch.linalg.det(matrices), torch.ones_like(eigenvalues[..., 0]), atol=1e-5, rtol=1e-5)


def test_b2_determinant_is_free_and_matches_exp_4m() -> None:
    c2 = torch.ones(1, 1, 2)
    s2 = torch.zeros_like(c2)
    d = torch.full_like(c2, 0.2)
    m = torch.tensor([[[-0.3, 0.3]]])
    metric = metric_from_axial(c2, s2, d, m).permute(0, 3, 4, 1, 2)
    determinant = torch.linalg.det(metric)
    torch.testing.assert_close(determinant, torch.exp(4.0 * m), atol=1e-6, rtol=1e-6)
    assert determinant[0, 0, 0] != determinant[0, 0, 1]


def test_identity_and_axial_reversal_invariance() -> None:
    torch.testing.assert_close(_metric(0.0, 0.0, 0.0), torch.eye(2), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(_metric(0.37, 0.2), _metric(0.37 + math.pi, 0.2), atol=1e-6, rtol=1e-6)


def test_known_rotation_scaling_and_shear_transport() -> None:
    clean = _metric(0.0, 0.3)
    angle = 0.41
    rotation = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    rotated = transport_metric(clean, rotation)
    torch.testing.assert_close(rotated, rotation @ clean @ rotation.T, atol=1e-6, rtol=1e-6)
    scaling = torch.diag(torch.tensor([2.0, 0.5]))
    scaled = transport_metric(clean, scaling)
    torch.testing.assert_close(torch.linalg.det(scaled), torch.linalg.det(clean), atol=1e-5, rtol=1e-5)
    shear = torch.tensor([[1.0, 0.35], [0.0, 1.0]])
    sheared = transport_metric(clean, shear)
    torch.testing.assert_close(sheared, shear @ clean @ shear.T, atol=1e-6, rtol=1e-6)


def test_output_to_input_jacobian_is_inverted_for_forward_transport() -> None:
    height, width = 12, 15
    yy, _xx = np.mgrid[:height, :width]
    shear = 0.2
    displacement = np.stack((np.zeros_like(yy, dtype=np.float64), shear * yy), axis=0)
    backward = output_to_input_jacobian_xy(displacement)
    forward = forward_jacobian_xy(displacement)
    expected_backward = np.asarray([[1.0, shear], [0.0, 1.0]])
    expected_forward = np.asarray([[1.0, -shear], [0.0, 1.0]])
    np.testing.assert_allclose(backward, np.broadcast_to(expected_backward, backward.shape), atol=1e-12)
    np.testing.assert_allclose(forward, np.broadcast_to(expected_forward, forward.shape), atol=1e-12)


def test_area_normalization_and_photometric_identity() -> None:
    jacobian = torch.tensor([[1.7, 0.2], [0.1, 0.9]])
    normalized = area_normalize_jacobian(jacobian)
    torch.testing.assert_close(torch.linalg.det(normalized), torch.tensor(1.0), atol=1e-6, rtol=1e-6)
    clean = _metric(0.2, 0.25, 0.1)
    torch.testing.assert_close(transport_metric(clean, torch.eye(2)), clean, atol=1e-6, rtol=1e-6)


def test_spd_log_reconstructs_and_has_finite_gradient() -> None:
    strength = torch.tensor([[[0.23]]], requires_grad=True)
    metric = metric_from_axial(
        torch.tensor([[[0.8]]]), torch.tensor([[[0.6]]]),
        strength, torch.tensor([[[0.07]]]),
    )
    logged = spd_matrix_log(metric)
    reconstructed = spd_matrix_exp(logged)
    torch.testing.assert_close(reconstructed, metric, atol=1e-5, rtol=1e-5)
    loss = metric_equivariance_loss(metric, metric.detach() * 1.01, torch.ones(1, 1, 1))
    loss.backward()
    assert strength.grad is not None and torch.isfinite(strength.grad).all()


def test_train_geometry_target_follows_horizontal_ridge_and_is_bounded() -> None:
    mask = np.full((65, 81, 3), WHITE, dtype=np.uint8)
    mask[31:34, 8:73] = np.asarray(BLUE, dtype=np.uint8)
    target = geometry_target([mask])
    selected = target["supervision"] & (np.indices(mask.shape[:2])[1] > 15) & (np.indices(mask.shape[:2])[1] < 65)
    assert selected.any()
    assert float(np.median(target["target_c2"][selected])) > 0.9
    assert float(np.median(np.abs(target["target_s2"][selected]))) < 0.1
    assert 0.0 <= float(target["target_d"].min()) <= float(target["target_d"].max()) <= 0.35 + 1e-6
    assert np.all(target["geometry_weight"][~target["supervision"]] == 0)


def test_b2_b3_parameter_difference_is_below_one_percent() -> None:
    audit = parameter_audit()
    assert audit["passes_one_percent"] is True
    assert audit["B2_B3_relative_difference"] < 0.01
