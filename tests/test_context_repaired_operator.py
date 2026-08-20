from __future__ import annotations

import math

import torch

from models.azconv_context_repaired import (
    ContextGatedResidualANZA,
    context_head_macs_per_pixel,
    context_head_parameter_count,
    doubled_angle_vector,
)
from models.segmentation_context_repaired import build_context_repaired_model
from models.segmentation_v2 import build_comparable_model


def test_zero_residual_is_exactly_seed_matched_v1() -> None:
    torch.manual_seed(31)
    v1 = build_comparable_model("anza_v1", widths=(8, 12, 16, 20)).eval()
    model = build_context_repaired_model(widths=(8, 12, 16, 20), seed_matched_v1=v1).eval()
    image = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        assert torch.equal(model(image), v1(image))


def test_context_head_reaches_two_pixels_while_transport_stays_three() -> None:
    layer = ContextGatedResidualANZA(3, 4, num_rules=4)
    assert layer.routing_kernel_size == 3
    with torch.no_grad():
        layer.context_dw1.weight.fill_(1.0)
        layer.context_dw2.weight.fill_(1.0)
        layer.context_projection.weight.fill_(1.0)
    baseline = torch.zeros(1, 4, 9, 9)
    perturbed = baseline.clone()
    perturbed[:, :, 4, 6] = 1.0
    center_baseline = layer.context_features(baseline)[0, :, 4, 4]
    center_perturbed = layer.context_features(perturbed)[0, :, 4, 4]
    assert torch.all(center_perturbed > center_baseline)


def test_depthwise_context_initialization_preserves_every_channel() -> None:
    layer = ContextGatedResidualANZA(3, 4, num_rules=4)
    impulse = torch.zeros(1, 4, 7, 7)
    impulse[0, :, 3, 3] = torch.arange(1.0, 5.0)
    context = layer.context_features(impulse)
    assert torch.all(context[0, :, 3, 3] > 0.0)


def test_doubled_angle_representation_is_axial_and_normalized() -> None:
    layer = ContextGatedResidualANZA(3, 8, num_rules=4)
    features = torch.randn(2, 8, 12, 12)
    geometry = layer.geometry(features)
    norm = geometry["orientation_cos2"].square() + geometry["orientation_sin2"].square()
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)
    theta = geometry["theta"]
    assert torch.all(theta >= 0.0)
    assert torch.all(theta < math.pi)


def test_doubled_angle_vector_is_exactly_pi_periodic() -> None:
    theta = torch.linspace(-2.0 * math.pi, 2.0 * math.pi, 41, dtype=torch.float64)
    assert torch.allclose(doubled_angle_vector(theta), doubled_angle_vector(theta + math.pi), atol=1e-12)


def test_direct_gate_can_change_when_diagnostic_ambiguity_is_fixed() -> None:
    layer = ContextGatedResidualANZA(3, 4, num_rules=4)
    with torch.no_grad():
        layer.direct_gate_head.weight.fill_(1.0)
        layer.direct_gate_head.bias.zero_()
    first = layer.geometry(torch.zeros(1, 4, 8, 8))
    second = layer.geometry(torch.ones(1, 4, 8, 8))
    assert torch.allclose(first["ambiguity"], second["ambiguity"])
    assert not torch.allclose(first["ambiguity_gate"], second["ambiguity_gate"])


def test_context_branch_has_finite_wakeup_gradients_and_auditable_cost() -> None:
    layer = ContextGatedResidualANZA(3, 8, num_rules=4)
    with torch.no_grad():
        layer.residual_lambda_raw.fill_(0.1)
    output = layer(torch.randn(2, 3, 16, 16)).square().mean()
    output.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in layer.parameters())
    assert context_head_parameter_count(layer) > 0
    assert context_head_macs_per_pixel(layer) > context_head_parameter_count(layer) // 10
