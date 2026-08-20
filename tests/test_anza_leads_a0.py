from __future__ import annotations

import math

import numpy as np
import torch

from anza_hs.operators import ANZAHyperbolicConv, GenericAnisoConv
from anza_leads.model import LEADS_VARIANTS, WIDTHS, build_leads_model
from anza_leads.orientation import crowd_orientation_loss, crowd_orientation_targets
from anza_leads.protocol import PROTOCOL, build_split_manifest, expected_fixed_scales


def test_exact_l2_l3_initial_geometry_and_learnability() -> None:
    generic = GenericAnisoConv(2)
    anza = ANZAHyperbolicConv(2)
    assert torch.allclose(generic.kernels(), anza.kernels(), atol=1e-7)
    assert generic.raw_sigma_u.requires_grad and generic.raw_sigma_s.requires_grad
    assert anza.raw_sigma_u is None and anza.raw_sigma_s is None
    expected_u, expected_s = expected_fixed_scales()
    sigma_u, sigma_s = anza.scales()
    assert np.allclose(sigma_u.numpy(), expected_u)
    assert np.allclose(sigma_s.numpy(), expected_s)


def test_gamma_zero_is_exact_identity_and_evidence_is_independent() -> None:
    layer = ANZAHyperbolicConv(2)
    x = torch.randn(2, 2, 17, 19)
    output, logits = layer(x)
    assert torch.equal(output, x)
    with torch.no_grad():
        layer.evidence_head.weight.zero_()
        layer.evidence_head.bias.zero_()
    _output, logits = layer(x)
    assert torch.allclose(torch.sigmoid(logits).sum(dim=1), torch.full_like(logits[:, 0], 4.0))


def test_all_variants_share_backbone_and_equal_orientation_auxiliary() -> None:
    for variant in LEADS_VARIANTS:
        model = build_leads_model(variant)
        assert model.widths == WIDTHS
        result = model(torch.randn(1, 3, 32, 32), return_aux=True)
        assert len(result["orientation_logits"]) == 2
        assert [value.shape[1] for value in result["orientation_logits"]] == [8, 8]


def test_orientation_targets_use_only_explicit_positive_pixels() -> None:
    target = torch.zeros(1, 1, 15, 15)
    weight = torch.zeros_like(target)
    target[:, :, 7, 4:11] = 1.0
    weight[:, :, 7, 4:8] = 1.0
    weight[:, :, 7, 8:11] = 0.5
    weight[:, :, 3, 3] = 1.0  # orange: target remains zero
    bank, valid = crowd_orientation_targets(target, weight)
    assert valid[0, 0, 7, 6] == 1.0
    assert valid[0, 0, 7, 9] == 0.5
    assert valid[0, 0, 3, 3] == 0.0
    assert valid[0, 0, 0, 0] == 0.0
    assert bank[0, 0, 7, 6] > 0.99


def test_orientation_auxiliary_has_finite_gradients() -> None:
    logits = [torch.randn(1, 8, 8, 8, requires_grad=True), torch.randn(1, 8, 16, 16, requires_grad=True)]
    target = torch.zeros(2, 8, 32, 32)
    confidence = torch.zeros(2, 1, 32, 32)
    target[:, 0, 10:20, 10:20] = 1.0
    confidence[:, :, 10:20, 10:20] = torch.tensor([1.0, 0.5]).view(2, 1, 1, 1)
    loss = crowd_orientation_loss(logits, target, confidence)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in logits)


def test_section_protocol_is_disjoint_and_expert_locked() -> None:
    split = build_split_manifest()
    keys = ("training_pool", "train_calibration_buffer", "calibration", "calibration_development_buffer", "development")
    groups = [set(split[key]) for key in keys]
    assert not any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i + 1, len(groups)))
    assert len(split["calibration"]) == 32
    assert len(split["development"]) >= 70
    assert split["training_evaluation_annotators_disjoint"]
    assert PROTOCOL["data"]["expert"] == "LOCKED_NOT_ACCESSED"


def test_axial_bank_has_pi_periodicity() -> None:
    layer = ANZAHyperbolicConv(1)
    kernel = layer.kernels()[3]
    assert torch.allclose(kernel, torch.flip(kernel, dims=(-2, -1)), atol=1e-7)
