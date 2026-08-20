from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from anza_fs.foliation_conv import ANZAFoliationConv, FreeFoliationConv, SharedOrientationTransform
from anza_fs.geometry import axial_bank, frozen_foliation_geometry, reciprocal_scales
from anza_fs.hard_bench_v6 import (
    CASES,
    NEGATIVE_EVENTS_CALIBRATION_PLUS_DEVELOPMENT,
    POSITIVE_EVENTS_CALIBRATION_PLUS_DEVELOPMENT,
    SPLIT_BASE,
    SPLIT_SIZE,
    generate_hard_sample,
    hard_bench_config,
)
from anza_fs.kernels import kernel_centroids
from anza_fs.metrics import false_bridge_event, select_recall95_threshold
from anza_fs.model import build_h3_model
from anza_fs.protocol import protocol_payload


def test_axial_invariance():
    angles, _, _ = axial_bank(8)
    assert torch.allclose(torch.stack((torch.cos(2 * angles), torch.sin(2 * angles))), torch.stack((torch.cos(2 * (angles + math.pi)), torch.sin(2 * (angles + math.pi)))), atol=1e-6)


def test_stable_and_unstable_axes_are_orthonormal():
    _, unstable, stable = axial_bank(8)
    assert torch.allclose((unstable * stable).sum(-1), torch.zeros(8), atol=1e-7)
    assert torch.allclose(unstable.norm(dim=-1), torch.ones(8), atol=1e-7)


def test_reciprocal_scales():
    sigma_u, sigma_s = reciprocal_scales()
    assert sigma_u * sigma_s == pytest.approx(1.5**2)


def test_five_lobe_centers_follow_axes():
    layer = ANZAFoliationConv(1)
    centers = kernel_centroids(layer.kernels())
    assert centers[0, 0].norm() < 1e-6
    assert centers[0, 1, 0] > 2.0 and abs(float(centers[0, 1, 1])) < 1e-6
    assert centers[0, 2, 0] < -2.0
    assert centers[0, 3, 1] > 0.8 and centers[0, 4, 1] < -0.8


def test_unstable_response_samples_longitudinally():
    layer = ANZAFoliationConv(1)
    value = torch.zeros(1, 1, 65, 65)
    value[:, :, 32, 29] = 1.0
    value[:, :, 32, 35] = 1.0
    _center, longitudinal, _transverse = layer.structural_responses(value)
    assert longitudinal[0, 0, 0, 32, 32] > longitudinal[0, 4, 0, 32, 32]


def test_stable_response_samples_transversely():
    layer = ANZAFoliationConv(1)
    value = torch.zeros(1, 1, 65, 65)
    value[:, :, 30, :] = 1.0
    value[:, :, 34, :] = 1.0
    _center, _longitudinal, center_minus_stable = layer.structural_responses(value)
    assert center_minus_stable[0, 0, 0, 32, 32] < 0


def test_parallel_background_fixture_has_negative_center_minus_stable():
    layer = ANZAFoliationConv(1)
    value = torch.zeros(1, 1, 65, 65)
    value[:, :, 30, :] = 1.0
    value[:, :, 34, :] = 1.0
    _, _, response = layer.structural_responses(value)
    assert float(response[0, 0, 0, 32, 32]) < 0.0


def test_true_line_fixture_has_positive_center_minus_stable():
    layer = ANZAFoliationConv(1)
    value = torch.zeros(1, 1, 65, 65)
    value[:, :, 32, :] = 1.0
    _, _, response = layer.structural_responses(value)
    assert float(response[0, 0, 0, 32, 32]) > 0.0


def test_free_foliation_reproduces_anza_geometry_at_initialization():
    assert torch.allclose(FreeFoliationConv(2).kernels(), ANZAFoliationConv(2).kernels(), atol=1e-6)


def test_shared_psi_reuses_one_transform_for_every_orientation():
    transform = SharedOrientationTransform(4).eval()
    one = torch.randn(2, 1, 12, 8, 8)
    result = transform(one.expand(-1, 8, -1, -1, -1))
    assert torch.allclose(result[:, 0], result[:, 7], atol=1e-7)


def test_fuzzy_aggregation_is_finite():
    layer = ANZAFoliationConv(3).eval()
    with torch.no_grad():
        output, aux = layer(torch.randn(2, 3, 24, 24), return_aux=True)
    assert torch.isfinite(output).all() and torch.isfinite(aux["evidence_entropy"])


def test_gamma_zero_initialization_is_exact_identity():
    layer = ANZAFoliationConv(3).eval()
    value = torch.randn(2, 3, 24, 24)
    output, _ = layer(value)
    assert torch.equal(output, value)


def test_h3_confirm_is_access_locked():
    with pytest.raises(PermissionError):
        generate_hard_sample("confirm", 0)
    assert hard_bench_config()["confirm_status"] == "LOCKED_UNOPENED"


def test_v6_splits_have_no_seed_leakage_and_enough_events():
    ranges = {name: set(range(base, base + SPLIT_SIZE[name])) for name, base in SPLIT_BASE.items()}
    assert all(not (ranges[first] & ranges[second]) for index, first in enumerate(ranges) for second in list(ranges)[index + 1 :])
    assert NEGATIVE_EVENTS_CALIBRATION_PLUS_DEVELOPMENT >= 1000
    assert POSITIVE_EVENTS_CALIBRATION_PLUS_DEVELOPMENT >= 1000


def test_false_bridge_event_counting_oracle():
    sample = generate_hard_sample("train", CASES.index("parallel_gap_confuser"))
    prediction = np.asarray(sample["visible_fault_mask"], dtype=bool).copy()
    assert not false_bridge_event(prediction, sample)
    first = np.argwhere(sample["negative_anchor_masks"][0]).mean(axis=0)
    second = np.argwhere(sample["negative_anchor_masks"][1]).mean(axis=0)
    for fraction in np.linspace(0.0, 1.0, 64):
        y, x = np.rint((1 - fraction) * first + fraction * second).astype(int)
        prediction[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = True
    assert false_bridge_event(prediction, sample)


def test_matched_recall_threshold_selection_uses_highest_eligible_threshold():
    curve = [
        {"threshold": 0.2, "branch_recall": 0.99},
        {"threshold": 0.4, "branch_recall": 0.96},
        {"threshold": 0.6, "branch_recall": 0.94},
    ]
    selected = select_recall95_threshold(curve)
    assert selected["threshold"] == 0.4 and selected["recall95_achieved"] is True


def test_protocol_freezes_losses_geometry_and_downstream_locks():
    protocol = protocol_payload()
    assert protocol["architecture"] == {
        "M": 8, "support": 9, "base_scale": 1.5, "lambda": 0.35,
        "delta_u": "1.5*sigma_u", "delta_s": "1.5*sigma_s",
        "placements": ["decoder_1_4", "decoder_1_2"], "gamma_init": 0.0,
        "feature_tuple": ["C", "U-C", "C-S"],
    }
    assert protocol["training"]["epochs"] == 15
    assert not any(protocol[key] for key in ("confirm_opened", "cracks_accessed", "expert_accessed", "H4_opened", "lambda_tuned", "M_tuned"))


@pytest.mark.parametrize("variant", ["F0_backbone", "F1_old_generic", "F2_free_foliation", "F3_anza_fs"])
def test_h3_model_forward_is_finite(variant):
    model = build_h3_model(variant).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32), return_aux=True)
    assert output["visible_logits"].shape == (1, 1, 32, 32)
    assert torch.isfinite(output["visible_logits"]).all()
