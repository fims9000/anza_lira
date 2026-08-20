from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from anza_hs.model import build_h1_model
from anza_hs.operators import ANZAHyperbolicConv, GenericAnisoConv, IsotropicOrientConv
from anza_hs.orientation_bank import axial_angles, orientation_bank_targets
from anza_hs.protocol import protocol_payload
from anza_hs.shadowing import bank_axes, inverse_map_points, symmetric_shadow_distance, top_axial_peaks
from anza_hs.stress_bench import CASES, SPLIT_BASE, SPLIT_SIZE, generate_stress_sample, stressbench_config


def test_axial_bank_has_pi_periodicity():
    angles = axial_angles(8)
    first = np.stack((np.cos(2 * angles), np.sin(2 * angles)), axis=1)
    shifted = np.stack((np.cos(2 * (angles + np.pi)), np.sin(2 * (angles + np.pi))), axis=1)
    assert np.allclose(first, shifted, atol=1e-6)


def test_x_target_has_two_separated_peaks():
    index = CASES.index("x_crossing")
    sample = generate_stress_sample("train", index)
    target, _valid = orientation_bank_targets(sample)
    y, x = np.argwhere(sample["junction_map"])[len(np.argwhere(sample["junction_map"])) // 2]
    peaks = top_axial_peaks(torch.from_numpy(target[:, y, x]))
    assert len(peaks) == 2
    assert target[int(peaks[0]), y, x] > 0.5 and target[int(peaks[1]), y, x] > 0.5


def test_fixed_bank_has_no_arbitrary_mode_permutation():
    assert np.allclose(axial_angles(8), np.arange(8) * np.pi / 8)


def test_lambda_zero_hyperbolic_kernel_is_isotropic():
    hyperbolic = ANZAHyperbolicConv(2, hyperbolicity=0.0)
    isotropic = IsotropicOrientConv(2)
    assert torch.allclose(hyperbolic.kernels(), isotropic.kernels(), atol=1e-7)


def test_reciprocal_scale_constraint():
    layer = ANZAHyperbolicConv(2)
    sigma_u, sigma_s = layer.scales()
    assert torch.allclose(sigma_u * sigma_s, torch.full_like(sigma_u, 1.5**2), atol=1e-6)


@pytest.mark.parametrize("layer", [IsotropicOrientConv(2), GenericAnisoConv(2), ANZAHyperbolicConv(2)])
def test_every_kernel_is_normalized(layer):
    assert torch.allclose(layer.kernels().sum(dim=(-2, -1)), torch.ones(8), atol=1e-6)


def test_hyperbolic_weight_is_larger_along_axis_than_transverse():
    kernel = ANZAHyperbolicConv(1).kernels()[0]
    center = kernel.shape[0] // 2
    assert kernel[center, center + 3] > kernel[center + 3, center]


@pytest.mark.parametrize("variant", ["B1_isotropic", "B2_generic_aniso", "B3_anza_hyperbolic"])
def test_zero_gamma_makes_bank_residual_identity(variant):
    model = build_h1_model(variant).eval(); image = torch.randn(1, 3, 32, 32)
    block = model.bank_quarter; value = torch.randn(1, block.channels, 8, 8)
    output, _ = block(value)
    assert torch.equal(output, value)


def test_generic_kernel_initially_reproduces_hyperbolic_kernel():
    generic = GenericAnisoConv(1); hyperbolic = ANZAHyperbolicConv(1)
    assert torch.allclose(generic.kernels(), hyperbolic.kernels(), atol=1e-6)


def test_shadow_distance_zero_for_identical_singleton_trajectory():
    xy = torch.tensor([[2.0, 3.0]]); axis = torch.tensor([[1.0, 0.0]])
    assert symmetric_shadow_distance(xy, axis, xy, axis).item() == pytest.approx(0.0, abs=1e-7)


def test_inverse_transform_restores_translation():
    moved = torch.tensor([[3.0, 4.0]]); inverse = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(inverse_map_points(moved, inverse), torch.tensor([[2.0, 2.0]]))


def test_small_translation_has_small_finite_shadow_distance():
    first = torch.tensor([[0.0, 0.0], [1.0, 0.0]]); second = first + torch.tensor([0.2, 0.0]); axis = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    value = symmetric_shadow_distance(first, axis, second, axis)
    assert torch.isfinite(value) and value < 0.2


def test_orthogonal_trajectory_has_higher_shadow_cost():
    xy = torch.tensor([[0.0, 0.0]]); horizontal = torch.tensor([[1.0, 0.0]]); vertical = torch.tensor([[0.0, 1.0]])
    assert symmetric_shadow_distance(xy, horizontal, xy, vertical) > symmetric_shadow_distance(xy, horizontal, xy, horizontal)


def test_multipeak_rollout_selects_two_separate_axial_branches():
    evidence = torch.tensor([0.9, 0.1, 0.05, 0.1, 0.8, 0.1, 0.05, 0.1])
    peaks = top_axial_peaks(evidence)
    axes = bank_axes(peaks, 8)
    assert peaks.tolist() == [0, 4] and axes.shape == (2, 2)


def test_stressbench_is_balanced_and_split_seeds_disjoint():
    assert SPLIT_SIZE["train"] % len(CASES) == 0 and SPLIT_SIZE["dev"] % len(CASES) == 0
    ranges = [set(range(base, base + SPLIT_SIZE[name])) for name, base in SPLIT_BASE.items()]
    assert not (ranges[0] & ranges[1] or ranges[0] & ranges[2] or ranges[1] & ranges[2])


def test_stressbench_generation_is_deterministic():
    first = generate_stress_sample("train", 0); second = generate_stress_sample("train", 0)
    assert np.array_equal(first["image"], second["image"]) and np.array_equal(first["visible_fault_mask"], second["visible_fault_mask"])


def test_stressbench_confirm_is_locked():
    with pytest.raises(PermissionError):
        generate_stress_sample("confirm", 0)


def test_protocol_locks_all_downstream_data_and_tuning():
    protocol = protocol_payload()
    assert protocol["confirm_opened"] is False and protocol["cracks_accessed"] is False and protocol["expert_accessed"] is False
    assert protocol["continuation_trained"] is False and protocol["lambda_tuned"] is False and protocol["M_tuned"] is False
    assert stressbench_config()["confirm_status"] == "LOCKED_UNOPENED"


@pytest.mark.parametrize("variant", ["B0_backbone", "B1_isotropic", "B2_generic_aniso", "B3_anza_hyperbolic"])
def test_h1_model_forward_is_finite(variant):
    model = build_h1_model(variant).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32), return_aux=True)
    assert output["visible_logits"].shape == (1, 1, 32, 32) and torch.isfinite(output["visible_logits"]).all()
