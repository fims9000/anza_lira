"""Focused contracts for CRACKS Structural Stability V1 SS0-SS1."""

from __future__ import annotations

import numpy as np

from cracks_experiment.partial_labels import map_partial_annotation
from datasets.cracks import BLUE, GREEN, ORANGE, WHITE
from structural_stability_v1.agreement import crowd_agreement
from structural_stability_v1.data import build_split
from structural_stability_v1.perturb import apply_perturbation, transform_rgb_mask, warp_jacobian
from structural_stability_v1.perturb.seeds import perturbation_seed
from structural_stability_v1.protocol import TRAIN_SEVERITIES


def _rgb_row() -> np.ndarray:
    return np.asarray([[BLUE, GREEN, ORANGE, WHITE]], dtype=np.uint8)


def test_partial_semantics_preserve_unknown_white() -> None:
    target, weight = map_partial_annotation(_rgb_row())
    np.testing.assert_array_equal(target, [[1.0, 1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(weight, [[1.0, 0.5, 1.0, 0.0]])


def test_agreement_matches_frozen_formula_and_unlabeled_zero() -> None:
    masks = [
        np.asarray([[BLUE, WHITE]], dtype=np.uint8),
        np.asarray([[GREEN, WHITE]], dtype=np.uint8),
        np.asarray([[ORANGE, WHITE]], dtype=np.uint8),
    ]
    result = crowd_agreement(masks)
    # W+ = 1.5, W- = 1.0, so |2*(1.5/2.5)-1|^2 * (2.5/3).
    expected = abs(2.0 * (1.5 / 2.5) - 1.0) ** 2 * (2.5 / 3.0)
    np.testing.assert_allclose(result["agreement"][0, 0], expected, atol=1e-7)
    assert result["agreement"][0, 1] == 0.0
    assert result["labeled_weight"][0, 1] == 0.0


def test_rank_split_is_complete_disjoint_and_uses_rank_not_integer_id() -> None:
    section_ids = [value for value in range(1, 401) if value not in {9, 185, 249, 336}]
    split = build_split(section_ids)
    assert len(split["SS_TRAIN"]) == 220
    assert len(split["SS_CALIBRATION"]) == 40
    assert len(split["SS_DEVELOPMENT"]) == 50
    assert len(split["SS_CONFIRM"]) == 56
    flattened = [value for values in split.values() for value in values]
    assert len(flattened) == len(set(flattened)) == len(section_ids)
    assert sorted(flattened) == section_ids
    assert split["SS_CALIBRATION"][0] != 231


def test_seed_and_all_nonwarp_perturbations_are_deterministic() -> None:
    image = np.linspace(-1.0, 1.0, 3 * 32 * 40, dtype=np.float32).reshape(3, 32, 40)
    assert perturbation_seed(7, "crop", "noise", 2, 0) == perturbation_seed(7, "crop", "noise", 2, 0)
    assert perturbation_seed(7, "crop", "noise", 2, 0) != perturbation_seed(7, "crop", "noise", 2, 1)
    for family in ("gain", "noise", "bandlimit", "phase"):
        first = apply_perturbation(image, 7, "crop", family, 2)
        second = apply_perturbation(image, 7, "crop", family, 2)
        assert np.array_equal(first.image, second.image)
        assert np.isfinite(first.image).all()


def test_bandlimit_and_phase_operate_only_on_depth_axis() -> None:
    depth_profile = np.sin(np.arange(32, dtype=np.float32) / 3.0)
    image = np.broadcast_to(depth_profile[None, :, None], (3, 32, 40)).copy()
    for family in ("bandlimit", "phase"):
        result = apply_perturbation(image, 11, "axis", family, 2).image
        # Every lateral column remains identical; only the depth signal changes.
        np.testing.assert_allclose(result[:, :, 0], result[:, :, -1], atol=1e-6)


def test_warp_is_bounded_deterministic_and_preserves_palette() -> None:
    image = np.linspace(-1.0, 1.0, 3 * 48 * 64, dtype=np.float32).reshape(3, 48, 64)
    first = apply_perturbation(image, 13, "warp", "warp", 3)
    second = apply_perturbation(image, 13, "warp", "warp", 3)
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.displacement_yx, second.displacement_yx)
    displacement = first.displacement_yx
    assert displacement is not None
    magnitude = np.sqrt(np.sum(displacement * displacement, axis=0))
    assert float(magnitude.max()) <= 2.0 + 1e-5
    determinant, condition = warp_jacobian(displacement)
    assert float(determinant.min()) >= 0.75
    assert float(determinant.max()) <= 1.25
    assert float(condition.max()) <= 1.5
    palette = np.tile(_rgb_row(), (48, 16, 1))[:, :64]
    transformed = transform_rgb_mask(palette, first)
    allowed = {BLUE, GREEN, ORANGE, WHITE}
    assert set(map(tuple, np.unique(transformed.reshape(-1, 3), axis=0))) <= allowed


def test_severity_three_is_evaluation_only() -> None:
    assert TRAIN_SEVERITIES == (1, 2)
    assert 3 not in TRAIN_SEVERITIES
