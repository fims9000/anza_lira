import math

import numpy as np
import pytest

from structural_reachability.geometry import (
    compute_axial_consistency,
    compute_directed_anisotropic_factor,
    compute_fuzzy_compatibility,
    compute_scale_compatibility,
    log_geometric_mean,
    symmetrize_affinity,
)


def test_axial_consistency_is_pi_invariant_and_bounded() -> None:
    first, second = 0.21, 1.03
    value = compute_axial_consistency(first, second)
    assert value == pytest.approx(compute_axial_consistency(first + math.pi, second))
    assert value == pytest.approx(compute_axial_consistency(first, second + math.pi))
    assert 0.0 <= value <= 1.0
    assert compute_axial_consistency(first, first) == pytest.approx(1.0)


def test_fuzzy_and_scale_compatibility_have_hand_checked_limits() -> None:
    assert compute_fuzzy_compatibility(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert compute_fuzzy_compatibility(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert compute_scale_compatibility(2.0, 2.0, 0.5, 0.5) == pytest.approx(1.0)
    assert compute_scale_compatibility(2.0, 4.0, 0.5, 1.0) == pytest.approx(0.25)


def test_anisotropic_factor_prefers_longitudinal_displacement() -> None:
    along = compute_directed_anisotropic_factor(0.0, 2.0, 0.25, 0.0, 1.0)
    across = compute_directed_anisotropic_factor(0.0, 2.0, 0.25, 1.0, 0.0)
    assert 0.0 <= across < along <= 1.0
    assert along == pytest.approx(compute_directed_anisotropic_factor(math.pi, 2.0, 0.25, 0.0, 1.0))


def test_symmetrization_and_component_fusion_are_bounded() -> None:
    assert symmetrize_affinity(0.25, 1.0) == pytest.approx(0.5)
    assert symmetrize_affinity(0.25, 1.0, method="minimum") == pytest.approx(0.25)
    assert symmetrize_affinity(0.25, 1.0, method="average") == pytest.approx(0.625)
    assert log_geometric_mean(np.asarray([[0.25, 1.0]]))[0] == pytest.approx(0.5)
