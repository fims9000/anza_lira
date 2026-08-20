from __future__ import annotations

import numpy as np
import pytest

from anza_ek.e0_audit import run_e0
from anza_ek.kernels import METHODS, generated_kernel_bank, seed_probe
from anza_ek.torus import CAT_INVERSE, CAT_MAP, cat_eigensystem, exact_discrete_permutation, koopman_transport, torus_grid, torus_map


def test_cat_map_is_integer_area_preserving_hyperbolic():
    eigenvalues, _, _, _ = cat_eigensystem()
    absolute = np.sort(np.abs(eigenvalues))
    assert round(np.linalg.det(CAT_MAP)) == 1
    assert absolute[0] < 1 < absolute[1]
    assert np.prod(absolute) == pytest.approx(1.0)


def test_cat_inverse_is_exact_and_origin_fixed():
    assert np.array_equal(CAT_MAP @ CAT_INVERSE, np.eye(2, dtype=np.int64))
    assert np.array_equal(torus_map(np.zeros((1, 2)), CAT_MAP), np.zeros((1, 2)))


def test_forward_then_inverse_restores_continuous_torus_points():
    rng = np.random.default_rng(41)
    points = rng.uniform(-0.5, 0.5, size=(1000, 2))
    restored = torus_map(torus_map(points, CAT_MAP), CAT_MAP, power=-1)
    assert np.allclose(restored, points, atol=1e-12)


def test_exact_discrete_cat_is_permutation_preserving_sum_and_energy():
    rng = np.random.default_rng(41)
    field = rng.normal(size=(64, 64))
    moved = exact_discrete_permutation(field, CAT_MAP)
    restored = exact_discrete_permutation(moved, CAT_MAP, power=-1)
    assert np.array_equal(restored, field)
    assert moved.sum() == pytest.approx(field.sum(), abs=1e-12)
    assert np.linalg.norm(moved) == pytest.approx(np.linalg.norm(field), abs=1e-12)


def test_bilinear_koopman_keeps_constant_observable():
    constant = np.ones((65, 65))
    assert np.array_equal(koopman_transport(constant, CAT_MAP), constant)


def test_seed_probe_is_mean_zero_unit_energy():
    probe = seed_probe()
    assert probe.mean() == pytest.approx(0.0, abs=1e-14)
    assert np.linalg.norm(probe) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("method", METHODS)
def test_every_frozen_kernel_bank_is_mean_zero_unit_energy(method):
    bank = generated_kernel_bank(method, orientation=0.0)
    assert bank.shape == (7, 65, 65)
    assert np.allclose(bank.mean(axis=(1, 2)), 0.0, atol=1e-12)
    assert np.allclose(np.linalg.norm(bank, axis=(1, 2)), 1.0, atol=1e-12)


def test_cat_forward_and_backward_orbit_kernels_differ():
    bank = generated_kernel_bank("E1_3_cat_koopman", orientation=0.0)
    assert np.linalg.norm(bank[0] - bank[-1]) > 0.25


def test_e0_full_mathematical_audit_passes_and_reports_discretization():
    result = run_e0(grid_size=129)
    assert result["status"] == "ANZA_EK_E0_PASS"
    assert result["l2_relative_error_bilinear_grid"] >= 0.0
    assert "approximation" in result["discrete_claim"]
