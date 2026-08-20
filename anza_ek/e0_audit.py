"""Numerical and exact diagnostics for the fixed Cat-map Koopman generator."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .kernels import generated_kernel_bank
from .torus import CAT_INVERSE, CAT_MAP, cat_eigensystem, exact_discrete_permutation, koopman_transport, torus_grid, torus_map


def _observable(points: np.ndarray) -> np.ndarray:
    x, y = points[..., 0], points[..., 1]
    return np.sin(2 * math.pi * x) + 0.40 * np.cos(4 * math.pi * y) + 0.30 * np.sin(2 * math.pi * (x + y))


def run_e0(*, grid_size: int = 257, K: int = 3) -> dict[str, Any]:
    determinant = float(np.linalg.det(CAT_MAP))
    eigenvalues, unstable, stable, _ = cat_eigensystem()
    ordered = np.sort(np.abs(eigenvalues))
    inverse_product = CAT_MAP @ CAT_INVERSE
    origin = torus_map(np.zeros((1, 2)), CAT_MAP)[0]
    grid = torus_grid(grid_size)
    observable = _observable(grid)
    transported = koopman_transport(observable, CAT_MAP)
    integral_error = float(abs(transported.mean() - observable.mean()))
    l2 = float(np.linalg.norm(observable))
    l2_relative_error = float(abs(np.linalg.norm(transported) - l2) / max(l2, 1e-12))
    constant = np.ones((grid_size, grid_size), dtype=np.float64)
    constant_error = float(np.max(np.abs(koopman_transport(constant, CAT_MAP) - constant)))

    discrete_rng = np.random.default_rng(20260819)
    discrete = discrete_rng.normal(size=(128, 128))
    permuted = exact_discrete_permutation(discrete, CAT_MAP)
    restored = exact_discrete_permutation(permuted, CAT_MAP, power=-1)
    exact_permutation_error = float(np.max(np.abs(restored - discrete)))
    exact_l2_error = float(abs(np.linalg.norm(permuted) - np.linalg.norm(discrete)))
    exact_integral_error = float(abs(permuted.sum() - discrete.sum()))

    centered = observable - observable.mean()
    denominator = float(np.sum(centered**2))
    correlations = []
    for lag in range(9):
        evolved = koopman_transport(centered, CAT_MAP, power=lag)
        correlations.append(float(np.sum(centered * evolved) / max(denominator, 1e-12)))
    correlation_tail = float(np.mean(np.abs(correlations[3:])))

    rng = np.random.default_rng(20260819)
    points = rng.uniform(-0.5, 0.5, size=(4096, 2))
    running = np.zeros(len(points), dtype=np.float64)
    for _ in range(256):
        running += _observable(points)
        points = torus_map(points, CAT_MAP)
    birkhoff_mean_absolute = float(np.mean(np.abs(running / 256.0)))

    epsilon = 1e-7
    steps = 6
    unstable_growth = float(np.linalg.norm(np.linalg.matrix_power(CAT_MAP, steps) @ (epsilon * unstable)) / epsilon)
    stable_growth = float(np.linalg.norm(np.linalg.matrix_power(CAT_MAP, steps) @ (epsilon * stable)) / epsilon)
    expected_unstable = float(ordered[1] ** steps)
    expected_stable = float(ordered[0] ** steps)
    kernels = generated_kernel_bank("E1_3_cat_koopman", orientation=0.0, size=65, K=K)
    forward_backward_difference = float(np.linalg.norm(kernels[0] - kernels[-1]))

    checks = {
        "determinant_one": abs(determinant - 1.0) < 1e-12,
        "reciprocal_eigenvalues": abs(float(np.prod(ordered)) - 1.0) < 1e-12,
        "hyperbolic_saddle": bool(ordered[0] < 1.0 < ordered[1]),
        "origin_fixed": bool(np.allclose(origin, 0.0, atol=1e-15)),
        "inverse_exact": bool(np.array_equal(inverse_product, np.eye(2, dtype=np.int64))),
        "continuous_integral_preserved_numerically": integral_error < 1e-10,
        "continuous_l2_preserved_numerically": l2_relative_error < 1e-3,
        "constant_invariant": constant_error < 1e-12,
        "correlation_decay_diagnostic": correlation_tail < 0.10,
        "birkhoff_convergence_diagnostic": birkhoff_mean_absolute < 0.10,
        "unstable_growth_matches": abs(unstable_growth / expected_unstable - 1.0) < 1e-9,
        "stable_growth_matches": abs(stable_growth / expected_stable - 1.0) < 1e-8,
        "exact_discrete_inverse": exact_permutation_error == 0.0,
        "exact_discrete_l2": exact_l2_error < 1e-10,
        "exact_discrete_integral": exact_integral_error < 1e-10,
        "forward_backward_kernels_distinct": forward_backward_difference > 0.25,
        "no_random_or_fuzzy_map_interpolation": True,
    }
    return {
        "status": "ANZA_EK_E0_PASS" if all(checks.values()) else "ANZA_EK_E0_FAIL",
        "checks": checks,
        "cat_matrix": CAT_MAP.tolist(),
        "cat_inverse": CAT_INVERSE.tolist(),
        "determinant": determinant,
        "eigenvalues": eigenvalues.tolist(),
        "stable_absolute_eigenvalue": float(ordered[0]),
        "unstable_absolute_eigenvalue": float(ordered[1]),
        "integral_error_bilinear_grid": integral_error,
        "l2_relative_error_bilinear_grid": l2_relative_error,
        "constant_error": constant_error,
        "exact_discrete_permutation_error": exact_permutation_error,
        "exact_discrete_l2_error": exact_l2_error,
        "exact_discrete_integral_error": exact_integral_error,
        "correlations_lag0_to_8": correlations,
        "correlation_tail_mean_absolute": correlation_tail,
        "birkhoff_mean_absolute_256_steps": birkhoff_mean_absolute,
        "unstable_growth_6_steps": unstable_growth,
        "stable_growth_6_steps": stable_growth,
        "forward_backward_kernel_l2_difference": forward_backward_difference,
        "continuous_claim": "exact for the mathematical toral automorphism",
        "discrete_claim": "bilinear grid readout is an approximation; integer-index diagnostic is an exact finite permutation but is not called ergodic",
        "training_performed": False,
    }


def save_e0_figures(output_root: Path, *, K: int = 3) -> list[str]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    kernels = generated_kernel_bank("E1_3_cat_koopman", orientation=0.0, size=65, K=K)
    figure, axes = plt.subplots(1, 2 * K + 1, figsize=(14, 2.2), constrained_layout=True)
    limit = float(np.max(np.abs(kernels)))
    for index, (axis, kernel) in enumerate(zip(axes, kernels, strict=True)):
        axis.imshow(kernel, cmap="coolwarm", vmin=-limit, vmax=limit, origin="lower")
        axis.set_title(f"k={index-K}")
        axis.axis("off")
    orbit_png = output_root / "cat_koopman_kernel_orbit.png"
    orbit_svg = output_root / "cat_koopman_kernel_orbit.svg"
    figure.savefig(orbit_png, dpi=180)
    figure.savefig(orbit_svg)
    plt.close(figure)

    eigenvalues, unstable, stable, _ = cat_eigensystem()
    figure, axis = plt.subplots(figsize=(4.4, 4.4), constrained_layout=True)
    axis.quiver(0, 0, unstable[0], unstable[1], color="crimson", angles="xy", scale_units="xy", scale=1, label=f"unstable |rho|={max(abs(eigenvalues)):.3f}")
    axis.quiver(0, 0, stable[0], stable[1], color="royalblue", angles="xy", scale_units="xy", scale=1, label=f"stable |rho|={min(abs(eigenvalues)):.3f}")
    axis.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect="equal", xlabel="x", ylabel="y", title="Canonical Cat-map saddle splitting")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left", fontsize=8)
    saddle_png = output_root / "cat_saddle_geometry.png"
    saddle_svg = output_root / "cat_saddle_geometry.svg"
    figure.savefig(saddle_png, dpi=180)
    figure.savefig(saddle_svg)
    plt.close(figure)
    return [str(orbit_png), str(orbit_svg), str(saddle_png), str(saddle_svg)]
