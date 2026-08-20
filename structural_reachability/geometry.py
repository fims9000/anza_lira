"""Hand-checkable axial/fuzzy components used by the frozen Phase-A probe."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np


EPSILON = 1e-8


def compute_axial_consistency(theta_p: np.ndarray | float, theta_q: np.ndarray | float) -> np.ndarray:
    """Return axial agreement in [0, 1], invariant to either angle plus pi."""

    first = np.asarray(theta_p, dtype=np.float64)
    second = np.asarray(theta_q, dtype=np.float64)
    value = 0.5 * (1.0 + np.cos(2.0 * (first - second)))
    return np.clip(value, 0.0, 1.0)


def compute_fuzzy_compatibility(
    membership_p: np.ndarray,
    membership_q: np.ndarray,
    *,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """Cosine compatibility of nonnegative fuzzy-membership vectors."""

    first = np.asarray(membership_p, dtype=np.float64)
    second = np.asarray(membership_q, dtype=np.float64)
    if first.shape != second.shape or first.ndim < 1:
        raise ValueError("membership arrays must have the same non-scalar shape")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("membership arrays must be finite")
    if np.any(first < 0) or np.any(second < 0):
        raise ValueError("fuzzy memberships must be nonnegative")
    numerator = np.sum(first * second, axis=-1)
    denominator = np.sqrt(np.sum(first * first, axis=-1) + epsilon)
    denominator *= np.sqrt(np.sum(second * second, axis=-1) + epsilon)
    return np.clip(numerator / denominator, 0.0, 1.0)


def compute_scale_compatibility(
    sigma_parallel_p: np.ndarray | float,
    sigma_parallel_q: np.ndarray | float,
    sigma_perpendicular_p: np.ndarray | float,
    sigma_perpendicular_q: np.ndarray | float,
    *,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """Log-ratio compatibility for positive longitudinal/transverse scales."""

    sp_p, sp_q, st_p, st_q = np.broadcast_arrays(
        np.asarray(sigma_parallel_p, dtype=np.float64),
        np.asarray(sigma_parallel_q, dtype=np.float64),
        np.asarray(sigma_perpendicular_p, dtype=np.float64),
        np.asarray(sigma_perpendicular_q, dtype=np.float64),
    )
    for value in (sp_p, sp_q, st_p, st_q):
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("geometry scales must be finite and nonnegative")
    penalty = np.abs(np.log((sp_p + epsilon) / (sp_q + epsilon)))
    penalty += np.abs(np.log((st_p + epsilon) / (st_q + epsilon)))
    return np.clip(np.exp(-penalty), 0.0, 1.0)


def compute_directed_anisotropic_factor(
    theta_p: np.ndarray | float,
    sigma_parallel_p: np.ndarray | float,
    sigma_perpendicular_p: np.ndarray | float,
    delta_y: np.ndarray | float,
    delta_x: np.ndarray | float,
    *,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """Directed Gaussian geometry from p to q using the frozen specification."""

    theta, sp, st, dy, dx = np.broadcast_arrays(
        np.asarray(theta_p, dtype=np.float64),
        np.asarray(sigma_parallel_p, dtype=np.float64),
        np.asarray(sigma_perpendicular_p, dtype=np.float64),
        np.asarray(delta_y, dtype=np.float64),
        np.asarray(delta_x, dtype=np.float64),
    )
    if not all(np.isfinite(value).all() for value in (theta, sp, st, dy, dx)):
        raise ValueError("anisotropic geometry inputs must be finite")
    if np.any(sp <= 0) or np.any(st <= 0):
        raise ValueError("anisotropic scales must be positive")
    parallel = dx * np.cos(theta) + dy * np.sin(theta)
    perpendicular = -dx * np.sin(theta) + dy * np.cos(theta)
    exponent = -0.5 * (parallel / (sp + epsilon)) ** 2
    exponent -= 0.5 * (perpendicular / (st + epsilon)) ** 2
    return np.clip(np.exp(exponent), 0.0, 1.0)


def symmetrize_affinity(
    forward: np.ndarray | float,
    reverse: np.ndarray | float,
    *,
    method: Literal["geometric_mean", "minimum", "average"] = "geometric_mean",
) -> np.ndarray:
    """Symmetrize directed affinities without selecting a rule from test data."""

    first, second = np.broadcast_arrays(
        np.asarray(forward, dtype=np.float64), np.asarray(reverse, dtype=np.float64)
    )
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("affinities must be finite")
    if np.any((first < 0) | (first > 1)) or np.any((second < 0) | (second > 1)):
        raise ValueError("affinities must lie in [0, 1]")
    if method == "geometric_mean":
        return np.sqrt(first * second)
    if method == "minimum":
        return np.minimum(first, second)
    if method == "average":
        return 0.5 * (first + second)
    raise ValueError(f"unknown symmetrization method: {method}")


def axial_mean(theta: np.ndarray, membership: np.ndarray) -> float:
    """Membership-weighted mean of axial angles."""

    angles = np.asarray(theta, dtype=np.float64)
    weights = np.asarray(membership, dtype=np.float64)
    if angles.shape != weights.shape or angles.ndim != 1:
        raise ValueError("theta and membership must be same-length vectors")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("membership vector must have positive mass")
    cosine = float(np.sum(weights * np.cos(2.0 * angles)) / total)
    sine = float(np.sum(weights * np.sin(2.0 * angles)) / total)
    return 0.5 * math.atan2(sine, cosine)


def log_geometric_mean(values: np.ndarray, *, epsilon: float = EPSILON) -> np.ndarray:
    """Equal-weight geometric mean, used to avoid a component-count scale bias."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] < 1:
        raise ValueError("values must have a nonempty component axis")
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError("relation components must be finite in [0, 1]")
    return np.exp(np.mean(np.log(np.clip(array, epsilon, 1.0)), axis=-1))
