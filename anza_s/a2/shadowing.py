"""Normalized two-sided shadowing inside transported uncertainty tubes."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp

from anza_s.rollout import TrajectoryPoint


def hyperbolic_shadowing(
    left: tuple[TrajectoryPoint, ...],
    right: tuple[TrajectoryPoint, ...],
    left_covariance: tuple[np.ndarray, ...],
    right_covariance: tuple[np.ndarray, ...],
    *,
    eta_theta: float = 2.0,
    temperature: float = 0.25,
    epsilon: float = 1e-6,
) -> tuple[float, float, tuple[int, int], np.ndarray]:
    if not left or not right or len(left) != len(left_covariance) or len(right) != len(right_covariance):
        raise ValueError("trajectory/covariance lengths must match and be nonempty")
    if eta_theta < 0 or temperature <= 0 or epsilon <= 0:
        raise ValueError("invalid shadowing parameters")
    energy = np.empty((len(left), len(right)), dtype=np.float64)
    identity = np.eye(2, dtype=np.float64)
    for i, first in enumerate(left):
        for j, second in enumerate(right):
            delta = second.xy - first.xy
            covariance = np.asarray(left_covariance[i]) + np.asarray(right_covariance[j]) + epsilon * identity
            spatial = float(delta @ np.linalg.solve(covariance, delta))
            axial = float(eta_theta * (1.0 - float(np.dot(first.direction, second.direction)) ** 2))
            energy[i, j] = spatial + axial
    flat = energy.ravel()
    # The mean, rather than the unnormalised sum, prevents score inflation when
    # the number of rollout states changes.
    soft_energy = float(-temperature * (logsumexp(-flat / temperature) - math.log(flat.size)))
    score = float(math.exp(-max(soft_energy, 0.0)))
    meeting = tuple(int(value) for value in np.unravel_index(int(np.argmin(energy)), energy.shape))
    if not (0.0 < score <= 1.0) or not np.isfinite(energy).all():
        raise AssertionError("shadowing score must be finite and bounded")
    return soft_energy, score, meeting, energy
