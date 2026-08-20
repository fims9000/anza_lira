"""Two-sided pseudo-orbit meeting energy and score."""

from __future__ import annotations

import math

import numpy as np

from .rollout import TrajectoryPoint, trajectory_arrays


def meeting_energy_matrix(
    left: tuple[TrajectoryPoint, ...],
    right: tuple[TrajectoryPoint, ...],
    *,
    sigma_x: float = 1.5,
    eta_theta: float = 2.0,
) -> np.ndarray:
    if sigma_x <= 0 or eta_theta < 0:
        raise ValueError("invalid shadowing parameters")
    left_xy, left_u = trajectory_arrays(left); right_xy, right_u = trajectory_arrays(right)
    spatial = np.sum((left_xy[:, None] - right_xy[None]) ** 2, axis=2) / sigma_x**2
    # cos(2 delta theta) = 2 (u dot v)^2 - 1 for unit axial frames.
    dot = np.clip(left_u @ right_u.T, -1.0, 1.0)
    orientation = eta_theta * (2.0 - 2.0 * dot**2)
    return spatial + orientation


def two_sided_shadowing(
    left: tuple[TrajectoryPoint, ...],
    right: tuple[TrajectoryPoint, ...],
    *,
    sigma_x: float = 1.5,
    eta_theta: float = 2.0,
    temperature: float = 0.25,
) -> tuple[float, float, tuple[int, int]]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    energies = meeting_energy_matrix(left, right, sigma_x=sigma_x, eta_theta=eta_theta)
    scaled = -energies / temperature
    maximum = float(scaled.max())
    # Exact packet formula: unnormalized soft minimum over all k,j alignments.
    energy = -temperature * (maximum + math.log(float(np.exp(scaled - maximum).sum())))
    score = math.exp(-energy)
    meeting = tuple(map(int, np.unravel_index(int(np.argmin(energies)), energies.shape)))
    return float(energy), float(score), meeting


def terminal_meeting_score(
    left: tuple[TrajectoryPoint, ...], right: tuple[TrajectoryPoint, ...], *, sigma_x: float = 1.5
) -> float:
    distance2 = float(np.sum((left[-1].xy - right[-1].xy) ** 2))
    return math.exp(-distance2 / sigma_x**2)
