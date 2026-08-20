"""Reset-local and composed covariance transport on one frozen centerline."""

from __future__ import annotations

import math

import numpy as np

from anza_s.frame import HyperbolicFrame
from anza_s.rollout import TrajectoryPoint

from .cauchy_green import cocycle_product, finite_time_diagnostics


def frame_matrices(
    trajectory: tuple[TrajectoryPoint, ...], *, hyperbolicity: float, delta: float = 1.0,
) -> tuple[np.ndarray, ...]:
    if len(trajectory) < 2 or hyperbolicity < 0 or delta <= 0:
        raise ValueError("trajectory needs at least one transition and valid parameters")
    return tuple(
        HyperbolicFrame(
            theta=math.atan2(point.uy, point.ux),
            curvature=point.curvature,
            hyperbolicity=hyperbolicity,
            step=delta,
        ).matrix()
        for point in trajectory[:-1]
    )


def covariance_sequence(
    trajectory: tuple[TrajectoryPoint, ...], *, mode: str, hyperbolicity: float,
    delta: float = 1.0, epsilon: float = 1e-9,
) -> tuple[np.ndarray, ...]:
    """Return covariance at every trajectory point.

    ``isotropic`` keeps I. ``local_reset`` applies only the immediately
    preceding J to I. ``composed`` recursively transports the prior covariance.
    This makes local_reset and composed exactly equal for a one-step rollout.
    """

    if mode not in {"isotropic", "local_reset", "composed"}:
        raise ValueError(f"unknown covariance mode: {mode}")
    matrices = frame_matrices(trajectory, hyperbolicity=hyperbolicity, delta=delta)
    identity = np.eye(2, dtype=np.float64)
    output = [identity]
    covariance = identity
    for matrix in matrices:
        if mode == "isotropic":
            covariance = identity
        elif mode == "local_reset":
            covariance = matrix @ identity @ matrix.T
        else:
            covariance = matrix @ covariance @ matrix.T
        covariance = 0.5 * (covariance + covariance.T)
        if np.linalg.eigvalsh(covariance)[0] <= epsilon:
            raise ValueError("transported covariance lost positive definiteness")
        output.append(covariance)
    return tuple(output)


def trajectory_diagnostics(
    trajectory: tuple[TrajectoryPoint, ...], *, hyperbolicity: float, delta: float = 1.0,
) -> list[dict[str, float]]:
    matrices = frame_matrices(trajectory, hyperbolicity=hyperbolicity, delta=delta)
    rows = []
    for step in range(1, len(trajectory)):
        product = cocycle_product(matrices[:step])
        rows.append({"step": float(step), **finite_time_diagnostics(product, elapsed=step * delta)})
    return rows
