"""Generic tangent and hyperbolic-cocycle local trajectory rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .frame import HyperbolicFrame
from .oracle_field import OracleCocycleField, aligned_mode


@dataclass(frozen=True)
class TrajectoryPoint:
    step: int
    x: float
    y: float
    ux: float
    uy: float
    branch_id: int
    membership: float
    curvature: float

    @property
    def xy(self) -> np.ndarray:
        return np.asarray((self.x, self.y), dtype=np.float64)

    @property
    def direction(self) -> np.ndarray:
        return np.asarray((self.ux, self.uy), dtype=np.float64)


def _unit(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (2,) or not np.isfinite(vector).all() or norm <= 1e-10:
        raise ValueError("finite nonzero direction required")
    return vector / norm


def rollout(
    field: OracleCocycleField,
    start_xy: np.ndarray,
    initial_direction: np.ndarray,
    *,
    steps: int = 4,
    delta: float = 1.0,
    hyperbolicity: float = 0.35,
    cocycle: bool,
) -> tuple[TrajectoryPoint, ...]:
    if steps < 1 or delta <= 0:
        raise ValueError("positive steps and delta required")
    position = np.asarray(start_xy, dtype=np.float64).copy()
    direction = _unit(initial_direction)
    output = []
    for step in range(steps + 1):
        local = field.query(position)
        axis, curvature, branch_id, membership = aligned_mode(local, direction)
        if step == 0:
            direction = axis
        output.append(TrajectoryPoint(
            step, float(position[0]), float(position[1]), float(direction[0]), float(direction[1]),
            branch_id, membership, curvature,
        ))
        if step == steps:
            break
        next_position = position + float(delta) * direction
        if cocycle:
            theta = float(np.arctan2(direction[1], direction[0]))
            frame = HyperbolicFrame(theta, curvature, hyperbolicity, float(delta))
            transported = frame.transport(direction)
            next_local = field.query(next_position)
            direction, _curvature, _branch, _membership = aligned_mode(next_local, transported)
        else:
            next_local = field.query(next_position)
            direction, _curvature, _branch, _membership = aligned_mode(next_local, direction)
        position = next_position
    return tuple(output)


def trajectory_arrays(points: Iterable[TrajectoryPoint]) -> tuple[np.ndarray, np.ndarray]:
    values = tuple(points)
    return (
        np.asarray([[point.x, point.y] for point in values], dtype=np.float64),
        np.asarray([[point.ux, point.uy] for point in values], dtype=np.float64),
    )


def cocycle_product(frames: Iterable[HyperbolicFrame]) -> np.ndarray:
    product = np.eye(2, dtype=np.float64)
    for frame in frames:
        product = frame.matrix() @ product
    return product


def stable_widths(steps: int, hyperbolicity: float, *, width0: float = 1.5, width_min: float = 0.25) -> np.ndarray:
    if steps < 0 or hyperbolicity < 0 or width0 < width_min:
        raise ValueError("invalid stable-width parameters")
    accumulated = hyperbolicity * np.arange(steps + 1, dtype=np.float64)
    return width_min + (width0 - width_min) * np.exp(-accumulated)


def bilinear_sample(array: np.ndarray, xy: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    point = np.asarray(xy, dtype=np.float64)
    if values.ndim not in (2, 3) or point.shape != (2,):
        raise ValueError("array must be HxW[xC] and xy a 2-vector")
    height, width = values.shape[:2]
    x = float(np.clip(point[0], 0, width - 1)); y = float(np.clip(point[1], 0, height - 1))
    x0, y0 = int(np.floor(x)), int(np.floor(y)); x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    wx, wy = x - x0, y - y0
    result = (
        (1 - wx) * (1 - wy) * values[y0, x0]
        + wx * (1 - wy) * values[y0, x1]
        + (1 - wx) * wy * values[y1, x0]
        + wx * wy * values[y1, x1]
    )
    if not np.isfinite(result).all():
        raise ValueError("bilinear sample is not finite")
    return result


def residual_output(base: np.ndarray, correction: np.ndarray, gamma: float = 0.0) -> np.ndarray:
    return np.asarray(base) + float(gamma) * np.asarray(correction)
