"""Secondary invariant-cone diagnostics; never part of the A2 primary gate."""

from __future__ import annotations

import math

import numpy as np


def contracted_half_angle(alpha: float, hyperbolicity: float) -> float:
    if not 0 <= alpha < math.pi / 2 or hyperbolicity < 0:
        raise ValueError("invalid cone parameters")
    return float(math.atan(math.exp(-2.0 * hyperbolicity) * math.tan(alpha)))


def axial_angle(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64); b = np.asarray(second, dtype=np.float64)
    a /= np.linalg.norm(a); b /= np.linalg.norm(b)
    return float(math.acos(np.clip(abs(float(a @ b)), 0.0, 1.0)))


def inside_cone(direction: np.ndarray, axis: np.ndarray, half_angle: float) -> bool:
    return axial_angle(direction, axis) <= half_angle + 1e-12
