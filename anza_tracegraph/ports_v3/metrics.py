"""Synthetic-only branch identity metrics for SBPP."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .branches import Branch


def branch_match(branch: Branch, truth_yx: np.ndarray, *, radius: float = 3.0, minimum_fraction: float = 0.60) -> tuple[bool, float, float]:
    distances = cKDTree(np.asarray(truth_yx, dtype=float)).query(np.asarray(branch.points_yx, dtype=float))[0]
    fraction = float(np.mean(distances <= radius)); median = float(np.median(distances))
    return bool(fraction >= minimum_fraction), fraction, median


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0: return 0.0, 0.0
    p = successes / total; denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return float(center - half), float(center + half)
