"""Deterministic 12-pixel virtual landing bands."""

from __future__ import annotations

import numpy as np

from .branches import Branch
from .terminal_ports import Port


LANDING_BAND_PX = 12.0
LANDING_STEP_PX = 2.0


def _sample_from_end(branch: Branch, end_index: int) -> tuple[Port, ...]:
    points = branch.points_yx if end_index == 0 else branch.points_yx[::-1]
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    output = []
    for target in np.arange(0.0, min(LANDING_BAND_PX, float(cumulative[-1])) + 1e-8, LANDING_STEP_PX):
        index = min(int(np.searchsorted(cumulative, target, side="right")), len(points) - 1); left = max(0, index - 1)
        span = max(float(cumulative[index] - cumulative[left]), 1e-8); weight = float((target - cumulative[left]) / span)
        point = points[left] * (1 - weight) + points[index] * weight
        # The destination tangent points from the branch interior toward this end.
        inward_index = min(len(points) - 1, index + 3); vector = points[0] - points[inward_index]; vector /= max(float(np.linalg.norm(vector)), 1e-8)
        output.append(Port(branch.branch_id, tuple(map(float, point)), tuple(map(float, vector)), branch.mean_probability, "virtual_landing", end_index))
    return tuple(output)


def virtual_landing_ports(branch: Branch) -> tuple[Port, ...]:
    return _sample_from_end(branch, 0) + _sample_from_end(branch, -1)
