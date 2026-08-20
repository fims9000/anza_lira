"""One directed junction-arm port per incident branch."""

from __future__ import annotations

import numpy as np

from .branches import Branch
from .terminal_ports import Port


JUNCTION_OFFSET_PX = 4.0


def _point_at(points: np.ndarray, distance: float) -> tuple[np.ndarray, np.ndarray]:
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    target = min(distance, float(cumulative[-1])); index = min(int(np.searchsorted(cumulative, target, side="right")), len(points) - 1)
    left = max(0, index - 1); span = max(float(cumulative[index] - cumulative[left]), 1e-8); weight = (target - cumulative[left]) / span
    point = points[left] * (1 - weight) + points[index] * weight
    direction = points[min(index + 2, len(points) - 1)] - points[max(0, index - 2)]; direction /= max(float(np.linalg.norm(direction)), 1e-8)
    return point, direction


def junction_arm_ports(branches: tuple[Branch, ...]) -> tuple[Port, ...]:
    output = []
    for branch in branches:
        if branch.start_type == "junction":
            point, direction = _point_at(branch.points_yx, JUNCTION_OFFSET_PX); output.append(Port(branch.branch_id, tuple(map(float, point)), tuple(map(float, direction)), branch.mean_probability, "junction_arm", 0))
        if branch.end_type == "junction":
            reversed_points = branch.points_yx[::-1]; point, direction = _point_at(reversed_points, JUNCTION_OFFSET_PX); output.append(Port(branch.branch_id, tuple(map(float, point)), tuple(map(float, direction)), branch.mean_probability, "junction_arm", -1))
    return tuple(output)
