"""Directed terminal ports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .branches import Branch


@dataclass(frozen=True)
class Port:
    branch_id: int
    point_yx: tuple[float, float]
    tangent_yx: tuple[float, float]
    confidence: float
    port_type: str
    end_index: int


def _outward(points: np.ndarray, end_index: int, count: int = 5) -> tuple[float, float]:
    if end_index == 0: vector = points[0] - points[min(count, len(points) - 1)]
    else: vector = points[-1] - points[max(0, len(points) - 1 - count)]
    vector = vector / max(float(np.linalg.norm(vector)), 1e-8); return tuple(map(float, vector))


def terminal_ports(branches: tuple[Branch, ...]) -> tuple[Port, ...]:
    output = []
    for branch in branches:
        for end_index, endpoint_type in ((0, branch.start_type), (-1, branch.end_type)):
            if endpoint_type == "junction": continue
            point = branch.points_yx[end_index]
            output.append(Port(branch.branch_id, tuple(map(float, point)), _outward(branch.points_yx, end_index), branch.mean_probability, "terminal", end_index))
    return tuple(output)
