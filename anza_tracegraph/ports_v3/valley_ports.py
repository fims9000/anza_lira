"""Logical ports around persistent confidence valleys; masks remain unchanged."""

from __future__ import annotations

import numpy as np

from .branches import Branch
from .terminal_ports import Port


def _sample(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    pixels = np.rint(points).astype(int); pixels[:, 0] = np.clip(pixels[:, 0], 0, field.shape[0] - 1); pixels[:, 1] = np.clip(pixels[:, 1], 0, field.shape[1] - 1)
    return field[pixels[:, 0], pixels[:, 1]]


def confidence_valley_ports(branches: tuple[Branch, ...], probability: np.ndarray) -> tuple[Port, ...]:
    output: list[Port] = []
    for branch in branches:
        values = np.asarray(_sample(probability, branch.points_yx), dtype=float)
        if len(values) < 15: continue
        ratios = np.ones(len(values), dtype=float)
        for index in range(6, len(values) - 6):
            support = min(float(values[index - 6 : index].mean()), float(values[index + 1 : index + 7].mean()))
            if support > 1e-8: ratios[index] = values[index] / support
        candidates = [index for index in range(6, len(values) - 6) if ratios[index] < 0.80 and (ratios[max(0, index - 1)] < 0.80 or ratios[min(len(values) - 1, index + 1)] < 0.80)]
        selected: list[int] = []
        for index in sorted(candidates, key=lambda item: (ratios[item], item)):
            if all(abs(index - old) > 5 for old in selected): selected.append(index)
        for index in sorted(selected):
            left = max(0, index - 2); right = min(len(values) - 1, index + 2)
            for side, toward, label in ((left, index, "valley_left"), (right, index, "valley_right")):
                vector = branch.points_yx[toward] - branch.points_yx[side]; vector /= max(float(np.linalg.norm(vector)), 1e-8)
                output.append(Port(branch.branch_id, tuple(map(float, branch.points_yx[side])), tuple(map(float, vector)), float(values[side]), label, side))
    return tuple(output)
