"""Frozen 0.70-radian curvature-discontinuity splitting."""

from __future__ import annotations

import math

import numpy as np


CURVATURE_SPLIT_RADIANS = 0.70


def _axial_distance(first: float, second: float) -> float:
    delta = abs((first - second) % math.pi)
    return float(min(delta, math.pi - delta))


def robust_tangents(points_yx: np.ndarray) -> np.ndarray:
    points = np.asarray(points_yx, dtype=np.float64)
    if len(points) < 2: raise ValueError("tangent needs at least two points")
    output = np.empty((len(points), 2), dtype=np.float64)
    for index in range(len(points)):
        left = max(0, index - 2); right = min(len(points) - 1, index + 2)
        vector = points[right] - points[left]
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            vector = points[min(index + 1, len(points) - 1)] - points[max(index - 1, 0)]; norm = float(np.linalg.norm(vector))
        output[index] = vector / max(norm, 1e-8)
    return output


def split_at_curvature(points_yx: np.ndarray, threshold: float = CURVATURE_SPLIT_RADIANS) -> tuple[np.ndarray, ...]:
    """Split only after two consecutive adjacent tangent discontinuities."""
    points = np.asarray(points_yx, dtype=np.float64)
    if len(points) < 7: return (points,)
    tangents = robust_tangents(points); angles = np.arctan2(tangents[:, 0], tangents[:, 1])
    # Compare the robust estimates immediately to either side of a center.
    # Comparing overlapping center estimates directly makes a right-angle
    # corner decay into four sub-threshold rotations and renders the declared
    # split inert.
    jumps = np.asarray([_axial_distance(float(angles[index - 1]), float(angles[index + 1])) > threshold for index in range(1, len(angles) - 1)])
    split_indices: list[int] = []
    start = 0
    while start < len(jumps):
        if not jumps[start]: start += 1; continue
        end = start
        while end + 1 < len(jumps) and jumps[end + 1]: end += 1
        if end - start + 1 >= 2:
            location = (start + end + 2) // 2
            if location >= 3 and len(points) - location >= 3: split_indices.append(location)
        start = end + 1
    if not split_indices: return (points,)
    boundaries = [0] + sorted(set(split_indices)) + [len(points) - 1]
    parts = tuple(points[left : right + 1] for left, right in zip(boundaries[:-1], boundaries[1:]) if right - left + 1 >= 2)
    return parts or (points,)
