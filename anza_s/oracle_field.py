"""Generator-derived oracle tangent/curvature atlas for ANZA-S Phase A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from synthetic.crossing_trace_bench_v4 import sample_seed_v4
from synthetic.geometry_generator import GeometrySample, generate_geometry, scale_geometry


@dataclass(frozen=True)
class LocalModes:
    axes: np.ndarray
    memberships: np.ndarray
    curvatures: np.ndarray
    branch_ids: np.ndarray


def geometry_for_sample(sample: dict[str, Any]) -> GeometrySample:
    seed = sample_seed_v4(str(sample["split"]), int(sample["index"]))
    geometry = generate_geometry(str(sample["case"]), np.random.default_rng(seed))
    return scale_geometry(geometry, int(sample["image_size"]))


class OracleCocycleField:
    """Continuous nearest-polyline oracle; used only for feasibility, never supervision."""

    def __init__(self, geometry: GeometrySample, *, support_sigma: float = 2.5) -> None:
        self.geometry = geometry
        self.support_sigma = float(support_sigma)
        if self.support_sigma <= 0:
            raise ValueError("support_sigma must be positive")
        self._branches = []
        for branch in geometry.branches:
            points = np.asarray(branch.points_xy, dtype=np.float64)
            delta = np.gradient(points, axis=0)
            lengths = np.linalg.norm(delta, axis=1)
            tangent = delta / np.maximum(lengths[:, None], 1e-8)
            theta = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
            segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
            arc = np.r_[0.0, np.cumsum(segment)]
            curvature = np.gradient(theta, arc, edge_order=1)
            self._branches.append((int(branch.branch_id), points, tangent, curvature))

    def query(self, xy: np.ndarray) -> LocalModes:
        point = np.asarray(xy, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("xy must be a finite 2-vector")
        axes, memberships, curvatures, branch_ids = [], [], [], []
        for branch_id, points, tangent, curvature in self._branches:
            distances = np.sum((points - point) ** 2, axis=1)
            selected = int(np.argmin(distances))
            axes.append(tangent[selected])
            memberships.append(np.exp(-0.5 * distances[selected] / self.support_sigma**2))
            curvatures.append(curvature[selected])
            branch_ids.append(branch_id)
        return LocalModes(
            axes=np.asarray(axes, dtype=np.float64),
            memberships=np.clip(np.asarray(memberships, dtype=np.float64), 1e-8, 1.0),
            curvatures=np.asarray(curvatures, dtype=np.float64),
            branch_ids=np.asarray(branch_ids, dtype=np.int64),
        )


def aligned_mode(local: LocalModes, incoming: np.ndarray, *, temperature: float = 0.08) -> tuple[np.ndarray, float, int, float]:
    """Top-1 oracle diagnostic after soft axial compatibility scoring."""

    direction = np.asarray(incoming, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    axes = local.axes / np.maximum(np.linalg.norm(local.axes, axis=1, keepdims=True), 1e-8)
    logits = np.log(local.memberships) + (axes @ direction) ** 2 / float(temperature)
    selected = int(np.argmax(logits))
    sign = 1.0 if float(np.dot(axes[selected], direction)) >= 0 else -1.0
    axis = sign * axes[selected]
    curvature = sign * float(local.curvatures[selected])
    return axis, curvature, int(local.branch_ids[selected]), float(local.memberships[selected])
