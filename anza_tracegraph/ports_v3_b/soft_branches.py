"""Bounded source-directed soft-support branch extraction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

from trace_extraction.graph import extract_trace_graph
from trace_extraction.skeleton import skeletonize_mask

from anza_tracegraph.ports_v3.curvature_split import robust_tangents, split_at_curvature
from anza_tracegraph.ports_v3.terminal_ports import Port


@dataclass(frozen=True)
class SoftBranch:
    branch_id: int
    points_yx: np.ndarray
    mean_probability: float
    mean_contrast: float
    orientation_coherence: float
    candidate_only: bool = True
    candidate_only_soft: bool = True
    start_type: str = "soft_endpoint"
    end_type: str = "soft_endpoint"
    start_node: int | None = None
    end_node: int | None = None
    hysteresis_rule: str = ""

    @property
    def length(self) -> float: return float(np.linalg.norm(np.diff(self.points_yx, axis=0), axis=1).sum())


def source_sector_mask(shape: tuple[int, int], source: Port, *, minimum_distance: float = 6.0, maximum_distance: float = 68.0, maximum_angle: float = math.radians(78.0)) -> np.ndarray:
    y, x = np.mgrid[: shape[0], : shape[1]]; delta = np.stack((y - source.point_yx[0], x - source.point_yx[1]), axis=-1); distance = np.linalg.norm(delta, axis=-1)
    direction = delta / np.maximum(distance[..., None], 1e-8); dot = direction[..., 0] * source.tangent_yx[0] + direction[..., 1] * source.tangent_yx[1]
    return (distance >= minimum_distance) & (distance <= maximum_distance) & (dot > 0.0) & (dot >= math.cos(maximum_angle))


def _sample(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    pixels = np.rint(points).astype(int); pixels[:, 0] = np.clip(pixels[:, 0], 0, field.shape[0] - 1); pixels[:, 1] = np.clip(pixels[:, 1], 0, field.shape[1] - 1)
    return np.asarray(field[pixels[:, 0], pixels[:, 1]], dtype=np.float64)


def _coherence(points: np.ndarray) -> float:
    tangent = robust_tangents(points); angles = np.arctan2(tangent[:, 0], tangent[:, 1]); return float(np.hypot(np.cos(2 * angles).mean(), np.sin(2 * angles).mean()))


def extract_soft_branches(probability: np.ndarray, image_scalar: np.ndarray, hard_mask: np.ndarray, source: Port, *, tau_s: float, excluded_mask: np.ndarray | None = None) -> tuple[SoftBranch, ...]:
    """Extract H1/H2 components without modifying ``hard_mask``."""
    sector = source_sector_mask(probability.shape, source); soft_mask = (np.asarray(probability) >= tau_s) & sector
    if excluded_mask is not None: soft_mask &= ~np.asarray(excluded_mask, dtype=bool)
    components, count = label(soft_mask, structure=np.ones((3, 3), dtype=np.uint8)); hard_points = np.argwhere(hard_mask); hard_tree = cKDTree(hard_points) if len(hard_points) else None
    output: list[SoftBranch] = []
    for component_id in range(1, count + 1):
        component = components == component_id; pixels = np.argwhere(component)
        if len(pixels) < 4: continue
        anchored = bool(hard_tree is not None and float(np.min(hard_tree.query(pixels)[0])) <= 3.0)
        graph = extract_trace_graph(skeletonize_mask(component), border_margin=0); parts: list[tuple[np.ndarray, float, float, float]] = []
        for segment in graph.segments:
            for points in split_at_curvature(np.asarray(segment.pixels, dtype=np.float64)):
                length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
                if length < 4.0: continue
                parts.append((points, length, float(_sample(probability, points).mean()), _coherence(points)))
        if not parts: continue
        total_length = sum(item[1] for item in parts); weighted_probability = sum(item[1] * item[2] for item in parts) / max(total_length, 1e-8)
        orientation_vectors = []
        for points, length, _, _ in parts:
            vector = points[-1] - points[0]; angle = math.atan2(float(vector[0]), float(vector[1])); orientation_vectors.append((length * math.cos(2 * angle), length * math.sin(2 * angle)))
        component_coherence = math.hypot(sum(item[0] for item in orientation_vectors), sum(item[1] for item in orientation_vectors)) / max(total_length, 1e-8)
        self_supported = total_length >= 6.0 and weighted_probability >= tau_s + 0.03 and component_coherence >= 0.60
        if not anchored and not self_supported: continue
        for points, _, mean_probability, coherence in parts:
            output.append(SoftBranch(len(output), points, mean_probability, float(_sample(image_scalar, points).mean()), coherence, hysteresis_rule="H1_hard_anchored" if anchored else "H2_self_supported"))
    return tuple(output)
