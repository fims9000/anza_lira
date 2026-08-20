"""Hard trace branches plus candidate-only short branches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trace_extraction.graph import extract_trace_graph
from trace_extraction.skeleton import skeletonize_mask

from .curvature_split import robust_tangents, split_at_curvature


@dataclass(frozen=True)
class Branch:
    branch_id: int
    points_yx: np.ndarray
    mean_probability: float
    mean_contrast: float
    orientation_coherence: float
    candidate_only: bool
    start_type: str
    end_type: str
    start_node: int | None
    end_node: int | None

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.points_yx, axis=0), axis=1).sum())


def _values(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    pixels = np.rint(points).astype(int); pixels[:, 0] = np.clip(pixels[:, 0], 0, field.shape[0] - 1); pixels[:, 1] = np.clip(pixels[:, 1], 0, field.shape[1] - 1)
    return np.asarray(field[pixels[:, 0], pixels[:, 1]], dtype=np.float64)


def _coherence(points: np.ndarray) -> float:
    tangents = robust_tangents(points); angles = np.arctan2(tangents[:, 0], tangents[:, 1])
    return float(np.hypot(np.cos(2 * angles).mean(), np.sin(2 * angles).mean()))


def extract_branches(mask: np.ndarray, probability: np.ndarray, image_scalar: np.ndarray, *, tau_micro: float, min_normal_length: float = 8.0) -> tuple[Branch, ...]:
    graph = extract_trace_graph(skeletonize_mask(mask), border_margin=0)
    output: list[Branch] = []
    for segment in graph.segments:
        original = np.asarray(segment.pixels, dtype=np.float64)
        parts = split_at_curvature(original)
        for part_index, points in enumerate(parts):
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if length < 4.0: continue
            probabilities = _values(probability, points); coherence = _coherence(points); candidate_only = length < min_normal_length
            if candidate_only and (float(probabilities.mean()) < tau_micro or coherence < 0.55): continue
            start_original = part_index == 0; end_original = part_index == len(parts) - 1
            output.append(Branch(
                branch_id=len(output), points_yx=points, mean_probability=float(probabilities.mean()), mean_contrast=float(_values(image_scalar, points).mean()), orientation_coherence=coherence, candidate_only=candidate_only,
                start_type=segment.start_type if start_original else "curvature_split", end_type=segment.end_type if end_original else "curvature_split",
                start_node=segment.start_node if start_original else None, end_node=segment.end_node if end_original else None,
            ))
    return tuple(output)
