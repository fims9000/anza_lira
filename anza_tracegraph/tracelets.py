"""Deterministic ordered tracelets and endpoint geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from trace_extraction.graph import extract_trace_graph
from trace_extraction.skeleton import skeletonize_mask


@dataclass(frozen=True)
class Tracelet:
    tracelet_id: int
    points_yx: np.ndarray
    mean_probability: float
    mean_contrast: float

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.points_yx, axis=0), axis=1).sum())


@dataclass(frozen=True)
class Endpoint:
    tracelet_id: int
    end_index: int
    point_yx: tuple[float, float]
    outgoing_tangent_yx: tuple[float, float]
    confidence: float


def axial_tangent(points_yx: np.ndarray, *, end_index: int, tangent_points: int = 5) -> tuple[float, float]:
    points = np.asarray(points_yx, dtype=np.float64)
    if len(points) < 2 or end_index not in (0, -1): raise ValueError("tracelet tangent needs >=2 points and endpoint index 0/-1")
    if end_index == 0:
        vector = points[0] - points[min(tangent_points, len(points) - 1)]
    else:
        vector = points[-1] - points[max(0, len(points) - 1 - tangent_points)]
    norm = float(np.linalg.norm(vector))
    if norm <= 0: raise ValueError("degenerate tracelet tangent")
    vector /= norm
    return float(vector[0]), float(vector[1])


def endpoints(tracelet: Tracelet, tangent_points: int = 5) -> tuple[Endpoint, Endpoint]:
    first = Endpoint(tracelet.tracelet_id, 0, tuple(map(float, tracelet.points_yx[0])), axial_tangent(tracelet.points_yx, end_index=0, tangent_points=tangent_points), tracelet.mean_probability)
    last = Endpoint(tracelet.tracelet_id, -1, tuple(map(float, tracelet.points_yx[-1])), axial_tangent(tracelet.points_yx, end_index=-1, tangent_points=tangent_points), tracelet.mean_probability)
    return first, last


def extract_tracelets(mask: np.ndarray, probability: np.ndarray, image_scalar: np.ndarray, *, min_length: int = 8) -> tuple[Tracelet, ...]:
    skeleton = skeletonize_mask(mask)
    graph = extract_trace_graph(skeleton, border_margin=0)
    output = []
    for segment in graph.segments:
        points = np.asarray(segment.pixels, dtype=np.float64)
        if len(points) < min_length: continue
        pixels = np.asarray(segment.pixels, dtype=int); values = probability[pixels[:, 0], pixels[:, 1]]; contrast = image_scalar[pixels[:, 0], pixels[:, 1]]
        output.append(Tracelet(len(output), points, float(values.mean()), float(contrast.mean())))
    return tuple(output)


def tracelet_token(tracelet: Tracelet, probability: np.ndarray, image_scalar: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    points = tracelet.points_yx; diffs = np.diff(points, axis=0); angles = np.arctan2(diffs[:, 0], diffs[:, 1])
    curvature = np.diff(np.unwrap(angles)) if len(angles) > 1 else np.zeros(1)
    pixels = np.clip(np.rint(points).astype(int), (0, 0), (shape[0] - 1, shape[1] - 1)); probs = probability[pixels[:, 0], pixels[:, 1]]; intensities = image_scalar[pixels[:, 0], pixels[:, 1]]
    first, last = endpoints(tracelet)
    y_extent = max(float(np.ptp(points[:, 0])), 1.0)
    x_extent = max(float(np.ptp(points[:, 1])), 1.0)
    values = [
        tracelet.length / math.hypot(*shape), probs.mean(), probs.min(), probs.max(), probs[0], probs[-1],
        intensities.mean(), intensities.std(), float(np.mean(np.abs(curvature))), float(np.var(curvature)),
        x_extent / y_extent, first.outgoing_tangent_yx[0], first.outgoing_tangent_yx[1],
        last.outgoing_tangent_yx[0], last.outgoing_tangent_yx[1], 1.0,
    ]
    return np.asarray(values, dtype=np.float32)
