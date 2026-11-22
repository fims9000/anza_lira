"""Label-blind full-section fault-trace proposals from predicted probability.

This is a candidate-coverage stage, not geological identity assignment.
No expert gap rectangles, hidden paths, or class labels are inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree

from .graph import TraceGraph, extract_trace_graph
from .skeleton import skeletonize_mask


@dataclass(frozen=True)
class TracePort:
    port_id: int
    trace_id: int
    side: str
    pixel: tuple[int, int]
    outward_yx: tuple[float, float]


@dataclass(frozen=True)
class TracePair:
    first_port: int
    second_port: int
    first_trace: int
    second_trace: int
    gap_px: float
    source_alignment: float
    target_alignment: float
    geometric_cost: float


@dataclass(frozen=True)
class SectionCandidateSet:
    threshold: float
    graph: TraceGraph
    ports: tuple[TracePort, ...]
    pairs: tuple[TracePair, ...]
    foreground_fraction: float
    candidate_cap_per_port: int


def _outward(points: tuple[tuple[int, int], ...], side: str,
             tangent_pixels: int) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    if side == "start":
        a, b = points[0], points[min(len(points) - 1, tangent_pixels)]
    else:
        a, b = points[-1], points[max(0, len(points) - 1 - tangent_pixels)]
    v = np.asarray(a, float) - np.asarray(b, float)
    length = float(np.linalg.norm(v))
    if length < 1e-6:
        return None
    return float(v[0] / length), float(v[1] / length)


def full_section_candidate_graph(
    probability: np.ndarray, *, threshold: float = .65,
    min_segment_length_px: float = 8.,
    max_gap_px: float = 40.,
    min_alignment: float = .2,
    max_neighbors_per_port: int = 16,
    border_margin: int = 5,
    tangent_pixels: int = 5,
) -> SectionCandidateSet:
    p = np.asarray(probability, np.float32)
    if (p.ndim != 2 or min(p.shape) < 8
            or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1))):
        raise ValueError("Probability must be finite 2-D values in [0,1]")
    if not (0 < threshold < 1
            and min_segment_length_px > 0 and max_gap_px > 2
            and -1 <= min_alignment <= 1
            and isinstance(max_neighbors_per_port, int)
            and 1 <= max_neighbors_per_port <= 256
            and isinstance(border_margin, int) and border_margin >= 0
            and isinstance(tangent_pixels, int) and tangent_pixels >= 1):
        raise ValueError("Invalid full-section candidate policy")
    hard = p >= threshold
    fraction = float(hard.mean())
    if fraction > .38:
        raise ValueError("Dense segmentation cannot form reliable trace graph")
    graph = extract_trace_graph(skeletonize_mask(hard), border_margin=border_margin)
    ports = []
    for segment in graph.segments:
        if segment.pixel_length < min_segment_length_px:
            continue
        for side in ("start", "end"):
            if getattr(segment, side + "_type") != "endpoint":
                continue
            if getattr(segment, side + "_border_truncated"):
                continue
            orientation = _outward(segment.pixels, side, tangent_pixels)
            if orientation is None:
                continue
            pixel = segment.pixels[0 if side == "start" else -1]
            ports.append(TracePort(len(ports), segment.trace_id, side,
                                   pixel, orientation))

    # No global four-source or 48-pair cap. Endpoint-local budgets prevent a
    # crowded patch from monopolizing the whole-section candidate list.
    pairs = []
    if ports:
        positions = np.asarray([port.pixel for port in ports], dtype=float)
        tree = cKDTree(positions)
        for i, port in enumerate(ports):
            for j in tree.query_ball_point(positions[i], max_gap_px):
                if j <= i or ports[j].trace_id == port.trace_id:
                    continue
                delta = positions[j] - positions[i]
                length = float(np.linalg.norm(delta))
                if length < 2.:
                    continue
                direction = delta / length
                align_a = float(np.dot(port.outward_yx, direction))
                align_b = float(np.dot(ports[j].outward_yx, -direction))
                if min(align_a, align_b) < min_alignment:
                    continue
                cost = length + 8. * (
                    math.acos(float(np.clip(align_a, -1., 1.))) +
                    math.acos(float(np.clip(align_b, -1., 1.))))
                pairs.append(TracePair(
                    i, j, port.trace_id, ports[j].trace_id,
                    length, align_a, align_b, cost))
    pairs.sort(key=lambda item: (
        item.geometric_cost, item.first_port, item.second_port))
    budget = [0] * len(ports)
    kept = []
    for pair in pairs:
        if (budget[pair.first_port] >= max_neighbors_per_port
                or budget[pair.second_port] >= max_neighbors_per_port):
            continue
        kept.append(pair)
        budget[pair.first_port] += 1
        budget[pair.second_port] += 1
    return SectionCandidateSet(threshold, graph, tuple(ports),
                               tuple(kept), fraction, max_neighbors_per_port)


def multi_threshold_section_candidates(
    probability: np.ndarray, thresholds: tuple[float, ...] = (.65, .50, .35),
    **candidate_policy,
) -> tuple[SectionCandidateSet, ...]:
    """Preserve separate threshold layers; never union unrelated tracks."""
    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("Thresholds must be nonempty and distinct")
    return tuple(full_section_candidate_graph(
        probability, threshold=threshold, **candidate_policy)
        for threshold in thresholds)
