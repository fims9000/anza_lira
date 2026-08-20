"""Endpoint candidates and deterministic maximum-bottleneck paths."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import numpy as np
from scipy.ndimage import binary_dilation

from models.azconv_affinity import LOCAL8_OFFSETS
from synthetic.structural_metrics import skeletonize_mask
from trace_extraction.graph import extract_trace_graph


Pixel = tuple[int, int]


@dataclass(frozen=True)
class EndpointPair:
    first: Pixel
    second: Pixel
    distance: float


def candidate_endpoint_pairs(
    visible_mask: np.ndarray,
    *,
    d_min: float,
    d_max: float,
    border_margin: int = 5,
) -> tuple[EndpointPair, ...]:
    if not 0 <= float(d_min) <= float(d_max):
        raise ValueError("endpoint distance bounds are invalid")
    graph = extract_trace_graph(skeletonize_mask(np.asarray(visible_mask, dtype=bool)), border_margin=border_margin)
    endpoints = [
        tuple(int(value) for value in point)
        for point, truncated in zip(graph.endpoints, graph.endpoint_border_truncated)
        if not truncated
    ]
    pairs = []
    for index, first in enumerate(endpoints):
        for second in endpoints[index + 1 :]:
            distance = math.dist(first, second)
            if float(d_min) <= distance <= float(d_max):
                pairs.append(EndpointPair(first, second, float(distance)))
    return tuple(sorted(pairs, key=lambda pair: (pair.distance, pair.first, pair.second)))


def widest_path(
    relation: np.ndarray,
    start: Pixel,
    goal: Pixel,
    *,
    offsets: Iterable[tuple[int, int]] = LOCAL8_OFFSETS,
) -> tuple[float, tuple[Pixel, ...]]:
    """Maximum bottleneck path, with shortest-path deterministic tie breaking."""

    offset_list = tuple((int(dx), int(dy)) for dx, dy in offsets)
    edges = np.asarray(relation, dtype=np.float32)
    if edges.ndim != 3 or edges.shape[0] != len(offset_list):
        raise ValueError("relation must be KxHxW")
    if not np.isfinite(edges).all() or np.any((edges < 0) | (edges > 1)):
        raise ValueError("relation must be finite in [0, 1]")
    height, width = edges.shape[1:]
    for point in (start, goal):
        if not (0 <= point[0] < height and 0 <= point[1] < width):
            raise ValueError("path endpoint outside relation grid")
    best: dict[Pixel, tuple[float, int]] = {start: (1.0, 0)}
    parent: dict[Pixel, Pixel] = {}
    queue: list[tuple[float, int, int, int]] = [(-1.0, 0, start[0], start[1])]
    while queue:
        negative_score, length, y, x = heapq.heappop(queue)
        point = (y, x)
        score = -negative_score
        if best.get(point) != (score, length):
            continue
        if point == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return float(score), tuple(reversed(path))
        for channel, (dx, dy) in enumerate(offset_list):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            edge = float(edges[channel, y, x])
            if edge <= 0.0:
                continue
            candidate = (min(score, edge), length + 1)
            previous = best.get((ny, nx), (-1.0, 2**31 - 1))
            if candidate[0] > previous[0] or (candidate[0] == previous[0] and candidate[1] < previous[1]):
                best[(ny, nx)] = candidate
                parent[(ny, nx)] = point
                heapq.heappush(queue, (-candidate[0], candidate[1], ny, nx))
    return 0.0, ()


def rasterize_path(path: Iterable[Pixel], shape: tuple[int, int], *, radius: int) -> np.ndarray:
    if int(radius) < 0:
        raise ValueError("path radius must be nonnegative")
    centerline = np.zeros(shape, dtype=bool)
    for y, x in path:
        centerline[int(y), int(x)] = True
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    footprint = xx * xx + yy * yy <= int(radius) ** 2
    return binary_dilation(centerline, structure=footprint)

