"""Deterministic domain-restricted widest path with an exhaustive tiny-graph oracle."""

from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np

from models.anza2.affinity import LOCAL8_OFFSETS
from .graph import restrict_relation_to_domain


Pixel = tuple[int, int]


def domain_restricted_widest_path(
    relation: np.ndarray,
    start: Pixel,
    goal: Pixel,
    *,
    domain: np.ndarray,
    offsets: Iterable[tuple[int, int]] = LOCAL8_OFFSETS,
) -> tuple[float, tuple[Pixel, ...]]:
    offset_list = tuple((int(dx), int(dy)) for dx, dy in offsets)
    mask = np.asarray(domain, dtype=bool)
    if not mask[start] or not mask[goal]:
        return 0.0, ()
    edges = restrict_relation_to_domain(relation, mask, offset_list)
    height, width = mask.shape
    initial_path = (start,)
    best: dict[Pixel, tuple[float, int, tuple[Pixel, ...]]] = {start: (1.0, 0, initial_path)}
    queue: list[tuple[float, int, tuple[Pixel, ...], Pixel]] = [(-1.0, 0, initial_path, start)]
    while queue:
        negative_score, length, path, point = heapq.heappop(queue)
        score = -negative_score
        if best.get(point) != (score, length, path):
            continue
        if point == goal:
            return float(score), path
        y, x = point
        neighbors = []
        for channel, (dx, dy) in enumerate(offset_list):
            neighbor = (y + dy, x + dx)
            if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width:
                neighbors.append((neighbor, float(edges[channel, y, x])))
        for neighbor, edge in sorted(neighbors):
            if edge <= 0 or neighbor in path:
                continue
            candidate = (min(score, edge), length + 1, path + (neighbor,))
            previous = best.get(neighbor)
            better = previous is None or candidate[0] > previous[0]
            if previous is not None and candidate[0] == previous[0]:
                better = candidate[1] < previous[1] or (
                    candidate[1] == previous[1] and candidate[2] < previous[2]
                )
            if better:
                best[neighbor] = candidate
                heapq.heappush(queue, (-candidate[0], candidate[1], candidate[2], neighbor))
    return 0.0, ()


def exhaustive_widest_path(
    relation: np.ndarray,
    start: Pixel,
    goal: Pixel,
    *,
    domain: np.ndarray,
    offsets: Iterable[tuple[int, int]] = LOCAL8_OFFSETS,
) -> tuple[float, tuple[Pixel, ...]]:
    """Exhaustive simple-path oracle for very small unit-test graphs."""

    offset_list = tuple((int(dx), int(dy)) for dx, dy in offsets)
    edges = restrict_relation_to_domain(relation, domain, offset_list)
    height, width = edges.shape[1:]
    best_score = 0.0
    best_path: tuple[Pixel, ...] = ()

    def visit(point: Pixel, score: float, path: tuple[Pixel, ...], seen: frozenset[Pixel]) -> None:
        nonlocal best_score, best_path
        if point == goal:
            better = score > best_score
            if score == best_score:
                better = not best_path or len(path) < len(best_path) or (
                    len(path) == len(best_path) and path < best_path
                )
            if better:
                best_score, best_path = float(score), path
            return
        y, x = point
        for channel, (dx, dy) in enumerate(offset_list):
            neighbor = (y + dy, x + dx)
            if not (0 <= neighbor[0] < height and 0 <= neighbor[1] < width):
                continue
            edge = float(edges[channel, y, x])
            if edge <= 0 or neighbor in seen:
                continue
            visit(neighbor, min(score, edge), path + (neighbor,), seen | {neighbor})

    visit(start, 1.0, (start,), frozenset({start}))
    return best_score, best_path
