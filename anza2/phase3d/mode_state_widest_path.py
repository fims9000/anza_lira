"""Deterministic widest path over ``(pixel, mode)`` states."""

from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np

from models.anza2.affinity import LOCAL8_OFFSETS


State = tuple[int, int, int]  # y, x, mode


def _validate(
    edges: np.ndarray,
    domain: np.ndarray,
    offsets: Iterable[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    values = np.asarray(edges, dtype=np.float32)
    mask = np.asarray(domain, dtype=bool)
    offset_list = tuple((int(dx), int(dy)) for dx, dy in offsets)
    if (0, 0) in offset_list:
        raise ValueError("free intra-pixel mode switching is forbidden")
    if values.ndim != 5 or values.shape[0] != len(offset_list):
        raise ValueError("edges must be CxRxSxHxW")
    if values.shape[1] != values.shape[2] or values.shape[-2:] != mask.shape:
        raise ValueError("mode-state edge/domain shapes are inconsistent")
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("mode-state weights must be finite in [0,1]")
    return values, mask, offset_list


def mode_state_widest_path(
    edges: np.ndarray,
    start_states: Iterable[State],
    goal_states: Iterable[State],
    *,
    domain: np.ndarray,
    offsets: Iterable[tuple[int, int]] = LOCAL8_OFFSETS,
) -> tuple[float, tuple[State, ...]]:
    values, mask, offset_list = _validate(edges, domain, offsets)
    modes, height, width = values.shape[1], values.shape[-2], values.shape[-1]
    starts = tuple(sorted(set(tuple(map(int, state)) for state in start_states)))
    goals = frozenset(tuple(map(int, state)) for state in goal_states)
    if not starts or not goals:
        return 0.0, ()
    for y, x, mode in starts + tuple(goals):
        if not (0 <= y < height and 0 <= x < width and 0 <= mode < modes):
            raise ValueError("state outside graph")
    starts = tuple(state for state in starts if mask[state[0], state[1]])
    goals = frozenset(state for state in goals if mask[state[0], state[1]])
    if not starts or not goals:
        return 0.0, ()

    best: dict[State, tuple[float, int, tuple[State, ...]]] = {}
    queue: list[tuple[float, int, tuple[State, ...], State]] = []
    for state in starts:
        path = (state,); record = (1.0, 0, path)
        best[state] = record; heapq.heappush(queue, (-1.0, 0, path, state))
    while queue:
        negative_score, length, path, state = heapq.heappop(queue)
        score = -negative_score
        if best.get(state) != (score, length, path):
            continue
        if state in goals:
            return float(score), path
        y, x, source_mode = state
        neighbors = []
        for channel, (dx, dy) in enumerate(offset_list):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width and mask[ny, nx]):
                continue
            for destination_mode in range(modes):
                edge = float(values[channel, source_mode, destination_mode, y, x])
                if edge > 0:
                    neighbors.append(((ny, nx, destination_mode), edge))
        for neighbor, edge in sorted(neighbors):
            if neighbor in path:
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


def exhaustive_mode_state_widest_path(
    edges: np.ndarray,
    start_states: Iterable[State],
    goal_states: Iterable[State],
    *,
    domain: np.ndarray,
    offsets: Iterable[tuple[int, int]] = LOCAL8_OFFSETS,
) -> tuple[float, tuple[State, ...]]:
    """Exhaustive simple-state-path reference for tiny tests."""

    values, mask, offset_list = _validate(edges, domain, offsets)
    modes, height, width = values.shape[1], values.shape[-2], values.shape[-1]
    goals = frozenset(tuple(map(int, state)) for state in goal_states)
    best_score = 0.0; best_path: tuple[State, ...] = ()

    def visit(state: State, score: float, path: tuple[State, ...], seen: frozenset[State]) -> None:
        nonlocal best_score, best_path
        if state in goals:
            better = score > best_score or (
                score == best_score and (not best_path or len(path) < len(best_path) or (len(path) == len(best_path) and path < best_path))
            )
            if better:
                best_score, best_path = float(score), path
            return
        y, x, source_mode = state
        for channel, (dx, dy) in enumerate(offset_list):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width and mask[ny, nx]):
                continue
            for destination_mode in range(modes):
                neighbor = (ny, nx, destination_mode)
                edge = float(values[channel, source_mode, destination_mode, y, x])
                if edge > 0 and neighbor not in seen:
                    visit(neighbor, min(score, edge), path + (neighbor,), seen | {neighbor})

    for start in sorted(set(tuple(map(int, state)) for state in start_states)):
        if mask[start[0], start[1]]:
            visit(start, 1.0, (start,), frozenset({start}))
    return best_score, best_path

