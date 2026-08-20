"""Eight-connected skeleton graph and trace-segment extraction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .geometry import axial_distance


Pixel = tuple[int, int]
OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))


@dataclass(frozen=True)
class TraceSegment:
    trace_id: int
    pixels: tuple[Pixel, ...]
    start_type: str
    end_type: str
    start_node: int | None
    end_node: int | None
    start_border_truncated: bool = False
    end_border_truncated: bool = False

    @property
    def pixel_length(self) -> float:
        return float(
            sum(math.hypot(y1 - y0, x1 - x0) for (y0, x0), (y1, x1) in zip(self.pixels, self.pixels[1:]))
        )

    @property
    def chord_length(self) -> float:
        if len(self.pixels) < 2:
            return 0.0
        y0, x0 = self.pixels[0]
        y1, x1 = self.pixels[-1]
        return math.hypot(y1 - y0, x1 - x0)


@dataclass(frozen=True)
class TraceGraph:
    skeleton: np.ndarray
    degrees: dict[Pixel, int]
    endpoints: tuple[Pixel, ...]
    junctions: tuple[tuple[Pixel, ...], ...]
    segments: tuple[TraceSegment, ...]
    endpoint_border_truncated: tuple[bool, ...]
    junction_border_truncated: tuple[bool, ...]


@dataclass(frozen=True)
class BranchPair:
    segment_a: int
    segment_b: int
    axial_error: float


def _neighbors(pixel: Pixel, pixels: set[Pixel]) -> tuple[Pixel, ...]:
    y, x = pixel
    return tuple(sorted((y + dy, x + dx) for dy, dx in OFFSETS if (y + dy, x + dx) in pixels))


def _components(pixels: set[Pixel]) -> list[set[Pixel]]:
    remaining = set(pixels)
    components = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component = {start}
        remaining.remove(start)
        while stack:
            current = stack.pop()
            for neighbor in _neighbors(current, pixels):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _edge(a: Pixel, b: Pixel) -> tuple[Pixel, Pixel]:
    return (a, b) if a < b else (b, a)


def _near_border(pixel: Pixel, shape: tuple[int, int], margin: int) -> bool:
    y, x = pixel
    height, width = shape
    return bool(margin > 0 and (y < margin or x < margin or y >= height - margin or x >= width - margin))


def extract_trace_graph(skeleton: np.ndarray, *, border_margin: int = 5) -> TraceGraph:
    array = np.asarray(skeleton, dtype=bool)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D skeleton, got {array.shape}")
    if border_margin < 0:
        raise ValueError("border_margin must be nonnegative")
    pixels = {tuple(point) for point in np.argwhere(array)}
    degrees = {pixel: len(_neighbors(pixel, pixels)) for pixel in pixels}
    endpoints = tuple(sorted(pixel for pixel, degree in degrees.items() if degree == 1))
    junction_components = [tuple(sorted(component)) for component in _components({p for p, d in degrees.items() if d >= 3})]
    junctions = tuple(sorted(junction_components, key=lambda component: component[0]))
    endpoint_border_truncated = tuple(_near_border(point, array.shape, border_margin) for point in endpoints)
    junction_border_truncated = tuple(
        any(_near_border(point, array.shape, border_margin) for point in component) for component in junctions
    )

    node_components: list[set[Pixel]] = [{endpoint} for endpoint in endpoints] + [set(component) for component in junctions]
    node_lookup = {pixel: node_id for node_id, component in enumerate(node_components) for pixel in component}
    node_types = ["endpoint"] * len(endpoints) + ["junction"] * len(junctions)
    node_border_truncated = list(endpoint_border_truncated) + list(junction_border_truncated)
    visited: set[tuple[Pixel, Pixel]] = set()
    paths: list[tuple[list[Pixel], int | None, int | None]] = []

    for start_node, component in enumerate(node_components):
        for start in sorted(component):
            for neighbor in _neighbors(start, pixels):
                if node_lookup.get(neighbor) == start_node:
                    visited.add(_edge(start, neighbor))
                    continue
                if _edge(start, neighbor) in visited:
                    continue
                path = [start, neighbor]
                visited.add(_edge(start, neighbor))
                previous, current = start, neighbor
                while current not in node_lookup:
                    candidates = [item for item in _neighbors(current, pixels) if item != previous]
                    if not candidates:
                        break
                    next_pixel = candidates[0]
                    visited.add(_edge(current, next_pixel))
                    path.append(next_pixel)
                    previous, current = current, next_pixel
                paths.append((path, start_node, node_lookup.get(current)))

    # Preserve closed components with no endpoints or junctions.
    for start in sorted(pixels):
        for neighbor in _neighbors(start, pixels):
            if _edge(start, neighbor) in visited:
                continue
            path = [start, neighbor]
            visited.add(_edge(start, neighbor))
            previous, current = start, neighbor
            while True:
                candidates = [item for item in _neighbors(current, pixels) if item != previous]
                unvisited = [item for item in candidates if _edge(current, item) not in visited]
                if not unvisited:
                    break
                next_pixel = unvisited[0]
                visited.add(_edge(current, next_pixel))
                path.append(next_pixel)
                previous, current = current, next_pixel
                if current == start:
                    break
            paths.append((path, None, None))

    segments = []
    for trace_id, (path, start_node, end_node) in enumerate(paths):
        segments.append(
            TraceSegment(
                trace_id=trace_id,
                pixels=tuple(path),
                start_type=node_types[start_node] if start_node is not None else "loop",
                end_type=node_types[end_node] if end_node is not None else "loop",
                start_node=start_node,
                end_node=end_node,
                start_border_truncated=node_border_truncated[start_node] if start_node is not None else False,
                end_border_truncated=node_border_truncated[end_node] if end_node is not None else False,
            )
        )
    return TraceGraph(
        array,
        degrees,
        endpoints,
        junctions,
        tuple(segments),
        endpoint_border_truncated,
        junction_border_truncated,
    )


def _line_pixels(start: Pixel, end: Pixel) -> tuple[Pixel, ...]:
    """Integer Bresenham bridge including both endpoints."""
    y0, x0 = start
    y1, x1 = end
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx + dy
    output = []
    while True:
        output.append((y0, x0))
        if (y0, x0) == (y1, x1):
            return tuple(output)
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


def _endpoint_tangent(path: tuple[Pixel, ...], *, at_end: bool, tangent_pixels: int) -> float:
    oriented = path if at_end else tuple(reversed(path))
    index = max(0, len(oriented) - 1 - min(max(tangent_pixels, 1), len(oriented) - 1))
    y0, x0 = oriented[index]
    y1, x1 = oriented[-1]
    return math.atan2(y1 - y0, x1 - x0)


def merge_trace_gaps(
    segments: Iterable[TraceSegment],
    *,
    max_gap_px: float,
    max_axial_angle: float = math.pi / 12,
    tangent_pixels: int = 5,
) -> tuple[TraceSegment, ...]:
    """Greedily bridge validation-approved, collinear endpoint gaps."""
    if max_gap_px < 0 or not 0 <= max_axial_angle <= math.pi / 2:
        raise ValueError("Invalid trace-gap merge threshold")
    active = list(segments)
    while True:
        candidates: list[tuple[float, float, int, int, tuple[Pixel, ...], tuple[Pixel, ...]]] = []
        for first_index, first in enumerate(active):
            for second_index in range(first_index + 1, len(active)):
                second = active[second_index]
                for first_path in (first.pixels, tuple(reversed(first.pixels))):
                    for second_path in (second.pixels, tuple(reversed(second.pixels))):
                        first_point, second_point = first_path[-1], second_path[0]
                        distance = math.dist(first_point, second_point)
                        if distance > max_gap_px:
                            continue
                        connection = math.atan2(second_point[0] - first_point[0], second_point[1] - first_point[1])
                        first_tangent = _endpoint_tangent(first_path, at_end=True, tangent_pixels=tangent_pixels)
                        second_tangent = _endpoint_tangent(second_path, at_end=False, tangent_pixels=tangent_pixels)
                        error = max(
                            float(axial_distance(first_tangent, connection)),
                            float(axial_distance(second_tangent, connection)),
                            float(axial_distance(first_tangent, second_tangent)),
                        )
                        if error <= max_axial_angle:
                            candidates.append(
                                (distance, error, first_index, second_index, first_path, second_path)
                            )
        if not candidates:
            break
        _, _, first_index, second_index, first_path, second_path = min(candidates, key=lambda item: item[:4])
        first, second = active[first_index], active[second_index]
        bridge = _line_pixels(first_path[-1], second_path[0])
        merged_pixels = first_path + bridge[1:-1] + second_path
        first_reversed = first_path[0] != first.pixels[0]
        second_reversed = second_path[0] != second.pixels[0]
        merged = TraceSegment(
            trace_id=min(first.trace_id, second.trace_id),
            pixels=merged_pixels,
            start_type=first.end_type if first_reversed else first.start_type,
            end_type=second.start_type if second_reversed else second.end_type,
            start_node=first.end_node if first_reversed else first.start_node,
            end_node=second.start_node if second_reversed else second.end_node,
            start_border_truncated=first.end_border_truncated if first_reversed else first.start_border_truncated,
            end_border_truncated=second.start_border_truncated if second_reversed else second.end_border_truncated,
        )
        active = [item for index, item in enumerate(active) if index not in (first_index, second_index)] + [merged]
    return tuple(
        TraceSegment(
            trace_id=index,
            pixels=segment.pixels,
            start_type=segment.start_type,
            end_type=segment.end_type,
            start_node=segment.start_node,
            end_node=segment.end_node,
            start_border_truncated=segment.start_border_truncated,
            end_border_truncated=segment.end_border_truncated,
        )
        for index, segment in enumerate(sorted(active, key=lambda item: (item.pixels[0], item.pixels[-1])))
    )


def _branch_angle(segment: TraceSegment, junction_node: int, tangent_pixels: int) -> float:
    if segment.start_node == junction_node:
        path = segment.pixels
    elif segment.end_node == junction_node:
        path = tuple(reversed(segment.pixels))
    else:
        raise ValueError(f"Trace {segment.trace_id} is not incident on junction node {junction_node}")
    point_index = min(max(1, tangent_pixels), len(path) - 1)
    y0, x0 = path[0]
    y1, x1 = path[point_index]
    return math.atan2(y1 - y0, x1 - x0)


def pair_junction_branches(
    graph: TraceGraph,
    *,
    junction_index: int,
    tangent_pixels: int = 5,
    max_axial_angle: float = math.pi / 6,
) -> tuple[BranchPair, ...]:
    junction_node = len(graph.endpoints) + junction_index
    incident = [
        segment for segment in graph.segments if segment.start_node == junction_node or segment.end_node == junction_node
    ]
    candidates = []
    for index, first in enumerate(incident):
        first_angle = _branch_angle(first, junction_node, tangent_pixels)
        for second in incident[index + 1 :]:
            second_angle = _branch_angle(second, junction_node, tangent_pixels)
            error = float(axial_distance(first_angle, second_angle))
            candidates.append((error, first.trace_id, second.trace_id))
    used: set[int] = set()
    pairs = []
    for error, first_id, second_id in sorted(candidates):
        if error > max_axial_angle or first_id in used or second_id in used:
            continue
        used.update((first_id, second_id))
        pairs.append(BranchPair(first_id, second_id, error))
    return tuple(pairs)
