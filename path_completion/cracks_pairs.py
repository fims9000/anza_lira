"""Deterministic real-domain endpoint pairs from CRACKS crowd-train traces."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

import numpy as np
from scipy.ndimage import label, map_coordinates

from trace_extraction.graph import extract_trace_graph
from trace_extraction.skeleton import skeletonize_mask


MATCH_TOLERANCES = np.asarray([6.0, 0.35, 0.08, 0.15], dtype=np.float64)


@dataclass(frozen=True)
class RealPairCandidate:
    first: tuple[int, int]
    second: tuple[int, int]
    bridge_pixels: tuple[tuple[int, int], ...]
    distance: float
    tangent_error: float
    contrast_difference: float
    depth: float
    label: int
    source_kind: str

    @property
    def descriptor(self) -> np.ndarray:
        return np.asarray(
            [self.distance, self.tangent_error, self.contrast_difference, self.depth],
            dtype=np.float64,
        )


def split_sections(
    section_ids: Iterable[int], *, salt: str = "cracks-real-pair-section-split-v1"
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    train, validation = [], []
    for section_id in sorted(int(value) for value in section_ids):
        digest = hashlib.sha256(f"{salt}:{section_id}".encode()).digest()
        (validation if int.from_bytes(digest[:8], "big") % 3 == 0 else train).append(section_id)
    if set(train) & set(validation) or not train or not validation:
        raise AssertionError("Real-pair section split is invalid")
    return tuple(train), tuple(validation)


def _line_pixels(first: tuple[int, int], second: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    y0, x0 = first
    y1, x1 = second
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx + dy
    result = []
    while True:
        result.append((y0, x0))
        if (y0, x0) == (y1, x1):
            return tuple(result)
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


def _axial_error(first: float, second: float) -> float:
    delta = abs(first - second) % math.pi
    return float(min(delta, math.pi - delta))


def _local_contrast(gray: np.ndarray, point: tuple[int, int], radius: int = 4) -> float:
    y, x = point
    patch = gray[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1]
    return float(np.std(patch))


def _candidate(
    first: tuple[int, int],
    second: tuple[int, int],
    bridge: tuple[tuple[int, int], ...],
    first_tangent: float,
    second_tangent: float,
    gray: np.ndarray,
    *,
    label_value: int,
    source_kind: str,
) -> RealPairCandidate:
    distance = float(math.dist(first, second))
    connection = math.atan2(second[0] - first[0], second[1] - first[1])
    tangent_error = max(
        _axial_error(first_tangent, connection),
        _axial_error(second_tangent, connection),
        _axial_error(first_tangent, second_tangent),
    )
    return RealPairCandidate(
        first,
        second,
        bridge,
        distance,
        tangent_error,
        abs(_local_contrast(gray, first) - _local_contrast(gray, second)),
        float((first[0] + second[0]) / (2.0 * gray.shape[0])),
        int(label_value),
        source_kind,
    )


def _positive_candidates(
    skeleton: np.ndarray,
    gray: np.ndarray,
    *,
    gap_lengths: tuple[int, ...] = (8, 12, 16),
) -> list[RealPairCandidate]:
    graph = extract_trace_graph(skeleton, border_margin=8)
    output: list[RealPairCandidate] = []
    for segment in graph.segments:
        path = segment.pixels
        if len(path) < max(gap_lengths) + 12:
            continue
        gap = gap_lengths[segment.trace_id % len(gap_lengths)]
        center = len(path) // 2
        start = center - gap // 2
        end = start + gap
        if start < 5 or end + 5 >= len(path):
            continue
        first, second = path[start], path[end]
        y0, x0 = path[start - 5]
        y1, x1 = first
        first_tangent = math.atan2(y1 - y0, x1 - x0)
        y2, x2 = second
        y3, x3 = path[end + 5]
        second_tangent = math.atan2(y3 - y2, x3 - x2)
        output.append(
            _candidate(
                first,
                second,
                tuple(path[start : end + 1]),
                first_tangent,
                second_tangent,
                gray,
                label_value=1,
                source_kind="same_trace_internal_gap",
            )
        )
    return output


def _endpoint_records(skeleton: np.ndarray) -> list[tuple[tuple[int, int], float, int]]:
    components, _ = label(skeleton, structure=np.ones((3, 3), dtype=np.uint8))
    graph = extract_trace_graph(skeleton, border_margin=8)
    records: list[tuple[tuple[int, int], float, int]] = []
    for segment in graph.segments:
        if len(segment.pixels) < 6:
            continue
        for path in (segment.pixels, tuple(reversed(segment.pixels))):
            point = path[0]
            if point not in graph.endpoints:
                continue
            y0, x0 = point
            y1, x1 = path[min(5, len(path) - 1)]
            tangent = math.atan2(y1 - y0, x1 - x0)
            records.append((point, tangent, int(components[point])))
    unique = {(point, component): (point, tangent, component) for point, tangent, component in records}
    return list(unique.values())


def _negative_candidates(
    skeleton: np.ndarray,
    gray: np.ndarray,
    *,
    d_min: float = 6.0,
    d_max: float = 24.0,
) -> list[RealPairCandidate]:
    records = _endpoint_records(skeleton)
    output = []
    for index, (first, first_tangent, first_component) in enumerate(records):
        for second, second_tangent, second_component in records[index + 1 :]:
            if first_component == second_component:
                continue
            distance = math.dist(first, second)
            if not d_min <= distance <= d_max:
                continue
            candidate = _candidate(
                first,
                second,
                _line_pixels(first, second),
                first_tangent,
                second_tangent,
                gray,
                label_value=0,
                source_kind="different_connected_traces",
            )
            if candidate.tangent_error <= math.pi / 3:
                output.append(candidate)
    return output


def matched_section_pairs(
    positive_mask: np.ndarray,
    image_chw: np.ndarray,
    *,
    max_pairs: int = 2,
) -> list[tuple[RealPairCandidate, RealPairCandidate]]:
    """Match positive gaps to different-component negatives by frozen descriptors."""
    mask = np.asarray(positive_mask, dtype=bool)
    image = np.asarray(image_chw, dtype=np.float32)
    gray = image.mean(axis=0)
    skeleton = skeletonize_mask(mask)
    positives = _positive_candidates(skeleton, gray)
    negatives = _negative_candidates(skeleton, gray)
    if not positives or not negatives:
        return []
    pairs = []
    available = list(negatives)
    for positive in sorted(positives, key=lambda item: (item.first, item.second)):
        eligible = [
            item for item in available
            if np.all(np.abs(item.descriptor - positive.descriptor) <= MATCH_TOLERANCES)
        ]
        if not eligible:
            continue
        negative = min(
            eligible,
            key=lambda item: (
                float(np.linalg.norm((item.descriptor - positive.descriptor) / MATCH_TOLERANCES)),
                item.first,
                item.second,
            ),
        )
        pairs.append((positive, negative))
        available.remove(negative)
        if len(pairs) >= int(max_pairs) or not available:
            break
    return pairs


def _erased_probability(
    probability: np.ndarray,
    bridge: tuple[tuple[int, int], ...],
    *,
    radius: int = 2,
) -> np.ndarray:
    result = np.asarray(probability, dtype=np.float32).copy()
    height, width = result.shape
    for y, x in bridge[1:-1]:
        result[max(0, y - radius) : min(height, y + radius + 1), max(0, x - radius) : min(width, x + radius + 1)] = 0.0
    return result


def oriented_real_pair_crop(
    fields: dict[str, np.ndarray],
    candidate: RealPairCandidate,
    *,
    crop_hw: tuple[int, int] = (33, 49),
    cross_extent: float = 16.0,
    longitudinal_padding: float = 8.0,
) -> np.ndarray:
    """Create RGB/base/marker/ANZA-geometry crop for one unordered pair."""
    height, width = crop_hw
    first = np.asarray(candidate.first, dtype=np.float64)
    second = np.asarray(candidate.second, dtype=np.float64)
    vector = second - first
    distance = float(np.linalg.norm(vector))
    if distance <= 0:
        raise ValueError("Pair endpoints must differ")
    along = vector / distance
    across = np.asarray([-along[1], along[0]])
    midpoint = 0.5 * (first + second)
    longitudinal_extent = 0.5 * distance + longitudinal_padding
    longitudinal = np.linspace(-longitudinal_extent, longitudinal_extent, width)
    transverse = np.linspace(-cross_extent, cross_extent, height)
    grid_longitudinal, grid_transverse = np.meshgrid(longitudinal, transverse)
    grid_y = midpoint[0] + along[0] * grid_longitudinal + across[0] * grid_transverse
    grid_x = midpoint[1] + along[1] * grid_longitudinal + across[1] * grid_transverse
    image = np.asarray(fields["image"], dtype=np.float32)
    probability = _erased_probability(fields["base_probability"], candidate.bridge_pixels)
    channels = [map_coordinates(channel, (grid_y, grid_x), order=1, mode="reflect") for channel in image]
    channels.append(map_coordinates(probability, (grid_y, grid_x), order=1, mode="reflect"))
    endpoint_position = distance / (2.0 * longitudinal_extent) * ((width - 1) / 2.0)
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    yy, xx = np.mgrid[:height, :width]
    markers = np.maximum(
        np.exp(-((xx - (center_x - endpoint_position)) ** 2 + (yy - center_y) ** 2) / 4.0),
        np.exp(-((xx - (center_x + endpoint_position)) ** 2 + (yy - center_y) ** 2) / 4.0),
    )
    channels.append(markers.astype(np.float32))
    for name in ("cos2theta", "sin2theta", "anisotropy"):
        channels.append(map_coordinates(fields[name], (grid_y, grid_x), order=1, mode="reflect"))
    output = np.stack(channels).astype(np.float32)
    if output.shape != (8, height, width) or not np.isfinite(output).all():
        raise AssertionError("Invalid CRACKS real-pair crop")
    return output
