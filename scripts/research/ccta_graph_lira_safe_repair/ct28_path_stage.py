#!/usr/bin/env python3
"""CT-only 3D maximum-bottleneck paths for accepted CT28 repairs.

This is a new path protocol, not a reconstruction of historical a(e).
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import product

import numpy as np

from extract_junction_ct_features import estimate_junction_center
from extract_radial_features_z04 import orthobasis, sample_world


OFFSETS = tuple(sorted(offset for offset in product((-1, 0, 1), repeat=3)
                       if offset != (0, 0, 0)))


@dataclass(frozen=True)
class PathConfig:
    corridor_radius_mm: float = 2.0
    center_refine_mm: float = 2.0
    hu_difference_scale: float = 150.0
    vessel_scale_floor: float = 80.0
    background_scale_floor: float = 40.0
    extra_endpoint_clearance_mm: float = 0.75

    def __post_init__(self) -> None:
        if min(self.corridor_radius_mm, self.center_refine_mm,
               self.hu_difference_scale, self.vessel_scale_floor,
               self.background_scale_floor, self.extra_endpoint_clearance_mm) <= 0:
            raise ValueError("path parameters must be positive")


@dataclass
class PathResult:
    status: str
    bottleneck: float
    length_mm: float
    paths_voxel: tuple[tuple[tuple[int, int, int], ...], ...]
    center_lps: tuple[float, float, float] | None
    vessel_hu: float
    background_hu: float
    contrast_hu: float


def lps_to_voxel(points: np.ndarray, inverse_affine: np.ndarray) -> np.ndarray:
    xyz = np.asarray(points, dtype=float).reshape(-1, 3)
    ras = xyz * np.array([-1.0, -1.0, 1.0])
    return (inverse_affine @ np.c_[ras, np.ones(len(ras))].T).T[:, :3]


def voxel_to_lps(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    vox = np.asarray(points, dtype=float).reshape(-1, 3)
    ras = (affine @ np.c_[vox, np.ones(len(vox))].T).T[:, :3]
    return ras * np.array([-1.0, -1.0, 1.0])


def distance_to_segment(points: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    direction = second - first
    denominator = float(direction @ direction)
    if denominator <= 1e-12:
        return np.linalg.norm(points - first, axis=-1)
    fraction = np.clip(np.sum((points - first) * direction, axis=-1) / denominator, 0.0, 1.0)
    return np.linalg.norm(points - first - fraction[..., None] * direction, axis=-1)


def visible_hu_reference(
    ct: np.ndarray, inverse_affine: np.ndarray,
    endpoints: np.ndarray, center: np.ndarray,
) -> tuple[float, float]:
    vessel_points, ring_points = [], []
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    for point in endpoints:
        outward = point - center
        length = float(np.linalg.norm(outward))
        if length < 1e-8:
            raise ValueError("center coincides with an endpoint")
        outward /= length
        u, v = orthobasis(outward.copy())
        visible = point[None, :] + np.array([0.5, 1.0, 1.5, 2.0])[:, None] * outward
        vessel_points.extend(visible)
        for radius in (1.5, 2.5):
            ring_points.extend(visible[None, :, :] + radius * (
                np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
            )[:, None, :])
    vessel_points = np.asarray(vessel_points)
    ring_points = np.asarray(ring_points).reshape(-1, 3)
    sample_vox = lps_to_voxel(np.vstack([vessel_points, ring_points]), inverse_affine)
    if np.any(sample_vox < 0) or np.any(sample_vox > np.asarray(ct.shape[:3]) - 1):
        raise ValueError("visible HU reference exits CT volume")
    vessel = np.clip(sample_world(ct, inverse_affine, vessel_points), -300.0, 1200.0)
    background = np.clip(sample_world(ct, inverse_affine, ring_points), -300.0, 1200.0)
    return float(np.median(vessel)), float(np.median(background))


def local_grid(
    ct: np.ndarray, affine: np.ndarray, endpoints: np.ndarray,
    segments: tuple[tuple[np.ndarray, np.ndarray], ...],
    config: PathConfig,
    forbidden_lps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inverse = np.linalg.inv(affine)
    control = np.vstack([endpoints, *[np.stack(pair) for pair in segments]])
    vox = lps_to_voxel(control, inverse)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    margin = (config.corridor_radius_mm + config.center_refine_mm + 1.0) / spacing
    lower = np.maximum(0, np.floor(vox.min(axis=0) - margin).astype(int))
    upper = np.minimum(np.array(ct.shape[:3]) - 1, np.ceil(vox.max(axis=0) + margin).astype(int))
    if np.any(lower > upper):
        raise ValueError("path ROI is outside CT")
    slices = tuple(slice(int(lo), int(hi) + 1) for lo, hi in zip(lower, upper))
    if int(np.prod(upper - lower + 1)) > 300_000:
        raise ValueError("path ROI exceeds 300000 voxels")
    hu = np.asarray(ct[slices], dtype=np.float32)
    grid = np.indices(hu.shape, dtype=np.float32).reshape(3, -1).T + lower
    world = voxel_to_lps(grid, affine).reshape((*hu.shape, 3))
    mask = np.zeros(hu.shape, dtype=bool)
    for first, second in segments:
        mask |= distance_to_segment(world, first, second) <= config.corridor_radius_mm
    snapped = np.rint(lps_to_voxel(endpoints, inverse)).astype(int)
    if np.any(snapped < 0) or np.any(snapped >= np.asarray(ct.shape[:3])):
        raise ValueError("path endpoint outside CT")
    for point in snapped - lower:
        mask[tuple(point)] = True
    if forbidden_lps is not None and len(forbidden_lps):
        forbidden = np.asarray(forbidden_lps, dtype=float).reshape(-1, 3)
        near_other = (np.linalg.norm(world[..., None, :] - forbidden, axis=-1)
                      <= config.extra_endpoint_clearance_mm).any(axis=-1)
        near_selected = (np.linalg.norm(world[..., None, :] - endpoints, axis=-1)
                         <= config.extra_endpoint_clearance_mm).any(axis=-1)
        mask &= ~(near_other & ~near_selected)
    return hu, mask, lower, world


def voxel_support(hu: np.ndarray, vessel_hu: float, background_hu: float,
                  config: PathConfig) -> np.ndarray:
    contrast = max(vessel_hu - background_hu, 0.0)
    vessel_scale = max(config.vessel_scale_floor, 0.5 * contrast)
    background_scale = max(config.background_scale_floor, 0.25 * contrast)
    vessel_match = np.exp(-0.5 * ((hu - vessel_hu) / vessel_scale) ** 2)
    background_reject = 1.0 / (1.0 + np.exp(np.clip(
        -(hu - background_hu) / background_scale, -60.0, 60.0
    )))
    return np.asarray(vessel_match * background_reject, dtype=np.float64)


def widest_paths(
    hu: np.ndarray, mask: np.ndarray, support: np.ndarray, affine: np.ndarray,
    start: tuple[int, int, int], hu_scale: float,
    goal: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int, int], tuple[int, int, int]]]:
    shape = hu.shape
    best = np.full(shape, -1.0, dtype=np.float64)
    lengths = np.full(shape, np.inf, dtype=np.float64)
    parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    best[start], lengths[start] = 1.0, 0.0
    heap = [(-1.0, 0.0, start)]
    steps = {offset: float(np.linalg.norm(affine[:3, :3] @ offset)) for offset in OFFSETS}
    while heap:
        negative, length, point = heapq.heappop(heap)
        score = -negative
        if score + 1e-12 < best[point] or length > lengths[point] + 1e-9:
            continue
        if point == goal:
            break
        for offset in OFFSETS:
            neighbor = tuple(point[axis] + offset[axis] for axis in range(3))
            if any(neighbor[axis] < 0 or neighbor[axis] >= shape[axis] for axis in range(3)):
                continue
            if not mask[neighbor]:
                continue
            edge = min(float(support[point]), float(support[neighbor])) * float(
                np.exp(-abs(float(hu[point]) - float(hu[neighbor])) / hu_scale)
            )
            candidate = min(score, edge)
            distance = length + steps[offset]
            if candidate > best[neighbor] + 1e-12 or (
                abs(candidate - best[neighbor]) <= 1e-12 and distance < lengths[neighbor] - 1e-9
            ):
                best[neighbor], lengths[neighbor] = candidate, distance
                parent[neighbor] = point
                heapq.heappush(heap, (-candidate, distance, neighbor))
    return best, lengths, parent


def reconstruct(
    parent: dict[tuple[int, int, int], tuple[int, int, int]],
    start: tuple[int, int, int], goal: tuple[int, int, int],
) -> tuple[tuple[int, int, int], ...]:
    path = [goal]
    seen = {goal}
    while path[-1] != start:
        if path[-1] not in parent:
            return ()
        predecessor = parent[path[-1]]
        if predecessor in seen:
            # Numerical support ties can produce a cyclic predecessor chain.
            # Such a chain is not a valid path to this source.
            return ()
        seen.add(predecessor)
        path.append(predecessor)
    return tuple(reversed(path))


def shortest_supported_paths(
    hu: np.ndarray, mask: np.ndarray, support: np.ndarray, affine: np.ndarray,
    start: tuple[int, int, int], hu_scale: float, minimum_edge: float,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int, int], tuple[int, int, int]]]:
    """Shortest physical paths whose every CT edge meets a fixed support floor."""
    lengths = np.full(hu.shape, np.inf, dtype=np.float64)
    bottlenecks = np.full(hu.shape, -1.0, dtype=np.float64)
    parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    lengths[start], bottlenecks[start] = 0.0, 1.0
    heap = [(0.0, -1.0, start)]
    steps = {offset: float(np.linalg.norm(affine[:3, :3] @ offset)) for offset in OFFSETS}
    while heap:
        length, negative, point = heapq.heappop(heap)
        score = -negative
        if length > lengths[point] + 1e-9 or (abs(length - lengths[point]) <= 1e-9 and
                                               score + 1e-12 < bottlenecks[point]):
            continue
        for offset in OFFSETS:
            neighbor = tuple(point[axis] + offset[axis] for axis in range(3))
            if any(neighbor[axis] < 0 or neighbor[axis] >= hu.shape[axis] for axis in range(3)):
                continue
            if not mask[neighbor]:
                continue
            edge = min(float(support[point]), float(support[neighbor])) * float(
                np.exp(-abs(float(hu[point]) - float(hu[neighbor])) / hu_scale))
            if edge + 1e-12 < minimum_edge:
                continue
            distance = length + steps[offset]
            candidate = min(score, edge)
            if distance < lengths[neighbor] - 1e-9 or (
                abs(distance - lengths[neighbor]) <= 1e-9 and
                candidate > bottlenecks[neighbor] + 1e-12
            ):
                lengths[neighbor], bottlenecks[neighbor] = distance, candidate
                parent[neighbor] = point
                heapq.heappush(heap, (distance, -candidate, neighbor))
    return lengths, bottlenecks, parent


def solve_path_with_bottleneck_slack(
    ct: np.ndarray, affine: np.ndarray, endpoints: np.ndarray,
    tangents: np.ndarray, config: PathConfig, slack_fraction: float,
    forbidden_lps: np.ndarray | None = None,
) -> PathResult:
    """Train-only ablation: minimize length while retaining near-optimal CT support."""
    if not 0.0 <= slack_fraction < 1.0:
        raise ValueError("bottleneck slack fraction must lie in [0, 1)")
    baseline = solve_path(ct, affine, endpoints, tangents, config, forbidden_lps=forbidden_lps)
    if slack_fraction == 0.0 or baseline.status != "ok":
        return baseline
    endpoints = np.asarray(endpoints, dtype=float)
    junction = len(endpoints) == 3
    center = estimate_junction_center(endpoints, tangents) if junction else endpoints.mean(axis=0)
    segments = tuple((point, center) for point in endpoints) if junction else ((endpoints[0], endpoints[1]),)
    inverse = np.linalg.inv(affine)
    hu, mask, origin, world = local_grid(ct, affine, endpoints, segments, config,
                                       forbidden_lps=forbidden_lps)
    support = voxel_support(hu, baseline.vessel_hu, baseline.background_hu, config)
    starts = [tuple(np.rint(point).astype(int) - origin)
              for point in lps_to_voxel(endpoints, inverse)]
    threshold = baseline.bottleneck * (1.0 - slack_fraction)
    searches = [shortest_supported_paths(hu, mask, support, affine, start,
                                         config.hu_difference_scale, threshold)
                for start in starts[:3 if junction else 1]]
    if not junction:
        goal = starts[1]
        path = reconstruct(searches[0][2], starts[0], goal)
        if not path:
            return baseline
        return PathResult("ok", float(searches[0][1][goal]), float(searches[0][0][goal]),
                          (tuple(tuple(int(x) for x in np.asarray(point) + origin) for point in path),),
                          None, baseline.vessel_hu, baseline.background_hu, baseline.contrast_hu)
    feasible = mask & (np.linalg.norm(world - center, axis=-1) <= config.center_refine_mm)
    for lengths, _, _ in searches:
        feasible &= np.isfinite(lengths)
    ordered = sorted((tuple(int(x) for x in node) for node in np.argwhere(feasible)
                      if tuple(int(x) for x in node) not in starts),
                     key=lambda node: (sum(float(search[0][node]) for search in searches),
                                       -min(float(search[1][node]) for search in searches), node))
    for node in ordered:
        local_paths = [reconstruct(search[2], start, node)
                       for search, start in zip(searches, starts)]
        if not all(local_paths):
            continue
        sets = [set(path[:-1]) for path in local_paths]
        overlap = set.union(*(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)))
        if any(np.linalg.norm(world[point] - world[node]) > 0.75 for point in overlap):
            continue
        paths = tuple(tuple(tuple(int(x) for x in np.asarray(point) + origin) for point in path)
                      for path in local_paths)
        return PathResult("ok", min(float(search[1][node]) for search in searches),
                          sum(float(search[0][node]) for search in searches), paths,
                          tuple(float(x) for x in world[node]), baseline.vessel_hu,
                          baseline.background_hu, baseline.contrast_hu)
    return baseline


def solve_path(
    ct: np.ndarray, affine: np.ndarray, endpoints: np.ndarray,
    tangents: np.ndarray, config: PathConfig,
    forbidden_lps: np.ndarray | None = None,
) -> PathResult:
    endpoints = np.asarray(endpoints, dtype=float)
    tangents = np.asarray(tangents, dtype=float)
    if endpoints.shape not in {(2, 3), (3, 3)} or tangents.shape != endpoints.shape:
        raise ValueError("only degree-2 PAIR and degree-3 JUNCTION are supported")
    junction = len(endpoints) == 3
    center = estimate_junction_center(endpoints, tangents) if junction else endpoints.mean(axis=0)
    segments = tuple((point, center) for point in endpoints) if junction else ((endpoints[0], endpoints[1]),)
    inverse = np.linalg.inv(affine)
    vessel_hu, background_hu = visible_hu_reference(ct, inverse, endpoints, center)
    hu, mask, origin, world = local_grid(ct, affine, endpoints, segments, config,
                                       forbidden_lps=forbidden_lps)
    support = voxel_support(hu, vessel_hu, background_hu, config)
    starts_global = np.rint(lps_to_voxel(endpoints, inverse)).astype(int)
    starts = [tuple(point - origin) for point in starts_global]
    if len(set(starts)) != len(starts):
        raise ValueError("multiple repair endpoints map to one CT voxel")
    if not junction:
        best, lengths, parent = widest_paths(
            hu, mask, support, affine, starts[0], config.hu_difference_scale, starts[1]
        )
        path = reconstruct(parent, starts[0], starts[1])
        paths = (tuple(tuple(int(x) for x in np.asarray(point) + origin) for point in path),) if path else ()
        return PathResult("ok" if path else "no_path", float(best[starts[1]]) if path else 0.0,
                          float(lengths[starts[1]]) if path else 0.0, paths, None,
                          vessel_hu, background_hu, vessel_hu - background_hu)
    searches = [widest_paths(hu, mask, support, affine, start, config.hu_difference_scale)
                for start in starts]
    feasible = mask & (np.linalg.norm(world - center, axis=-1) <= config.center_refine_mm)
    for scores, _, _ in searches:
        feasible &= scores >= 0.0
    centers = np.argwhere(feasible)
    ordered = sorted((tuple(int(x) for x in node) for node in centers
                      if tuple(int(x) for x in node) not in starts),
                     key=lambda node: (-min(float(search[0][node]) for search in searches),
                                       sum(float(search[1][node]) for search in searches), node))
    for node in ordered:
        local_paths = [reconstruct(search[2], start, node)
                       for search, start in zip(searches, starts)]
        if not all(local_paths):
            continue
        # Three arms may meet inside one voxel-sized neighborhood of the node.
        sets = [set(path[:-1]) for path in local_paths]
        overlap = set.union(*(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)))
        if any(np.linalg.norm(world[point] - world[node]) > 0.75 for point in overlap):
            continue
        paths = tuple(tuple(tuple(int(x) for x in np.asarray(point) + origin) for point in path)
                      for path in local_paths)
        return PathResult("ok", min(float(search[0][node]) for search in searches),
                          sum(float(search[1][node]) for search in searches), paths,
                          tuple(float(x) for x in world[node]), vessel_hu,
                          background_hu, vessel_hu - background_hu)
    return PathResult("no_common_node" if not ordered else "branch_overlap", 0.0, 0.0,
                      (), None, vessel_hu, background_hu, vessel_hu - background_hu)
