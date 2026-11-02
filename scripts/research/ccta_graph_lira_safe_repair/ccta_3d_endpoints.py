"""Unlabelled 3D endpoint/candidate extraction from a *predicted* vessel mask.

This module does not read ground truth, patient annotations, CT28 scene plans or
model predictions. The five-endpoint windows match K10/15 input dimensionality
but are not claimed to reproduce its controlled-scene sampling distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import combinations, product
from typing import Sequence

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)
OFFSETS = tuple(p for p in product((-1, 0, 1), repeat=3) if p != (0, 0, 0))
LPS_SIGN = np.array([-1.0, -1.0, 1.0])


@dataclass(frozen=True)
class Endpoint:
    endpoint_id: str
    component: int
    voxel: tuple[int, int, int]
    position_lps: tuple[float, float, float]
    tangent_lps: tuple[float, float, float]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    kind: str
    endpoint_ids: tuple[str, ...]


@dataclass(frozen=True)
class Scene:
    scene_id: str
    endpoint_ids: tuple[str, ...]
    candidates: tuple[Candidate, ...]


def _world_lps(voxel: Sequence[float], affine: np.ndarray) -> np.ndarray:
    return (affine[:3, :3] @ np.asarray(voxel, dtype=float) + affine[:3, 3]) * LPS_SIGN


def _mask_bbox(mask: np.ndarray, pad: int = 2) -> tuple[slice, slice, slice]:
    if not mask.any():
        return (slice(0, 0),) * 3
    axes = [
        np.flatnonzero(np.any(mask, axis=tuple(i for i in range(3) if i != axis)))
        for axis in range(3)
    ]
    return tuple(slice(max(0, int(a[0]) - pad),
                       min(mask.shape[i], int(a[-1]) + pad + 1))
                 for i, a in enumerate(axes))


def _endpoint_direction(
    start: tuple[int, int, int],
    skeleton: np.ndarray,
    affine: np.ndarray,
    trace_mm: float,
) -> np.ndarray | None:
    """Outward tangent: far interior skeleton point -> endpoint, in LPS mm."""
    spacing = {
        offset: float(np.linalg.norm(affine[:3, :3] @ np.array(offset)))
        for offset in OFFSETS
    }
    visited = {start: 0.0}
    frontier = [(0.0, start)]
    best = None
    best_distance = 0.0
    start_lps = _world_lps(start, affine)
    shape = skeleton.shape
    while frontier:
        distance, point = heappop(frontier)
        if distance > visited[point] + 1e-9:
            continue
        if distance > best_distance:
            current = _world_lps(point, affine)
            delta = start_lps - current
            if np.linalg.norm(delta) >= 0.35:
                best_distance, best = distance, delta
        for offset in OFFSETS:
            neighbor = tuple(point[d] + offset[d] for d in range(3))
            if any(neighbor[d] < 0 or neighbor[d] >= shape[d] for d in range(3)):
                continue
            if not skeleton[neighbor]:
                continue
            proposal = distance + spacing[offset]
            if proposal > trace_mm + 1e-9 or proposal >= visited.get(neighbor, float("inf")):
                continue
            visited[neighbor] = proposal
            heappush(frontier, (proposal, neighbor))
    if best is None:
        return None
    norm = np.linalg.norm(best)
    return best / norm if norm > 1e-8 else None


def extract_endpoints(
    mask: np.ndarray,
    affine: np.ndarray,
    *,
    min_component_voxels: int = 12,
    trace_mm: float = 3.0,
    max_endpoints: int = 256,
) -> tuple[list[Endpoint], dict]:
    if mask.ndim != 3 or mask.dtype != bool:
        raise ValueError("Expected 3D boolean predicted mask")
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Expected finite 4x4 affine")
    if not np.allclose(affine[3], [0, 0, 0, 1]):
        raise ValueError("Expected homogeneous affine")
    if min_component_voxels < 2 or trace_mm <= 0 or max_endpoints < 2:
        raise ValueError("Invalid endpoint extraction limits")
    roi = _mask_bbox(mask)
    if not mask.any():
        return [], {"components": 0, "skeleton_voxels": 0,
                    "raw_endpoints": 0, "filtered_endpoints": 0}
    crop = mask[roi]
    labels, n_components = ndimage.label(crop, structure=CONNECTIVITY)
    sizes = np.bincount(labels.ravel(), minlength=n_components + 1)
    skeleton = np.asarray(skeletonize(crop), dtype=bool)
    if not np.any(skeleton):
        return [], {"components": n_components, "skeleton_voxels": 0,
                    "raw_endpoints": 0, "filtered_endpoints": 0}
    neighbors = ndimage.convolve(
        skeleton.astype(np.uint8), CONNECTIVITY, mode="constant", cval=0
    ).astype(np.int16) - skeleton.astype(np.int16)
    raw = np.argwhere(skeleton & (neighbors == 1))
    if len(raw) > max_endpoints * 8:
        raise ValueError("Excessive skeleton endpoint density; inspect input mask")
    offset = np.array([part.start for part in roi], dtype=np.int64)
    local_affine = np.asarray(affine, dtype=float).copy()
    local_affine[:3, 3] += local_affine[:3, :3] @ offset
    extracted = []
    for position in raw:
        voxel = tuple(int(c) for c in position)
        component = int(labels[voxel])
        if component == 0 or sizes[component] < min_component_voxels:
            continue
        tangent = _endpoint_direction(voxel, skeleton, local_affine, trace_mm)
        if tangent is None:
            continue
        global_voxel = tuple(int(c) for c in position + offset)
        location = _world_lps(voxel, local_affine)
        extracted.append((component, global_voxel, location, tangent))
    extracted.sort(key=lambda item: (item[0], item[1]))
    if len(extracted) > max_endpoints:
        raise ValueError("Too many endpoints for complete runtime; no silent truncation")
    endpoints = [
        Endpoint(
            endpoint_id=f"E{i:04d}",
            component=component,
            voxel=voxel,
            position_lps=tuple(float(c) for c in location),
            tangent_lps=tuple(float(c) for c in tangent),
        )
        for i, (component, voxel, location, tangent) in enumerate(extracted)
    ]
    return endpoints, {
        "components": int(n_components),
        "skeleton_voxels": int(skeleton.sum()),
        "raw_endpoints": int(len(raw)),
        "filtered_endpoints": len(endpoints),
        "min_component_voxels": min_component_voxels,
        "trace_mm": trace_mm,
        "anatomical_source": "predicted mask only",
    }


def build_candidate_scenes(
    endpoints: Sequence[Endpoint],
    *,
    patient_label: str,
    pair_limit_mm: float = 12.0,
    junction_span_mm: float = 16.0,
    max_scenes: int = 128,
) -> list[Scene]:
    """Build deterministic five-endpoint local windows with no truth labels.

    Distance envelopes match CT28; scene selection does not. A global endpoint
    may appear in several windows: downstream global scheduling handles this.
    """
    if len({e.endpoint_id for e in endpoints}) != len(endpoints):
        raise ValueError("Duplicate endpoint ID")
    if not np.isfinite([pair_limit_mm, junction_span_mm]).all() or min(
        pair_limit_mm, junction_span_mm
    ) <= 0 or max_scenes < 1:
        raise ValueError("Invalid candidate limits")
    if len(endpoints) < 5:
        return []
    xyz = np.asarray([e.position_lps for e in endpoints])
    labels = np.asarray([e.component for e in endpoints])
    distances = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
    windows = set()
    for i in range(len(endpoints)):
        if not any(j != i and labels[i] != labels[j] and
                   distances[i, j] <= pair_limit_mm for j in range(len(endpoints))):
            continue
        ranked = sorted(range(len(endpoints)), key=lambda j: (distances[i, j], j))
        windows.add(tuple(sorted(ranked[:5])))
    if len(windows) > max_scenes:
        raise ValueError("Too many candidate windows; no silent truncation")
    output = []
    for window_index, window in enumerate(sorted(windows)):
        members = [endpoints[index] for index in window]
        candidates = []
        for size, kind, max_distance in ((2, "PAIR", pair_limit_mm),
                                         (3, "JUNCTION", junction_span_mm)):
            for subset in combinations(window, size):
                if len(set(int(labels[index]) for index in subset)) != size:
                    continue
                spread = max(distances[x, y] for x, y in combinations(subset, 2))
                if spread > max_distance:
                    continue
                ids = tuple(sorted(endpoints[index].endpoint_id for index in subset))
                candidates.append(Candidate(
                    candidate_id=kind + ":" + "+".join(ids),
                    kind=kind,
                    endpoint_ids=ids,
                ))
        if not candidates:
            continue
        scene_id = f"REAL3D:{patient_label}:W{window_index:04d}"
        output.append(Scene(scene_id, tuple(e.endpoint_id for e in members),
                            tuple(candidates)))
    return output
