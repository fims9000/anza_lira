"""Minimal topology-valid evidence cuts, independent of SBPP success."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label

from lira_graph_cut_v2.protocol import CUT_RADII, PROTOCOL


STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)


def rasterize(points_yx: np.ndarray, shape: tuple[int, int], radius: int) -> np.ndarray:
    points = np.rint(np.asarray(points_yx)).astype(int)
    points[:, 0] = np.clip(points[:, 0], 0, shape[0] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, shape[1] - 1)
    seed = np.zeros(shape, dtype=bool)
    seed[points[:, 0], points[:, 1]] = True
    if radius <= 0:
        return seed
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return binary_dilation(seed, structure=yy * yy + xx * xx <= radius * radius)


def connected(mask: np.ndarray, left_anchor: np.ndarray, right_anchor: np.ndarray) -> bool:
    components, _count = label(np.asarray(mask, dtype=bool), structure=STRUCTURE_8)
    left = set(components[np.asarray(left_anchor, dtype=bool)].tolist()) - {0}
    right = set(components[np.asarray(right_anchor, dtype=bool)].tolist()) - {0}
    return bool(left & right)


def tube_distance(hidden_yx: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    seed = rasterize(hidden_yx, shape, 0)
    return distance_transform_edt(~seed)


@dataclass(frozen=True)
class CutResult:
    status: str
    radius: int | None
    collateral_fraction: float
    pre_connected: bool
    post_connected: bool | None
    left_context_supported: int
    right_context_supported: int
    tube_pixels: int


def minimal_valid_cut(
    probability: np.ndarray,
    hidden_yx: np.ndarray,
    left_anchor_yx: np.ndarray,
    right_anchor_yx: np.ndarray,
    left_context_yx: np.ndarray,
    right_context_yx: np.ndarray,
    other_trace_mask: np.ndarray,
) -> tuple[CutResult, np.ndarray | None, np.ndarray | None]:
    threshold = float(PROTOCOL["treatment"]["validation_threshold"])
    support = np.asarray(probability) >= threshold
    left_anchor = rasterize(left_anchor_yx, support.shape, int(PROTOCOL["treatment"]["anchor_raster_radius_px"]))
    right_anchor = rasterize(right_anchor_yx, support.shape, int(PROTOCOL["treatment"]["anchor_raster_radius_px"]))
    if not (support & left_anchor).any() or not (support & right_anchor).any():
        return CutResult("INELIGIBLE_ANCHOR_SUPPORT", None, 0.0, False, None, 0, 0, 0), None, None
    pre_connected = connected(support, left_anchor, right_anchor)
    if not pre_connected:
        return CutResult("INELIGIBLE_PRE_DISCONNECTED", None, 0.0, False, None, 0, 0, 0), None, None
    distances = tube_distance(hidden_yx, support.shape)
    selected_radius = None
    selected_tube = None
    selected_support = None
    for radius in CUT_RADII:
        tube = distances <= float(radius)
        cut_support = support & ~tube
        if not connected(cut_support, left_anchor, right_anchor):
            selected_radius = int(radius)
            selected_tube = tube
            selected_support = cut_support
            break
    if selected_radius is None or selected_tube is None or selected_support is None:
        return CutResult("INVALID_NOT_LOCALLY_ISOLATABLE", None, 0.0, True, True, 0, 0, 0), None, None
    collateral = float(np.mean(np.asarray(other_trace_mask, dtype=bool)[selected_tube])) if selected_tube.any() else 0.0
    if collateral > float(PROTOCOL["treatment"]["maximum_collateral_fraction"]):
        return CutResult("INVALID_COLLATERAL_TRACE", selected_radius, collateral, True, False, 0, 0, int(selected_tube.sum())), selected_tube, selected_support
    left_points = rasterize(left_context_yx, support.shape, 0)
    right_points = rasterize(right_context_yx, support.shape, 0)
    left_count = int(np.sum(selected_support & left_points))
    right_count = int(np.sum(selected_support & right_points))
    minimum = int(PROTOCOL["placement"]["minimum_supported_context_points_each_side"])
    if left_count < minimum or right_count < minimum or not (selected_support & left_anchor).any() or not (selected_support & right_anchor).any():
        return CutResult("INVALID_CONTEXT_DESTROYED", selected_radius, collateral, True, False, left_count, right_count, int(selected_tube.sum())), selected_tube, selected_support
    return CutResult("VALID", selected_radius, collateral, True, False, left_count, right_count, int(selected_tube.sum())), selected_tube, selected_support

