"""Exact arclength-bounded flat-cap ribbon manipulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lira_graph_cut_v2.graph_cut import connected, rasterize
from lira_h1.protocol import CUT_RADII, PROTOCOL


def cumulative_arclength(points_yx: np.ndarray) -> np.ndarray:
    points = np.asarray(points_yx, dtype=np.float64)
    if len(points) < 2:
        raise ValueError("a trace needs at least two points")
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))


def flat_cap_ribbon(
    trace_yx: np.ndarray,
    s_a: float,
    s_b: float,
    radius: float,
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize pixels whose nearest full-trace projection lies in [s_a,s_b]."""
    points = np.asarray(trace_yx, dtype=np.float64)
    arc = cumulative_arclength(points)
    if not (0.0 <= s_a < s_b <= float(arc[-1]) + 1e-9):
        raise ValueError("invalid hidden arclength interval")
    expansion = int(PROTOCOL["treatment"]["roi_expansion_px"])
    in_interval = (arc >= s_a - 1e-9) & (arc <= s_b + 1e-9)
    relevant_points = points[in_interval]
    if not len(relevant_points):
        relevant_points = points
    y0 = max(0, int(np.floor(relevant_points[:, 0].min())) - expansion)
    y1 = min(shape[0], int(np.ceil(relevant_points[:, 0].max())) + expansion + 1)
    x0 = max(0, int(np.floor(relevant_points[:, 1].min())) - expansion)
    x1 = min(shape[1], int(np.ceil(relevant_points[:, 1].max())) + expansion + 1)
    best_d2 = np.full((y1 - y0, x1 - x0), np.inf, dtype=np.float64)
    best_s = np.zeros_like(best_d2)
    # Iterate in segment order and replace only on strict improvement. This is
    # the frozen smallest-segment-index tie break.
    for index, (first, second) in enumerate(zip(points[:-1], points[1:])):
        lo_y = max(y0, int(np.floor(min(first[0], second[0]) - radius)))
        hi_y = min(y1, int(np.ceil(max(first[0], second[0]) + radius)) + 1)
        lo_x = max(x0, int(np.floor(min(first[1], second[1]) - radius)))
        hi_x = min(x1, int(np.ceil(max(first[1], second[1]) + radius)) + 1)
        if lo_y >= hi_y or lo_x >= hi_x:
            continue
        yy, xx = np.mgrid[lo_y:hi_y, lo_x:hi_x]
        samples = np.stack((yy, xx), axis=-1).astype(np.float64)
        vector = second - first
        length2 = float(vector @ vector)
        if length2 <= 1e-12:
            t = np.zeros(samples.shape[:2], dtype=np.float64)
        else:
            t = np.clip(np.sum((samples - first) * vector, axis=-1) / length2, 0.0, 1.0)
        projected = first + t[..., None] * vector
        distance2 = np.sum((samples - projected) ** 2, axis=-1)
        ys = slice(lo_y - y0, hi_y - y0)
        xs = slice(lo_x - x0, hi_x - x0)
        old = best_d2[ys, xs]
        update = distance2 < old - 1e-12
        old[update] = distance2[update]
        local_s = arc[index] + t * float(np.sqrt(length2))
        best_s[ys, xs][update] = local_s[update]
    local = (best_d2 <= float(radius) ** 2 + 1e-9) & (best_s >= s_a - 1e-9) & (best_s <= s_b + 1e-9)
    output = np.zeros(shape, dtype=bool)
    output[y0:y1, x0:x1] = local
    return output


@dataclass(frozen=True)
class RibbonCutResult:
    status: str
    radius: int | None
    collateral_fraction: float
    pre_connected: bool
    post_connected: bool | None
    left_context_supported: int
    right_context_supported: int
    ribbon_pixels: int


def minimal_valid_ribbon_cut(
    probability: np.ndarray,
    trace_yx: np.ndarray,
    s_a: float,
    s_b: float,
    left_anchor_yx: np.ndarray,
    right_anchor_yx: np.ndarray,
    left_context_yx: np.ndarray,
    right_context_yx: np.ndarray,
    other_trace_mask: np.ndarray,
) -> tuple[RibbonCutResult, np.ndarray | None, np.ndarray | None]:
    threshold = float(PROTOCOL["treatment"]["validation_threshold"])
    support = np.asarray(probability) >= threshold
    anchor_radius = int(PROTOCOL["treatment"]["anchor_raster_radius_px"])
    left_anchor = rasterize(left_anchor_yx, support.shape, anchor_radius)
    right_anchor = rasterize(right_anchor_yx, support.shape, anchor_radius)
    if not (support & left_anchor).any() or not (support & right_anchor).any():
        return RibbonCutResult("INELIGIBLE_ANCHOR_SUPPORT", None, 0.0, False, None, 0, 0, 0), None, None
    if not connected(support, left_anchor, right_anchor):
        return RibbonCutResult("INELIGIBLE_PRE_DISCONNECTED", None, 0.0, False, None, 0, 0, 0), None, None
    selected = None
    for radius in CUT_RADII:
        ribbon = flat_cap_ribbon(trace_yx, s_a, s_b, radius, support.shape)
        cut_support = support & ~ribbon
        if not connected(cut_support, left_anchor, right_anchor):
            selected = (int(radius), ribbon, cut_support)
            break
    if selected is None:
        return RibbonCutResult("INVALID_NOT_LOCALLY_ISOLATABLE", None, 0.0, True, True, 0, 0, 0), None, None
    radius, ribbon, cut_support = selected
    collateral = float(np.mean(np.asarray(other_trace_mask, dtype=bool)[ribbon])) if ribbon.any() else 0.0
    if collateral > float(PROTOCOL["treatment"]["maximum_collateral_fraction"]):
        return RibbonCutResult("INVALID_COLLATERAL_TRACE", radius, collateral, True, False, 0, 0, int(ribbon.sum())), ribbon, cut_support
    left_points = rasterize(left_context_yx, support.shape, 0)
    right_points = rasterize(right_context_yx, support.shape, 0)
    left_count = int(np.sum(cut_support & left_points))
    right_count = int(np.sum(cut_support & right_points))
    minimum = int(PROTOCOL["placement"]["minimum_supported_context_points_each_side"])
    if left_count < minimum or right_count < minimum or not (cut_support & left_anchor).any() or not (cut_support & right_anchor).any():
        return RibbonCutResult("INVALID_CONTEXT_DESTROYED", radius, collateral, True, False, left_count, right_count, int(ribbon.sum())), ribbon, cut_support
    return RibbonCutResult("VALID", radius, collateral, True, False, left_count, right_count, int(ribbon.sum())), ribbon, cut_support

