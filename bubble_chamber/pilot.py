"""Auditable, conservative first-stage extraction of photographed particle tracks.

This is a classical image-processing baseline, NOT pretrained ANZA-LIRA. It
returns candidate 2D track fragments and REVIEW-ONLY joins. The camera,
exposure/film and stereo calibration are deliberately outside this pilot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import math

import cv2
import numpy as np
from scipy.ndimage import label as label_components
from skimage.morphology import skeletonize

from .ridges import frangi_dark_ridges
from .photometry import sample_track_profiles


VERSION = "bubble_classical_p0_v1"
OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)


@dataclass(frozen=True)
class PilotConfig:
    threshold: float = 18.0
    background_sigma: float = 14.0
    denoise_sigma: float = 0.7
    min_component_area: int = 15
    min_segment_length_px: float = 28.0
    points_per_fragment: int = 14
    max_gap_px: float = 42.0
    max_join_angle_deg: float = 32.0
    max_candidates_per_endpoint: int = 4
    segmentation_method: str = "contrast"  # or "frangi"; original default preserved
    ridge_response_fraction: float = 0.12
    ridge_scales: tuple[float, ...] = (1.0, 2.0, 3.0)


def _edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a < b else (b, a)


def extract_skeleton_segments(skeleton: np.ndarray, min_length_px: float = 0.0) -> list[np.ndarray]:
    """Split 8-connected skeleton into ordered chains at endpoints/junctions.

    Pixel coordinates are [y, x]. A crossing is a junction, never silently
    interpreted as a track continuation. Closed loops are supported.
    """
    if skeleton.ndim != 2:
        raise ValueError("skeleton must be HxW")
    coords = {tuple(map(int, p)) for p in np.argwhere(skeleton)}
    neighbors = {p: tuple(sorted((p[0] + dy, p[1] + dx) for dy, dx in OFFSETS if (p[0] + dy, p[1] + dx) in coords)) for p in coords}
    nodes = sorted(p for p in coords if len(neighbors[p]) != 2)
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paths: list[np.ndarray] = []

    def walk(start: tuple[int, int], nxt: tuple[int, int], loop: bool) -> None:
        path = [start, nxt]
        visited.add(_edge(start, nxt))
        prev, cur = start, nxt
        for _ in range(len(coords) + 1):
            if not loop and len(neighbors[cur]) != 2:
                break
            available = [q for q in neighbors[cur] if q != prev and _edge(cur, q) not in visited]
            if not available:
                break
            q = available[0]
            visited.add(_edge(cur, q))
            path.append(q)
            prev, cur = cur, q
            if cur == start:
                break
        points = np.asarray(path, dtype=np.float64)
        if len(points) > 1 and float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) >= min_length_px:
            paths.append(points)

    for start in nodes:
        for nxt in neighbors[start]:
            if _edge(start, nxt) not in visited:
                walk(start, nxt, False)
    for start in sorted(coords):
        for nxt in neighbors[start]:
            if _edge(start, nxt) not in visited:
                walk(start, nxt, True)
    return paths


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        raise ValueError("degenerate direction")
    return np.asarray(v, dtype=float) / norm


def fit_quadratic(points_yx: np.ndarray) -> dict[str, Any]:
    """Fit v=a*u^2+b*u+c in a local PCA-aligned coordinate frame.

    Input is an ordered *geometric* fragment; endpoint A/B are not physical
    production/stopping points without stereo/kinematic information.
    """
    yx = np.asarray(points_yx, dtype=np.float64)
    if yx.ndim != 2 or yx.shape[1] != 2 or len(yx) < 4:
        raise ValueError("at least four 2D points required")
    xy = yx[:, ::-1]
    center = xy.mean(axis=0)
    centered = xy - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    u_axis = vh[0]
    if float(np.dot(u_axis, xy[-1] - xy[0])) < 0:
        u_axis = -u_axis
    v_axis = np.asarray([-u_axis[1], u_axis[0]])
    u = centered @ u_axis
    v = centered @ v_axis
    design = np.column_stack((u * u, u, np.ones_like(u)))
    coef, _, rank, _ = np.linalg.lstsq(design, v, rcond=None)
    estimated = design @ coef
    rms = float(np.sqrt(np.mean((v - estimated) ** 2)))
    direction_diff = np.diff(u)
    monotone_fraction = float(max((direction_diff >= -1e-6).mean(), (direction_diff <= 1e-6).mean())) if direction_diff.size else 0.0
    return {
        "basis": "pca_local_xy",
        "origin_xy_px": center.tolist(),
        "u_axis_xy": u_axis.tolist(),
        "v_axis_xy": v_axis.tolist(),
        "coefficients_a_b_c": coef.tolist(),
        "u_range_px": [float(u.min()), float(u.max())],
        "residual_rms_px": rms,
        "single_valued_warning": bool(monotone_fraction < 0.8),
        "endpoint_A_xy_px": xy[0].tolist(),
        "endpoint_B_xy_px": xy[-1].tolist(),
    }


def arc_sample(points_yx: np.ndarray, count: int) -> np.ndarray:
    """Uniform arc-length samples in [y,x], including both observed endpoints."""
    points = np.asarray(points_yx, dtype=np.float64)
    if count < 2 or len(points) < 2:
        raise ValueError("count and number of points must be >=2")
    dist = np.linalg.norm(np.diff(points, axis=0), axis=1)
    lengths = np.concatenate(([0.0], np.cumsum(dist)))
    unique, indices = np.unique(lengths, return_index=True)
    if len(unique) < 2:
        raise ValueError("zero-length path")
    xs = np.interp(np.linspace(0, unique[-1], count), unique, points[indices, 1])
    ys = np.interp(np.linspace(0, unique[-1], count), unique, points[indices, 0])
    return np.column_stack((ys, xs))


def _endpoint(pt: np.ndarray, at_start: bool, n: int = 7) -> tuple[np.ndarray, np.ndarray]:
    if at_start:
        return pt[0], _unit(pt[0] - pt[min(n, len(pt) - 1)])
    return pt[-1], _unit(pt[-1] - pt[max(0, len(pt) - n - 1)])


def propose_joins(
    segments: list[np.ndarray],
    *,
    max_gap_px: float = 42.0,
    max_join_angle_deg: float = 32.0,
    max_candidates_per_endpoint: int = 4,
) -> list[dict[str, Any]]:
    """Rank geometrically plausible 2D joins WITHOUT joining any fragments.

    Returned candidates require human/stereo confirmation. When multiple
    destinations compete for an endpoint, the candidate is flagged ambiguous.
    """
    ends = []
    for i, points in enumerate(segments):
        if len(points) < 3:
            continue
        for at_start in (True, False):
            pos, outward = _endpoint(np.asarray(points), at_start)
            ends.append((i, "A" if at_start else "B", pos, outward))
    threshold = math.radians(max_join_angle_deg)
    proposals = []
    counts: dict[tuple[int, str], int] = {}
    for k, (i, label_i, pi, di) in enumerate(ends):
        for j, label_j, pj, dj in ends[k + 1:]:
            if i == j:
                continue
            delta = pj - pi
            distance = float(np.linalg.norm(delta))
            if not 2.0 <= distance <= max_gap_px:
                continue
            direction = delta / distance
            angle_a = math.acos(float(np.clip(np.dot(di, direction), -1, 1)))
            angle_b = math.acos(float(np.clip(np.dot(dj, -direction), -1, 1)))
            if max(angle_a, angle_b) > threshold:
                continue
            score = 0.5 * (angle_a + angle_b) / max(threshold, 1e-6) + distance / max_gap_px
            proposals.append({
                "segment_A": i, "endpoint_A": label_i, "segment_B": j, "endpoint_B": label_j,
                "gap_px": round(distance, 4), "angle_A_deg": round(math.degrees(angle_a), 3),
                "angle_B_deg": round(math.degrees(angle_b), 3), "rank_cost": round(score, 5),
                "status": "needs_review", "automatic_merge": False,
            })
            counts[(i, label_i)] = counts.get((i, label_i), 0) + 1
            counts[(j, label_j)] = counts.get((j, label_j), 0) + 1
    proposals.sort(key=lambda p: p["rank_cost"])
    for p in proposals:
        p["ambiguous"] = max(counts[(p["segment_A"], p["endpoint_A"])], counts[(p["segment_B"], p["endpoint_B"])]) > 1
    if max_candidates_per_endpoint < 1:
        return []
    kept: list[dict[str, Any]] = []
    used: dict[tuple[int, str], int] = {}
    for p in proposals:
        a = (p["segment_A"], p["endpoint_A"])
        b = (p["segment_B"], p["endpoint_B"])
        if used.get(a, 0) >= max_candidates_per_endpoint or used.get(b, 0) >= max_candidates_per_endpoint:
            continue
        kept.append(p)
        used[a] = used.get(a, 0) + 1
        used[b] = used.get(b, 0) + 1
    return kept


def analyze_frame(image_gray: np.ndarray, cfg: PilotConfig = PilotConfig(), *, roi_xyxy: tuple[int, int, int, int] | None = None) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Return JSON-compatible measurements, binary foreground and RGB overlay.

    The baseline is intentionally 2-D and calibration-free. It cannot infer
    mass, charge, physical track direction, or independent particle identity.
    """
    raw = np.asarray(image_gray)
    if raw.ndim != 2 or raw.dtype != np.uint8:
        raise ValueError("input must be a grayscale uint8 image")
    h, w = raw.shape
    roi = roi_xyxy if roi_xyxy is not None else (0, 0, w, h)
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("invalid ROI")
    crop = raw[y0:y1, x0:x1]
    smooth = cv2.GaussianBlur(crop, (0, 0), sigmaX=cfg.denoise_sigma)
    background = cv2.GaussianBlur(smooth, (0, 0), sigmaX=cfg.background_sigma)
    contrast = cv2.subtract(background, smooth)
    if cfg.segmentation_method == "contrast":
        raw_mask = contrast >= cfg.threshold
        components, _ = label_components(raw_mask, structure=np.ones((3, 3), dtype=np.uint8))
        component_sizes = np.bincount(components.ravel())
        mask = raw_mask & (components > 0) & (component_sizes[components] >= cfg.min_component_area)
        pipeline_id = VERSION
    elif cfg.segmentation_method == "frangi":
        mask, _ = frangi_dark_ridges(
            crop, response_fraction=cfg.ridge_response_fraction,
            scales=cfg.ridge_scales, min_area=cfg.min_component_area,
        )
        pipeline_id = "bubble_p2_multiscale_frangi_v1"
    else:
        raise ValueError("unknown segmentation_method: " + repr(cfg.segmentation_method))
    skeleton = skeletonize(mask)
    pieces = extract_skeleton_segments(skeleton, cfg.min_segment_length_px)
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    fitted = []
    shifted = []
    for seg in pieces:
        offset = np.asarray([y0, x0], dtype=np.float64)
        shifted_seg = seg + offset
        shifted.append(shifted_seg)
        poly = fit_quadratic(shifted_seg)
        sample = arc_sample(shifted_seg, cfg.points_per_fragment)
        pix = np.rint(seg).astype(int)
        widths = 2 * distance[np.clip(pix[:, 0], 0, len(crop) - 1), np.clip(pix[:, 1], 0, crop.shape[1] - 1)]
        darkness = contrast[np.clip(pix[:, 0], 0, len(crop) - 1), np.clip(pix[:, 1], 0, crop.shape[1] - 1)]
        segment_data = {"segment_id": len(fitted), "observed_length_px": round(float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()), 4),
                       "fit": poly, "measurement_xy_px": sample[:, ::-1].tolist(),
                       "width_proxy_px_mean": round(float(widths.mean()), 4),
                       "darkness_proxy_mean": round(float(darkness.mean()), 4),
                       "semantic_warning": "A/B are geometric endpoints, not physical start/stop"}
        if cfg.segmentation_method == "frangi":
            # Photometry uses source image intensity, not the binary Frangi mask.
            # Profile records are in original full-image XY coordinates.
            crop_sample_xy = sample[:, ::-1] - np.array([x0, y0], dtype=float)
            profiles = sample_track_profiles(crop, crop_sample_xy)
            for record, (x, y) in zip(profiles, sample[:, ::-1]):
                record["x_source_px"] = float(x)
                record["y_source_px"] = float(y)
            segment_data["transverse_profile_samples"] = profiles
            segment_data["profile_status"] = "IMAGE_SPACE_ONLY_NOT_CALIBRATED_DEDX"
        fitted.append(segment_data)
    proposals = propose_joins(shifted, max_gap_px=cfg.max_gap_px, max_join_angle_deg=cfg.max_join_angle_deg, max_candidates_per_endpoint=cfg.max_candidates_per_endpoint)
    overlay = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    for i, seg in enumerate(pieces):
        path = np.rint(seg[:, ::-1]).astype(np.int32)
        rgb = ((255, 70, 70), (70, 160, 255), (55, 190, 70), (255, 160, 55), (210, 80, 200))[i % 5]
        cv2.polylines(overlay, [path.reshape(-1, 1, 2)], False, rgb, 2, cv2.LINE_AA)
        # ID annotation is intentionally omitted: too dense on full film frames.
    result = {
        "pipeline": pipeline_id, "config": asdict(cfg), "roi_xyxy": list(roi),
        "source_sha256_grayscale_pixels": sha256(raw.tobytes()).hexdigest(),
        "frame_shape_yx": [h, w], "foreground_pixel_count": int(mask.sum()),
        "fragment_count": len(fitted), "proposed_joins_count": len(proposals),
        "automatic_join_count": 0, "segments": fitted, "join_proposals": proposals,
        "physical_result_status": "NOT_RECONSTRUCTED",
        "no_ground_truth": True,
    }
    return result, mask.astype(np.uint8) * 255, overlay
