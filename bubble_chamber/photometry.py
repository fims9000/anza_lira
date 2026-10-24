"""Image-space line-width / darkness profiles along measured track fragments.

These observables are NOT calibrated dE/dx or particle-ID measurements. Their
interpretation requires film development, scanner, chamber/run and 3D geometry.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def _halfmax_span(offsets: np.ndarray, deficit: np.ndarray, level: float, minimum: int) -> float | None:
    """Width of the contiguous above-half-depth interval around the local peak."""
    left = right = minimum
    while left > 0 and deficit[left - 1] >= level:
        left -= 1
    while right + 1 < len(offsets) and deficit[right + 1] >= level:
        right += 1
    if left == 0 or right == len(offsets) - 1:
        return None  # FWHM was not contained in the sampling window
    # Linear crossing with the neighboring below-half-depth samples.
    def cross(ia: int, ib: int) -> float:
        va, vb = float(deficit[ia]), float(deficit[ib])
        if abs(vb - va) < 1e-9:
            return float(offsets[ia])
        t = float(np.clip((level - va) / (vb - va), 0, 1))
        return float(offsets[ia] + t * (offsets[ib] - offsets[ia]))
    return cross(right, right + 1) - cross(left - 1, left)


def sample_track_profiles(
    image_gray: np.ndarray,
    points_xy: np.ndarray,
    *,
    radius_px: int = 8,
    min_darkness: float = 7.0,
) -> list[dict]:
    """Estimate image-space transverse profiles at ordered sample coordinates.

    Input points_xy must use the *cropped image* coordinate frame.
    Estimation uses bilinear interpolation with line-local normal and a
    two-sided background estimate. Only finite, in-frame observations are used.
    """
    image = np.asarray(image_gray)
    xy = np.asarray(points_xy, dtype=np.float64)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("image must be uint8 HxW")
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
        raise ValueError("points_xy must be Nx2 finite with >=3 ordered points")
    if radius_px < 4 or not np.isfinite(min_darkness) or min_darkness < 0:
        raise ValueError("invalid profile parameters")
    offsets = np.arange(-radius_px, radius_px + 1, dtype=np.float64)
    image_float = image.astype(np.float32, copy=False)
    output = []
    for idx, point in enumerate(xy):
        prev = xy[max(0, idx - 1)]
        nxt = xy[min(len(xy) - 1, idx + 1)]
        tangent = nxt - prev
        mag = float(np.linalg.norm(tangent))
        record = {"index": idx, "x_source_px": float(point[0]), "y_source_px": float(point[1]),
                  "status": "unmeasured", "width_fwhm_px": None, "darkness_px": None}
        if mag < 1e-7:
            record["status"] = "degenerate_tangent"; output.append(record); continue
        normal = np.array([-tangent[1], tangent[0]], dtype=float) / mag
        sample_xy = point[None, :] + offsets[:, None] * normal[None, :]
        if np.any(sample_xy[:, 0] < 0) or np.any(sample_xy[:, 0] > image.shape[1] - 1) or np.any(sample_xy[:, 1] < 0) or np.any(sample_xy[:, 1] > image.shape[0] - 1):
            record["status"] = "border_truncated"; output.append(record); continue
        intensities = map_coordinates(image_float, [sample_xy[:, 1], sample_xy[:, 0]], order=1, mode="nearest")
        flank_count = max(2, radius_px // 4)
        baseline = float(np.median(np.r_[intensities[:flank_count], intensities[-flank_count:]]))
        deficits = baseline - np.asarray(intensities, dtype=np.float64)
        mid = radius_px
        local = deficits[mid-2:mid+3]
        best_idx = int(np.argmax(local)) + mid - 2
        darkness = float(max(0.0, deficits[best_idx]))
        record["darkness_px"] = round(darkness, 5)
        if darkness < min_darkness:
            record["status"] = "low_local_contrast"; output.append(record); continue
        width = _halfmax_span(offsets, deficits, darkness / 2.0, best_idx)
        if width is None:
            record["status"] = "unbounded_profile"; output.append(record); continue
        record["status"] = "valid"
        record["width_fwhm_px"] = round(float(width), 5)
        output.append(record)
    return output
