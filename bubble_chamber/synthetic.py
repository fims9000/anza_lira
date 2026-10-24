"""Controlled photographic stand-ins for smoke tests; never real calibration data."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_dilation


def generate_case(case: str, seed: int = 41, shape: tuple[int, int] = (384, 384)) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return simulated film, *complete* reference centerline, and metadata.

    There is no claim these scenes reproduce the detector's physical statistics.
    Explicit masks are synthetic ground truth used only for software sanity.
    """
    if case not in ("continuous", "gap", "crossing", "parallel", "weak_noisy"):
        raise ValueError(case)
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    base = 230 + 11 * (xx / w) + 6 * (yy / h) + rng.normal(0, 12 if case == "weak_noisy" else 4, shape)
    observed = np.zeros(shape, dtype=np.uint8)
    reference = np.zeros(shape, dtype=np.uint8)

    def track(curve, x_start, x_end, *, omit_range=None):
        points = [(int(x), int(round(curve(x)))) for x in range(x_start, x_end + 1)]
        for a, b in zip(points, points[1:]):
            cv2.line(reference, a, b, 255, 1, cv2.LINE_8)
            if omit_range is None or not (omit_range[0] <= (a[0] + b[0]) / 2 <= omit_range[1]):
                cv2.line(observed, a, b, 255, 2, cv2.LINE_8)

    track(lambda x: 170 + 0.2 * (x - 200) + 0.0025 * (x - 200) ** 2, 48, 338, omit_range=(187, 213) if case == "gap" else None)
    if case == "crossing":
        track(lambda x: 235 - 0.39 * (x - 170), 75, 318)
    if case == "parallel":
        track(lambda x: 193 + 0.20 * (x - 200) + 0.0025 * (x - 200) ** 2, 48, 338)
    # Sparse foreground speckles, not part of ground truth.
    speck = rng.random(shape) < (0.004 if case == "weak_noisy" else 0.0005)
    base[observed > 0] -= 24 if case == "weak_noisy" else 85
    base[speck] -= 75
    return np.clip(base, 0, 255).astype(np.uint8), reference > 0, {"case": case, "seed": seed, "missing_observed_interval_x": [187, 213] if case == "gap" else None}


def centerline_scores(predicted_mask: np.ndarray, truth_centerline: np.ndarray, radius_px: int = 3) -> dict[str, float]:
    """Tolerant centerline precision/recall, not physical track identity metrics."""
    from skimage.morphology import skeletonize
    pred = skeletonize(predicted_mask.astype(bool))
    gt = np.asarray(truth_centerline, dtype=bool)
    yy, xx = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    disk = (xx * xx + yy * yy) <= radius_px ** 2
    gt_band = binary_dilation(gt, structure=disk)
    pred_band = binary_dilation(pred, structure=disk)
    precision = float(np.logical_and(pred, gt_band).sum() / max(int(pred.sum()), 1))
    recall = float(np.logical_and(gt, pred_band).sum() / max(int(gt.sum()), 1))
    return {"precision_tolerant": round(precision, 6), "recall_tolerant": round(recall, 6),
            "f1_tolerant": round(2 * precision * recall / max(precision + recall, 1e-12), 6),
            "reference_centerline_px": int(gt.sum()), "predicted_skeleton_px": int(pred.sum()),
            "radius_px": radius_px}
