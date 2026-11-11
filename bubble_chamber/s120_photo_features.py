"""S120: photo-only source morphology and contrast features.

Every feature comes from the unchanged original 8-bit BMP and observed S115
centerline. No manual labels, inferred gap pixels or predicted masks.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    'log_length', 'displacement_over_arc', 'bbox_aspect', 'p90_turn_deg',
    'mean_turn_deg', 'turn_variation_deg', 'mean_intensity', 'std_intensity',
    'mean_flank_contrast_4', 'p25_flank_contrast_4',
    'mean_flank_contrast_8', 'p25_flank_contrast_8',
    'fraction_contrast_gt_2', 'fraction_contrast_gt_5',
    'center_laterality_4', 'fraction_dark_strong', 'fraction_persistent_line',
    'photo_width_proxy', 'fraction_asymmetric_side',
)


def photo_features(gray: np.ndarray, points, max_samples: int = 100):
    """Sample normal profiles along original photo-supported centerlines."""
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError('requires original uint8 gray source')
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
        raise ValueError('invalid observed path')
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.r_[0., np.cumsum(steps)]
    total = float(arc[-1])
    if total <= 3:
        raise ValueError('observed path too short')
    targets = np.linspace(0., total, min(max_samples, max(12, int(total/5))))
    sx = np.interp(targets, arc, pts[:, 0])
    sy = np.interp(targets, arc, pts[:, 1])
    p = np.column_stack((sx, sy))
    before = np.column_stack((
        np.interp(np.maximum(0, targets-7), arc, pts[:, 0]),
        np.interp(np.maximum(0, targets-7), arc, pts[:, 1])))
    after = np.column_stack((
        np.interp(np.minimum(total, targets+7), arc, pts[:, 0]),
        np.interp(np.minimum(total, targets+7), arc, pts[:, 1])))
    vec = after - before
    norms = np.maximum(np.linalg.norm(vec, axis=1), 1e-6)
    u = vec / norms[:, None]
    normal = np.column_stack((-u[:, 1], u[:, 0]))
    h, w = gray.shape

    def sample(offset):
        loc = np.rint(p + normal*offset).astype(int)
        loc[:, 0] = np.clip(loc[:, 0], 0, w-1)
        loc[:, 1] = np.clip(loc[:, 1], 0, h-1)
        return gray[loc[:, 1], loc[:, 0]].astype(float)

    cen = sample(0)
    p2, n2 = sample(2), sample(-2)
    p4, n4 = sample(4), sample(-4)
    p8, n8 = sample(8), sample(-8)
    contrast4 = (p4+n4)/2-cen
    contrast8 = (p8+n8)/2-cen
    turn = np.arccos(np.clip(np.sum(u[:-1]*u[1:], axis=1), -1, 1))*180/np.pi
    if len(turn) < 3:
        turn = np.zeros(4)
    chord = np.linalg.norm(pts[-1]-pts[0])
    extent = np.ptp(pts, axis=0)
    return np.array([
        np.log1p(total), chord/max(total, 1e-6),
        min(extent)/max(1., max(extent)),
        np.quantile(turn, .90), np.mean(turn), np.std(turn),
        np.mean(cen), np.std(cen),
        np.mean(contrast4), np.quantile(contrast4, .25),
        np.mean(contrast8), np.quantile(contrast8, .25),
        np.mean(contrast4 > 2), np.mean(contrast4 > 5),
        np.mean(np.abs(p4-n4)), np.mean(cen < 110),
        np.mean((contrast4 > 2) & (contrast8 > 2)),
        np.mean((p2+n2)/2-cen > 2),
        np.mean(np.abs(p8-n8) > 15),
    ], dtype=float)
