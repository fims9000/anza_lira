"""S50 geometry-only 2D continuation audit; no native particle identities.

The curvature proxy is a quadratic **local image-plane** approximation. It is
neither a calibrated 3-D helix nor a mass/charge measurement, and its coefficients
must never be interpreted as Belyaev stopping-track parameters.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .s35_fragment_link import GEOMETRY_FIELDS, optimal_endpoints, pair_features

S50_FIELDS = (
    "normalized_signed_curvature_reversal_error",
    "normalized_half_gap_extrapolation_mismatch",
    "predicted_midgap_heading_mismatch",
)
ALL_FIELDS = GEOMETRY_FIELDS + S50_FIELDS


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-7:
        raise ValueError("Degenerate local tangent")
    return np.asarray(v, dtype=float) / n


def _oriented_end_fit(points: np.ndarray, end: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Return endpoint, outward tangent and signed 2-D curvature.

    The last sample is always the gap-facing endpoint, independent of original
    polyline ordering. Polynomial fit receives smoothed arc length; raw endpoint
    is retained for closure so extraction bias remains visible.
    """
    path = np.asarray(points, dtype=np.float64)
    if (path.ndim != 2 or path.shape[1] != 2 or len(path) < 12 or
            not np.isfinite(path).all() or end not in (0, -1)):
        raise ValueError("Invalid endpoint fit")
    oriented = path if end == -1 else path[::-1]
    # Use a fixed 31-point local support, not a source-photo-specific window.
    sample = oriented[-31:]
    smoothed = gaussian_filter1d(sample, sigma=2.2, axis=0, mode="nearest")
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(smoothed, axis=0), axis=1))]
    span = float(s[-1])
    if span < 7.0 or not np.isfinite(span):
        raise ValueError("Insufficient supported arc")
    normalized = (s - s[-1]) / span
    design = np.column_stack((np.ones(len(sample)), normalized, normalized**2))
    coeff, _, rank, _ = np.linalg.lstsq(design, smoothed, rcond=None)
    if rank != 3:
        raise ValueError("Ill-conditioned local polynomial")
    velocity = coeff[1] / span
    acceleration = 2.0 * coeff[2] / (span**2)
    speed = float(np.linalg.norm(velocity))
    if speed < .2 or not np.isfinite(speed):
        raise ValueError("Local path direction unidentifiable")
    curvature = float(np.linalg.det(np.stack((velocity, acceleration))) / (speed**3))
    if not np.isfinite(curvature):
        raise ValueError("Nonfinite image-plane curvature")
    return oriented[-1].copy(), _unit(velocity), curvature


def curvature_continuation_features(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Orientation-independent, finite 2-D continuation features only.

    A curvature sign changes when the segment's orientation reverses. Thus two
    endpoints facing each other on the same smooth curve have kA ~= -kB.
    """
    connection = optimal_endpoints(a, b)
    gap = float(connection["gap"])
    if gap < 1.0:
        raise ValueError("Gap too small for curvature continuation")
    pa, ta, ka = _oriented_end_fit(a, connection["endpoint_index_a"])
    pb, tb, kb = _oriented_end_fit(b, connection["endpoint_index_b"])
    h = gap / 2.0
    rot_a = np.array([-ta[1], ta[0]])
    rot_b = np.array([-tb[1], tb[0]])
    ma = pa + ta * h + .5 * ka * rot_a * h * h
    mb = pb + tb * h + .5 * kb * rot_b * h * h
    head_a = _unit(ta + ka * rot_a * h)
    head_b = _unit(tb + kb * rot_b * h)
    values = (
        abs(ka + kb) * gap,
        float(np.linalg.norm(ma - mb) / gap),
        float((1.0 + np.clip(head_a @ head_b, -1., 1.)) / 2.0),
    )
    if not np.isfinite(values).all():
        raise ValueError("Unidentifiable S50 features")
    return dict(zip(S50_FIELDS, map(float, values)))


def pair_geometry_and_curvature(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    base = pair_features(a, b, evidence={})
    extra = curvature_continuation_features(a, b)
    return {key: float(base[key]) for key in GEOMETRY_FIELDS} | extra


def matched_candidates(left: list[np.ndarray], right: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Remove S35 decoy length shortcut without altering injected pixels."""
    if len(left) != 2 or len(right) != 3:
        raise ValueError("Expected two fragments plus one synthetic decoy")
    if any(np.asarray(z).shape[0] < 31 for z in left + right):
        raise ValueError("Cannot match candidate lengths")
    return [np.asarray(z, dtype=float)[-31:] for z in left], [np.asarray(z, dtype=float)[:31] for z in right]
