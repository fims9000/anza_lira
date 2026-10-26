"""S53: local contrast intermittency along unverified 2D candidate curves.

The feature output is an image descriptor, NOT a likelihood of a physical
particle. Native H50Z centerlines have no independent dense ground truth.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from .s39_machine_auto import sample_bilinear
from .s43_artifact_triage import resample_polyline

SCHEMA = 's53_synthetic_renderer_texture_advisory_v1'
FEATURE_NAMES = (
    'contrast_path_q90_minus_q10',
    'positive_local_contrast_fraction',
    'contrast_highpass_to_total_rms',
    'contrast_peak_count_per_100px',
    'peak_prominence_median_to_contrast_rms',
    'spectral_midband_energy_fraction',
    'short_lag_autocorrelation_1_to_4_max',
    'excess_contrast_fluctuation_over_flank_noise',
)


def texture_features(gray: np.ndarray, curve: np.ndarray | list) -> np.ndarray:
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 96:
        raise ValueError('Expected raw grayscale image at native bit depth')
    p = resample_polyline(curve, gray.shape, n=100)
    delta = np.gradient(p, axis=0)
    normal = np.column_stack((-delta[:, 1], delta[:, 0]))
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1.e-7)
    center = sample_bilinear(gray, p).astype(float)
    flank_left = sample_bilinear(gray, p + normal * 6.).astype(float)
    flank_right = sample_bilinear(gray, p - normal * 6.).astype(float)
    reference = (flank_left + flank_right) / 2.
    contrast = reference - center
    # Preserve native exposures; do not invert or quantile-normalize a photo.
    slow = gaussian_filter1d(contrast, 7.0, mode='reflect')
    detrend = contrast - slow
    rms = float(np.sqrt(np.mean((contrast - np.mean(contrast)) ** 2)))
    highpass = float(np.sqrt(np.mean(detrend ** 2)))
    noise = (flank_left - flank_right) / 2.
    noise = noise - gaussian_filter1d(noise, 7., mode='reflect')
    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
    scale = float(np.quantile(contrast, .85) - np.quantile(contrast, .15))
    peak_prominence = max(2.0, scale * .26)
    peaks, properties = find_peaks(gaussian_filter1d(detrend, .65),
                                   prominence=peak_prominence, distance=2)
    arc_length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    fft = np.abs(np.fft.rfft(detrend - np.mean(detrend))) ** 2
    freq = np.fft.rfftfreq(len(detrend))
    use = (freq >= .04) & (freq <= .45)
    mid = (freq >= .12) & (freq <= .38)
    spectral = float(np.sum(fft[mid]) / (np.sum(fft[use]) + 1.e-6))
    centered = detrend - np.mean(detrend)
    denom = float(np.sum(centered ** 2)) + 1.e-6
    autocorrelation = max(float(np.dot(centered[:-lag], centered[lag:]) / denom)
                          for lag in (2, 3, 4))
    prominence = properties['prominences']
    arr = np.array([
        float(np.quantile(contrast, .90) - np.quantile(contrast, .10)),
        float(np.mean(contrast > 3.0)),
        highpass / (rms + 1.e-3),
        len(peaks) / max(arc_length, 1.) * 100.,
        float(np.median(prominence)) / (rms + 1.) if len(peaks) else 0.,
        spectral,
        autocorrelation,
        np.clip((highpass - noise_rms) / (rms + 1.), -5., 5.),
    ], dtype=float)
    if len(arr) != len(FEATURE_NAMES) or not np.isfinite(arr).all():
        raise ValueError('Non-finite S53 image descriptor')
    return arr


def merge_features(existing: np.ndarray, additional: np.ndarray) -> np.ndarray:
    old = np.asarray(existing, dtype=float)
    new = np.asarray(additional, dtype=float)
    if old.ndim != 2 or new.ndim != 2 or old.shape[0] != new.shape[0] or new.shape[1] != len(FEATURE_NAMES):
        raise ValueError('Mismatched S43/S53 image descriptors')
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError('Invalid source image descriptors')
    return np.concatenate([old, new], axis=1)
