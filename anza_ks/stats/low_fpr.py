"""Calibration-only operating thresholds and threshold-free low-FPR curves."""

from __future__ import annotations

import numpy as np


def threshold_at_fpr(negative_scores: np.ndarray, maximum_fpr: float = 0.05) -> float:
    negative = np.asarray(negative_scores, dtype=np.float64)
    if negative.ndim != 1 or len(negative) == 0 or not 0 <= maximum_fpr < 1:
        raise ValueError("invalid negative calibration scores or FPR budget")
    allowed = int(np.floor(maximum_fpr * len(negative)))
    descending = np.sort(negative)[::-1]
    if allowed == 0:
        return float(np.nextafter(descending[0], np.inf))
    return float(np.nextafter(descending[allowed - 1], np.inf))


def tpr_at_fpr_curve(positive_scores: np.ndarray, negative_scores: np.ndarray, maximum_fpr: float = 0.05) -> tuple[float, float, float]:
    positive = np.asarray(positive_scores, dtype=np.float64)
    negative = np.asarray(negative_scores, dtype=np.float64)
    thresholds = np.concatenate(([np.inf], np.unique(np.concatenate((positive, negative)))[::-1], [-np.inf]))
    best = (0.0, 0.0, float("inf"))
    for threshold in thresholds:
        fpr = float(np.mean(negative >= threshold))
        tpr = float(np.mean(positive >= threshold))
        if fpr <= maximum_fpr + 1e-12 and (tpr > best[0] or (tpr == best[0] and fpr < best[1])):
            best = (tpr, fpr, float(threshold))
    return best


def operating_curve(positive_scores: np.ndarray, negative_scores: np.ndarray) -> list[dict[str, float]]:
    positive = np.asarray(positive_scores, dtype=np.float64)
    negative = np.asarray(negative_scores, dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, 101)
    thresholds = np.unique(np.quantile(np.concatenate((positive, negative)), quantiles))[::-1]
    return [
        {
            "threshold": float(threshold),
            "tpr": float(np.mean(positive >= threshold)),
            "fpr": float(np.mean(negative >= threshold)),
        }
        for threshold in thresholds
    ]
