"""Threshold-free matched metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def matched_ranking(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if positive.shape != negative.shape or positive.ndim != 1:
        raise ValueError("paired scores must be aligned vectors")
    return float(np.mean((positive > negative) + 0.5 * (positive == negative)))


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    labels = np.concatenate((np.ones_like(positive), np.zeros_like(negative)))
    scores = np.concatenate((positive, negative))
    return float(roc_auc_score(labels, scores))
