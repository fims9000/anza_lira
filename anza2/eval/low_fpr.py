"""Exact inclusive low-FPR metrics without threshold-tie leakage."""

from __future__ import annotations

import numpy as np


def select_threshold(negative_scores: np.ndarray, *, max_fpr: float = 0.05) -> float:
    negatives = np.asarray(negative_scores, dtype=np.float64)
    if negatives.size == 0 or not np.isfinite(negatives).all():
        raise ValueError("finite negative scores are required")
    selected = float("inf")
    for candidate in np.unique(negatives)[::-1]:
        if float(np.mean(negatives >= candidate)) <= float(max_fpr):
            selected = float(candidate)
        else:
            break
    return selected


def _roc_arrays(positive_scores: np.ndarray, negative_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = np.asarray(positive_scores, dtype=np.float64)
    negative = np.asarray(negative_scores, dtype=np.float64)
    scores = np.concatenate((positive, negative))
    labels = np.concatenate((np.ones(positive.size, dtype=np.int8), np.zeros(negative.size, dtype=np.int8)))
    order = np.argsort(-scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    cumulative_tp = np.cumsum(labels)
    cumulative_fp = np.cumsum(1 - labels)
    last_for_threshold = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), scores.size - 1]
    thresholds = np.r_[np.inf, scores[last_for_threshold], -np.inf]
    tpr = np.r_[0.0, cumulative_tp[last_for_threshold] / positive.size, 1.0]
    fpr = np.r_[0.0, cumulative_fp[last_for_threshold] / negative.size, 1.0]
    return thresholds, tpr, fpr


def operating_curve(positive_scores: np.ndarray, negative_scores: np.ndarray) -> list[dict[str, float]]:
    thresholds, tpr, fpr = _roc_arrays(positive_scores, negative_scores)
    return [
        {"threshold": float(threshold), "tpr": float(true_rate), "fpr": float(false_rate)}
        for threshold, true_rate, false_rate in zip(thresholds, tpr, fpr, strict=True)
    ]


def _partial_auc(fpr: np.ndarray, tpr: np.ndarray, max_fpr: float) -> float:
    after = int(np.searchsorted(fpr, max_fpr, side="right"))
    xs = fpr[:after].tolist(); ys = tpr[:after].tolist()
    if not xs or xs[-1] < max_fpr:
        right = min(after, len(fpr) - 1); left = max(0, right - 1)
        fraction = 0.0 if fpr[right] == fpr[left] else (max_fpr - fpr[left]) / (fpr[right] - fpr[left])
        xs.append(float(max_fpr)); ys.append(float(tpr[left] + fraction * (tpr[right] - tpr[left])))
    return float(np.trapezoid(ys, xs) / max_fpr)


def sampled_operating_curve(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    *,
    max_points: int = 201,
) -> list[dict[str, float]]:
    thresholds, tpr, fpr = _roc_arrays(positive_scores, negative_scores)
    if len(thresholds) > max_points:
        indices = np.unique(np.linspace(0, len(thresholds) - 1, max_points, dtype=int))
        thresholds, tpr, fpr = thresholds[indices], tpr[indices], fpr[indices]
    return [
        {"threshold": float(threshold), "tpr": float(true_rate), "fpr": float(false_rate)}
        for threshold, true_rate, false_rate in zip(thresholds, tpr, fpr, strict=True)
    ]


def _ranking_probability(positive: np.ndarray, negative: np.ndarray) -> float:
    # Equivalent to AUROC, computed in bounded chunks to avoid an NxM matrix.
    negative_sorted = np.sort(negative)
    less = np.searchsorted(negative_sorted, positive, side="left")
    equal = np.searchsorted(negative_sorted, positive, side="right") - less
    return float(np.mean((less + 0.5 * equal) / len(negative_sorted)))


def low_fpr_metrics(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    *,
    max_fpr: float = 0.05,
) -> dict[str, float]:
    positive = np.asarray(positive_scores, dtype=np.float64)
    negative = np.asarray(negative_scores, dtype=np.float64)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("both positive and negative scores are required")
    threshold = select_threshold(negative, max_fpr=max_fpr)
    _thresholds, tpr, fpr = _roc_arrays(positive, negative)
    return {
        "threshold": threshold,
        "tpr_at_fpr_0_05": float(np.mean(positive >= threshold)),
        "fpr": float(np.mean(negative >= threshold)),
        "low_fpr_pauc_normalized": _partial_auc(fpr, tpr, max_fpr),
        "ranking_probability": _ranking_probability(positive, negative),
        "auroc": _ranking_probability(positive, negative),
    }
