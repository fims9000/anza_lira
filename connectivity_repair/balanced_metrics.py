"""Balanced matched-pair metrics for structural connectivity predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        selected = (probabilities >= left) & (
            probabilities < right if right < 1.0 else probabilities <= right
        )
        if selected.any():
            value += float(selected.mean()) * abs(
                float(labels[selected].mean()) - float(probabilities[selected].mean())
            )
    return float(value)


def balanced_matched_pair_metrics(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Score one hard negative for each positive continuation.

    The input arrays are pair-aligned: element ``i`` in each array is the
    matched positive/negative comparison.  Natural edge prevalence is never
    used by this endpoint.
    """

    positive = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    negative = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    if not len(positive) or len(positive) != len(negative):
        raise ValueError("balanced connectivity metrics require equal non-empty pairs")
    scores = np.concatenate((positive, negative))
    if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("connectivity scores must be finite probabilities in [0, 1]")
    labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
    ranking = np.mean((positive > negative).astype(float) + 0.5 * (positive == negative))
    return {
        "pair_count": int(len(positive)),
        "balanced_sample_count": int(len(scores)),
        "prevalence": 0.5,
        "auroc": float(roc_auc_score(labels, scores)),
        "balanced_auprc": float(average_precision_score(labels, scores)),
        "matched_pair_ranking_probability": float(ranking),
        "matched_pair_accuracy": float(np.mean(positive > negative)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, scores >= float(threshold))),
        "threshold": float(threshold),
        "brier": float(brier_score_loss(labels, scores)),
        "ece_10bin": _ece(labels, scores),
    }

