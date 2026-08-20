"""Fixed-score paired metrics for ANZA-EK E1."""

from __future__ import annotations

from typing import Any

import numpy as np


def matched_ranking(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if positive.shape != negative.shape or positive.ndim != 1:
        raise ValueError("matched scores must be equal-length vectors")
    return float(np.mean((positive > negative) + 0.5 * (positive == negative)))


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))


def tpr_at_fpr(positive: np.ndarray, negative: np.ndarray, maximum_fpr: float = 0.05) -> tuple[float, float, float]:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if not 0 <= maximum_fpr < 1:
        raise ValueError("maximum_fpr must be in [0,1)")
    candidates = np.unique(np.concatenate((positive, negative)))
    thresholds = np.concatenate(([np.inf], candidates[::-1], [-np.inf]))
    best = (0.0, 0.0, float("inf"))
    for threshold in thresholds:
        fpr = float(np.mean(negative >= threshold))
        if fpr <= maximum_fpr + 1e-12:
            tpr = float(np.mean(positive >= threshold))
            if (tpr, -fpr, threshold) > (best[0], -best[1], best[2]):
                best = (tpr, fpr, float(threshold))
    return best


def fisher_separation(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    denominator = np.sqrt(0.5 * (positive.var(ddof=1) + negative.var(ddof=1))).clip(min=1e-12)
    return float((positive.mean() - negative.mean()) / denominator)


def perturbation_stability(clean: np.ndarray, perturbed: np.ndarray) -> float:
    clean = np.asarray(clean, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)
    if clean.shape != perturbed.shape or clean.ndim != 1:
        raise ValueError("clean and perturbed scores must align")
    if clean.std() <= 1e-12 or perturbed.std() <= 1e-12:
        return 1.0 if np.allclose(clean, perturbed) else 0.0
    return float(np.corrcoef(clean, perturbed)[0, 1])


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    positive = np.asarray([row["positive_score"] for row in rows])
    negative = np.asarray([row["negative_score"] for row in rows])
    positive_perturbed = np.asarray([row["positive_perturbed_score"] for row in rows])
    negative_perturbed = np.asarray([row["negative_perturbed_score"] for row in rows])
    tpr, fpr, threshold = tpr_at_fpr(positive, negative)
    perturbed_tpr, perturbed_fpr, _ = tpr_at_fpr(positive_perturbed, negative_perturbed)
    clean_all = np.concatenate((positive, negative))
    perturbed_all = np.concatenate((positive_perturbed, negative_perturbed))
    return {
        "pair_count": len(rows),
        "matched_ranking": matched_ranking(positive, negative),
        "auroc": auroc(positive, negative),
        "tpr_at_fpr05": tpr,
        "realized_fpr": fpr,
        "threshold": threshold,
        "fisher_separation": fisher_separation(positive, negative),
        "perturbed_matched_ranking": matched_ranking(positive_perturbed, negative_perturbed),
        "perturbed_tpr_at_fpr05": perturbed_tpr,
        "perturbed_realized_fpr": perturbed_fpr,
        "perturbation_score_correlation": perturbation_stability(clean_all, perturbed_all),
    }
