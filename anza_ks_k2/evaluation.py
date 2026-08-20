"""Train-calibrated scene-level K2 evaluation and frozen gates."""

from __future__ import annotations

from typing import Any

import numpy as np

from anza_fs.metrics import _cldice, _fragmentation
from anza_ks.stats.low_fpr import threshold_at_fpr, tpr_at_fpr_curve


GRID = np.linspace(0.05, 0.95, 37)


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def _pixel_row(probability: np.ndarray, sample: dict[str, Any], threshold: float) -> dict[str, float]:
    prediction = np.asarray(probability) >= threshold; target = np.asarray(sample["target"], dtype=bool)
    intersection = int((prediction & target).sum()); predicted = int(prediction.sum()); truth = int(target.sum())
    return {
        "dice": float(2 * intersection / max(predicted + truth, 1)),
        "precision": float(intersection / max(predicted, 1)),
        "recall": float(intersection / max(truth, 1)),
        "cldice": _cldice(prediction, target),
        "fragmentation": _fragmentation(prediction, target),
        "foreground_fraction": float(prediction.mean()),
    }


def pixel_summary(probabilities: list[np.ndarray], samples: list[dict[str, Any]], threshold: float) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows = [_pixel_row(probability, sample, threshold) for probability, sample in zip(probabilities, samples, strict=True)]
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}, rows


def calibration_curve(probabilities: list[np.ndarray], samples: list[dict[str, Any]]) -> list[dict[str, float]]:
    return [{"threshold": float(threshold), **pixel_summary(probabilities, samples, float(threshold))[0]} for threshold in GRID]


def select_threshold(curve: list[dict[str, float]], metric: str, target: float | None = None) -> float:
    if target is None:
        return float(max(curve, key=lambda row: (row[metric], row["threshold"]))["threshold"])
    return float(min(curve, key=lambda row: (abs(row[metric] - target), -row["threshold"]))["threshold"])


def structure_scores(probabilities: list[np.ndarray], samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    target_scores = []; distractor_scores = []
    for probability, sample in zip(probabilities, samples, strict=True):
        target = np.asarray(sample["target"], dtype=bool); distractor = np.asarray(sample["distractor"], dtype=bool)
        target_scores.append(float(np.mean(probability[target])))
        distractor_scores.append(float(np.mean(probability[distractor])) if distractor.any() else 0.0)
    return np.asarray(target_scores), np.asarray(distractor_scores)


def recall95_threshold(target_scores: np.ndarray) -> float:
    ordered = np.sort(np.asarray(target_scores))
    return float(ordered[int(np.floor(0.05 * len(ordered)))])


def structural_summary(target_scores: np.ndarray, distractor_scores: np.ndarray, threshold: float) -> dict[str, Any]:
    target_accept = target_scores >= threshold; false_accept = distractor_scores >= threshold
    curve_tpr, curve_fpr, curve_threshold = tpr_at_fpr_curve(target_scores, distractor_scores, 0.05)
    return {
        "threshold": float(threshold),
        "target_recall": float(target_accept.mean()),
        "false_positive_rate": float(false_accept.mean()),
        "true_positive_count": int(target_accept.sum()),
        "false_positive_count": int(false_accept.sum()),
        "scene_count": len(target_scores),
        "tpr_at_fpr05": curve_tpr,
        "realized_fpr_at_curve": curve_fpr,
        "curve_threshold": curve_threshold,
        "false_accept_indicators": false_accept.astype(np.uint8),
    }


def paired_bootstrap_improvement(control: np.ndarray, candidate: np.ndarray, *, resamples: int = 10_000, seed: int = 2_019_451) -> dict[str, float | int]:
    improvement = np.asarray(control, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)
    rng = np.random.default_rng(seed); estimates = np.empty(resamples)
    for start in range(0, resamples, 1000):
        count = min(1000, resamples - start); indices = rng.integers(0, len(improvement), size=(count, len(improvement)))
        estimates[start : start + count] = improvement[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {"mean_improvement": float(improvement.mean()), "ci95_lower": float(lower), "ci95_upper": float(upper), "resamples": resamples, "unit": "independent_scene"}


def apply_gates(variants: dict[str, Any], matched: dict[str, Any], bootstraps: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    m1 = variants["M1_static"]; m2 = variants["M2_shear_ks"]; m3 = variants["M3_cat_raw"]; m4 = variants["M4_anza_ks"]
    dice_safe_m1 = m4["natural_primary"]["dice"] >= m1["natural_primary"]["dice"] - 0.005
    mechanism_m1 = m4["mechanism"]["false_positive_rate"] <= 0.70 * m1["mechanism"]["false_positive_rate"]
    natural_topology = matched["M1_static"]["cldice_delta"] >= 0.010 or matched["M1_static"]["fragmentation_ratio"] is not None and matched["M1_static"]["fragmentation_ratio"] <= 0.90
    practical = bool(dice_safe_m1 and mechanism_m1 and natural_topology)
    dice_safe_m3 = m4["natural_primary"]["dice"] >= m3["natural_primary"]["dice"] - 0.005
    fpr_m3 = m4["mechanism"]["false_positive_rate"] <= 0.80 * m3["mechanism"]["false_positive_rate"]
    tpr_m3 = m4["mechanism"]["tpr_at_fpr05"] - m3["mechanism"]["tpr_at_fpr05"] >= 0.08
    kolmogorov = bool(dice_safe_m3 and (fpr_m3 or tpr_m3))
    anosov_sign = m4["mechanism"]["false_positive_rate"] < m2["mechanism"]["false_positive_rate"]
    anosov_ci = bootstraps["M4_vs_M2"]["ci95_lower"] > 0
    anosov = bool(anosov_sign and anosov_ci)
    if practical and kolmogorov and anosov:
        status = "ANZA_KS_K2_STRONG_PASS"
    elif practical and kolmogorov:
        status = "ANZA_KS_K2_SYMBOLIC_PASS_ANOSOV_UNRESOLVED"
    else:
        status = "STOP_ANZA_KS_FEATURE_NOT_TRANSFERRED"
    return status, {"practical_M4_vs_M1": practical, "kolmogorov_M4_vs_M3": kolmogorov, "anosov_M4_vs_M2": anosov, "pixel_safety_M1": dice_safe_m1, "mechanism_M1": mechanism_m1, "natural_topology_M1": natural_topology, "anosov_positive_ci": anosov_ci}
