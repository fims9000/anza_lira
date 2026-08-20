"""Threshold-free hard-pair and natural segmentation evaluation for ANZA-KIR."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import rankdata

from anza_ks.stats.low_fpr import tpr_at_fpr_curve
from anza_ks_k2.evaluation import calibration_curve, pixel_summary, select_threshold


def pair_rows(probabilities: list[np.ndarray], samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for probability, sample in zip(probabilities, samples, strict=True):
        target = np.asarray(sample["target"], dtype=bool); distractor = np.asarray(sample["distractor"], dtype=bool)
        target_score = float(probability[target].mean()); distractor_score = float(probability[distractor].mean())
        rows.append({"target_score": target_score, "distractor_score": distractor_score, "margin": target_score - distractor_score, "pair_error": int(distractor_score >= target_score), "mechanism_task": sample.get("mechanism_task", sample["domain"]), "index": int(sample["index"])})
    return rows


def pair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = np.asarray([row["target_score"] for row in rows]); distractor = np.asarray([row["distractor_score"] for row in rows]); margins = target - distractor
    labels = np.r_[np.ones(len(target)), np.zeros(len(distractor))]; scores = np.r_[target, distractor]
    ranks = rankdata(scores); positives = len(target); negatives = len(distractor)
    auc = float((ranks[:positives].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    tpr, realized_fpr, threshold = tpr_at_fpr_curve(target, distractor, 0.05)
    per_task = {}
    for task in sorted({row["mechanism_task"] for row in rows}):
        local = [row for row in rows if row["mechanism_task"] == task]
        per_task[task] = {"count": len(local), "pair_error": float(np.mean([row["pair_error"] for row in local])), "mean_margin": float(np.mean([row["margin"] for row in local]))}
    return {"scene_count": len(rows), "pair_error": float(np.mean([row["pair_error"] for row in rows])), "mean_margin": float(margins.mean()), "pair_auc": auc, "tpr_at_fpr05": float(tpr), "realized_fpr": float(realized_fpr), "low_fpr_threshold": float(threshold), "per_task": per_task}


def paired_bootstrap(values: np.ndarray, *, resamples: int = 10_000, seed: int = 3_191_000_041) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64); rng = np.random.default_rng(seed); estimates = np.empty(resamples)
    for start in range(0, resamples, 1000):
        count = min(1000, resamples - start); index = rng.integers(0, len(values), size=(count, len(values))); estimates[start : start + count] = values[index].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"mean_improvement": float(values.mean()), "ci95_lower": float(low), "ci95_upper": float(high), "resamples": resamples, "unit": "independent_scene"}


def apply_gates(metrics: dict[str, Any], bootstraps: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    r0 = metrics["R0_static_residual"]; r1 = metrics["R1_shear_ks_residual"]; r2 = metrics["R2_cat_raw_residual"]; r3 = metrics["R3_anza_kir"]
    practical_pair = r3["hard"]["pair_error"] <= 0.70 * r0["hard"]["pair_error"]
    pixel_safe = r3["natural"]["dice"] >= r0["natural"]["dice"] - 0.005
    topology = r3["natural"]["cldice"] - r0["natural"]["cldice"] >= 0.010 or r3["natural"]["fragmentation"] <= 0.90 * r0["natural"]["fragmentation"]
    practical = bool(practical_pair and pixel_safe and topology)
    kolmogorov = bool(r3["hard"]["pair_error"] <= 0.85 * r2["hard"]["pair_error"] and bootstraps["R3_vs_R2"]["margin"]["ci95_lower"] > 0)
    anosov = bool(r3["hard"]["pair_error"] <= 0.90 * r1["hard"]["pair_error"] and bootstraps["R3_vs_R1"]["margin"]["ci95_lower"] > 0)
    if practical and kolmogorov and anosov:
        status = "ANZA_KIR_RESIDUAL_PASS"
    elif practical and kolmogorov:
        status = "ANZA_KIR_SYMBOLIC_PASS_ANOSOV_UNRESOLVED"
    else:
        status = "STOP_ANZA_LOCAL_SYMBOLIC_ARCHITECTURE"
    return status, {"practical_pair_error": practical_pair, "pixel_safety": pixel_safe, "natural_topology": topology, "practical_R3_vs_R0": practical, "kolmogorov_R3_vs_R2": kolmogorov, "anosov_R3_vs_R1": anosov}


__all__ = ["apply_gates", "calibration_curve", "pair_rows", "pair_summary", "paired_bootstrap", "pixel_summary", "select_threshold"]
