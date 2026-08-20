"""Scene- and pair-level TG2 evaluation and frozen causal gates."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from anza_ks.stats.low_fpr import tpr_at_fpr_curve

from .batch import K_MAX


def calibrate_p0_none(rows: list[dict[str, Any]]) -> float:
    candidates = np.unique(np.asarray([probability for row in rows for probability in row["pair_probabilities"]] + [0.0, 1.0]))
    def accuracy(threshold: float) -> float:
        correct = 0
        for row in rows:
            probability = np.asarray(row["pair_probabilities"]); prediction = K_MAX if probability.max(initial=0.0) < threshold else int(np.argmax(probability)); correct += prediction == row["label"]
        return correct / len(rows)
    return float(max(candidates, key=lambda threshold: (accuracy(float(threshold)), threshold)))


def _prediction(row: dict[str, Any], variant: str, p0_threshold: float) -> int:
    if variant == "P0_pair":
        probability = np.asarray(row["pair_probabilities"]); return K_MAX if probability.max(initial=0.0) < p0_threshold else int(np.argmax(probability))
    return int(np.argmax(np.asarray(row["scene_probabilities"])))


def evaluate_rows(rows: list[dict[str, Any]], variant: str, p0_threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = []; pair_rows = []
    for row in rows:
        prediction = _prediction(row, variant, p0_threshold); label = row["label"]; positive = label < K_MAX; selected_wrong = positive and prediction < K_MAX and prediction != label; none_case = not positive
        source_rows.append({"index": row["index"], "scene_type": row["scene_type"], "label": label, "prediction": prediction, "correct": int(prediction == label), "positive": int(positive), "none_case": int(none_case), "wrong_branch": int(selected_wrong), "selected_relation": int(prediction < K_MAX), "true_pair_score": float(row["pair_logits"][label]) if positive else None})
        pair_score = row["pair_logits"]
        for candidate, score in enumerate(pair_score): pair_rows.append({"index": row["index"], "scene_type": row["scene_type"], "candidate": candidate, "label": int(positive and candidate == label), "score": float(score)})
    correct = np.asarray([row["correct"] for row in source_rows]); positives = [row for row in source_rows if row["positive"]]; none_rows = [row for row in source_rows if row["none_case"]]
    none_predictions = np.asarray([row["prediction"] == K_MAX for row in source_rows]); none_truth = np.asarray([row["none_case"] for row in source_rows], dtype=bool)
    pair_labels = np.asarray([row["label"] for row in pair_rows]); pair_scores = np.asarray([row["score"] for row in pair_rows]); positives_count = int(pair_labels.sum()); negatives_count = len(pair_labels) - positives_count
    weights = np.where(pair_labels == 1, 0.5 / max(positives_count, 1), 0.5 / max(negatives_count, 1)); tpr, realized_fpr, threshold = tpr_at_fpr_curve(pair_scores[pair_labels == 1], pair_scores[pair_labels == 0], 0.05)
    parallel = [row for row in none_rows if row["scene_type"] in ("close_parallel", "parallel_gap_confuser")]; x_rows = [row for row in positives if row["scene_type"] in ("x_crossing", "acute_crossing")]
    metrics = {
        "source_count": len(source_rows), "positive_count": len(positives), "none_count": len(none_rows), "top1_none": float(correct.mean()),
        "none_precision": float((none_predictions & none_truth).sum() / max(none_predictions.sum(), 1)), "none_recall": float((none_predictions & none_truth).sum() / max(none_truth.sum(), 1)),
        "positive_continuation_recall": float(np.mean([row["correct"] for row in positives])), "wrong_branch_rate": float(np.mean([row["wrong_branch"] for row in positives])),
        "x_wrong_turn_rate": float(np.mean([row["wrong_branch"] for row in x_rows])) if x_rows else 0.0, "parallel_false_relation_rate": float(np.mean([row["selected_relation"] for row in parallel])) if parallel else 0.0,
        "pair_auroc": float(roc_auc_score(pair_labels, pair_scores)), "balanced_auprc": float(average_precision_score(pair_labels, pair_scores, sample_weight=weights)), "tpr_at_fpr05": float(tpr), "realized_fpr": float(realized_fpr), "low_fpr_threshold": float(threshold), "low_fpr_pauc": float(roc_auc_score(pair_labels, pair_scores, max_fpr=0.05)),
    }
    for row in source_rows:
        row["low_fpr_true_detected"] = int(bool(row["positive"]) and row["true_pair_score"] >= threshold)
    return metrics, source_rows, pair_rows


def paired_bootstrap(control: list[dict[str, Any]], candidate: list[dict[str, Any]], *, resamples: int = 10_000, seed: int = 4_141_000_041) -> dict[str, Any]:
    by_control = {row["index"]: row for row in control}; by_candidate = {row["index"]: row for row in candidate}; indices = sorted(by_control)
    top1 = np.asarray([by_candidate[i]["correct"] - by_control[i]["correct"] for i in indices], dtype=float)
    wrong = np.asarray([by_control[i]["wrong_branch"] - by_candidate[i]["wrong_branch"] for i in indices if by_control[i]["positive"]], dtype=float)
    low_fpr_tpr = np.asarray([by_candidate[i]["low_fpr_true_detected"] - by_control[i]["low_fpr_true_detected"] for i in indices if by_control[i]["positive"]], dtype=float)
    def interval(values: np.ndarray, offset: int) -> dict[str, Any]:
        rng = np.random.default_rng(seed + offset); estimates = np.empty(resamples)
        for start in range(0, resamples, 1000):
            count = min(1000, resamples - start); sampled = rng.integers(0, len(values), (count, len(values))); estimates[start : start + count] = values[sampled].mean(1)
        low, high = np.quantile(estimates, [0.025, 0.975]); return {"mean_improvement": float(values.mean()), "ci95_lower": float(low), "ci95_upper": float(high), "resamples": resamples, "unit": "source endpoint / independent scene"}
    return {"top1_none": interval(top1, 0), "wrong_branch": interval(wrong, 1), "low_fpr_tpr": interval(low_fpr_tpr, 2)}


def apply_gates(metrics: dict[str, Any], bootstraps: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    p0, p1, p2 = metrics["P0_pair"], metrics["P1_tracegraph"], metrics["P2_anza_tracegraph"]
    parallel_p1 = p1["parallel_false_relation_rate"] <= p0["parallel_false_relation_rate"] + 0.01
    architecture_effect = p1["tpr_at_fpr05"] - p0["tpr_at_fpr05"] >= 0.08 or p1["top1_none"] - p0["top1_none"] >= 0.08 or p1["wrong_branch_rate"] <= 0.70 * p0["wrong_branch_rate"]
    architecture = bool(architecture_effect and parallel_p1)
    parallel_p2 = p2["parallel_false_relation_rate"] <= p1["parallel_false_relation_rate"] + 0.01
    tpr_gain = p2["tpr_at_fpr05"] - p1["tpr_at_fpr05"] >= 0.05
    top1_gain = p2["top1_none"] - p1["top1_none"] >= 0.05 and bootstraps["P2_vs_P1"]["top1_none"]["ci95_lower"] > 0
    wrong_gain = p2["wrong_branch_rate"] <= 0.80 * p1["wrong_branch_rate"] and bootstraps["P2_vs_P1"]["wrong_branch"]["ci95_lower"] > 0
    tpr_gain_with_ci = tpr_gain and bootstraps["P2_vs_P1"]["low_fpr_tpr"]["ci95_lower"] > 0
    anza = bool(architecture and parallel_p2 and (top1_gain or wrong_gain or tpr_gain_with_ci))
    if not architecture: status = "STOP_TRACEGRAPH_RELATION_NO_ARCHITECTURE_GAIN"
    elif anza: status = "ANZA_TRACEGRAPH_CAUSAL_PASS"
    else: status = "TRACEGRAPH_PASS_ANZA_BIAS_NOT_INCREMENTAL"
    return status, {"p1_architecture_effect": architecture_effect, "p1_parallel_safety": parallel_p1, "p1_architecture_pass": architecture, "p2_tpr_effect": tpr_gain, "p2_tpr_effect_with_ci": tpr_gain_with_ci, "p2_top1_effect_with_ci": top1_gain, "p2_wrong_branch_effect_with_ci": wrong_gain, "p2_parallel_safety": parallel_p2, "p2_anza_pass": anza}
