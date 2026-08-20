"""Sample-level overlap, topology, and stress metrics for H1."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from trace_extraction.skeleton import skeletonize_mask


def _ratio(numerator: float, denominator: float, empty: float) -> float:
    return float(numerator / denominator) if denominator else float(empty)


def _cldice(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_skeleton = skeletonize_mask(prediction); target_skeleton = skeletonize_mask(target)
    precision = _ratio(int((pred_skeleton & target).sum()), int(pred_skeleton.sum()), 1.0 if not target_skeleton.any() else 0.0)
    recall = _ratio(int((target_skeleton & prediction).sum()), int(target_skeleton.sum()), 1.0)
    return _ratio(2 * precision * recall, precision + recall, 1.0)


def _skeleton_f1(prediction: np.ndarray, target: np.ndarray, tolerance: float = 2.0) -> float:
    pred = skeletonize_mask(prediction); truth = skeletonize_mask(target)
    precision = _ratio(int((pred & (distance_transform_edt(~truth) <= tolerance)).sum()), int(pred.sum()), 1.0 if not truth.any() else 0.0)
    recall = _ratio(int((truth & (distance_transform_edt(~pred) <= tolerance)).sum()), int(truth.sum()), 1.0)
    return _ratio(2 * precision * recall, precision + recall, 1.0)


def _component_count(mask: np.ndarray, minimum_pixels: int = 3) -> int:
    components, count = label(skeletonize_mask(mask), structure=np.ones((3, 3), dtype=np.uint8))
    return sum(int((components == index).sum()) >= minimum_pixels for index in range(1, count + 1))


def _fragmentation(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_count = _component_count(prediction); target_count = _component_count(target)
    return float(max(pred_count - target_count, 0) / max(target_count, 1))


def _branch_preservation(prediction: np.ndarray, sample: dict[str, Any]) -> float:
    distance = distance_transform_edt(~prediction)
    recalls = []
    for centerline in np.asarray(sample["branch_centerlines"], dtype=bool):
        recalls.append(_ratio(int((centerline & (distance <= 2.0)).sum()), int(centerline.sum()), 1.0))
    return float(np.mean(recalls)) if recalls else 1.0


def _parallel_false_connection(prediction: np.ndarray, sample: dict[str, Any]) -> float:
    instances = np.asarray(sample["instance_masks"], dtype=bool)
    if len(instances) < 2:
        return 0.0
    components, count = label(prediction, structure=np.ones((3, 3), dtype=np.uint8))
    for index in range(1, count + 1):
        component = components == index
        touched = sum(int((component & instance).sum()) >= 3 for instance in instances)
        if touched >= 2:
            return 1.0
    return 0.0


def sample_metrics(probability: np.ndarray, sample: dict[str, Any], threshold: float) -> dict[str, float | int | str]:
    prediction = np.asarray(probability) >= float(threshold)
    target = np.asarray(sample["visible_fault_mask"], dtype=bool)
    intersection = int((prediction & target).sum()); pred_count = int(prediction.sum()); target_count = int(target.sum())
    metrics: dict[str, float | int | str] = {
        "split": sample["split"], "index": int(sample["index"]), "case": sample["case"],
        "dice": _ratio(2 * intersection, pred_count + target_count, 1.0),
        "precision": _ratio(intersection, pred_count, 0.0 if target_count else 1.0),
        "recall": _ratio(intersection, target_count, 1.0),
        "cldice": _cldice(prediction, target),
        "skeleton_f1": _skeleton_f1(prediction, target),
        "fragmentation": _fragmentation(prediction, target),
        "branch_preservation": _branch_preservation(prediction, sample),
        "parallel_false_connection": _parallel_false_connection(prediction, sample) if sample["case"] in {"close_parallel", "history_confuser"} else 0.0,
        "predicted_foreground_fraction": float(prediction.mean()),
    }
    if not all(math.isfinite(float(value)) for key, value in metrics.items() if key not in {"split", "case"}):
        raise ValueError("non-finite H1 metric")
    return metrics


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ["dice", "precision", "recall", "cldice", "skeleton_f1", "fragmentation", "branch_preservation", "parallel_false_connection", "predicted_foreground_fraction"]
    overall = {name: float(np.mean([float(row[name]) for row in rows])) for name in metric_names}
    by_case = {}
    for case in sorted({str(row["case"]) for row in rows}):
        selected = [row for row in rows if row["case"] == case]
        by_case[case] = {name: float(np.mean([float(row[name]) for row in selected])) for name in metric_names}
        by_case[case]["count"] = len(selected)
    return {"overall": overall, "by_case": by_case, "sample_count": len(rows)}


def evaluate(probabilities: list[np.ndarray], samples: list[dict[str, Any]], threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(probabilities) != len(samples):
        raise ValueError("probability/sample count mismatch")
    rows = [sample_metrics(probability, sample, threshold) for probability, sample in zip(probabilities, samples, strict=True)]
    return aggregate(rows), rows
