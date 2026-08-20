"""Pair-level safety and segmentation metrics for frozen ANZA-FS H3."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from trace_extraction.skeleton import skeletonize_mask


CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)


def _ratio(numerator: float, denominator: float, empty: float) -> float:
    return float(numerator / denominator) if denominator else float(empty)


def connected_anchor_event(prediction: np.ndarray, anchors: np.ndarray) -> bool:
    prediction = np.asarray(prediction, dtype=bool)
    anchors = np.asarray(anchors, dtype=bool)
    if prediction.ndim != 2 or anchors.shape != (2, *prediction.shape):
        raise ValueError("prediction must be HxW and anchors 2xHxW")
    components, count = label(prediction, structure=CONNECTIVITY)
    for component_id in range(1, count + 1):
        component = components == component_id
        if (component & anchors[0]).any() and (component & anchors[1]).any():
            return True
    return False


def false_bridge_event(prediction: np.ndarray, sample: dict[str, Any]) -> bool:
    return connected_anchor_event(prediction, np.asarray(sample["negative_anchor_masks"]))


def positive_continuation_event(prediction: np.ndarray, sample: dict[str, Any]) -> bool:
    return connected_anchor_event(prediction, np.asarray(sample["positive_anchor_masks"]))


def branch_recall(prediction: np.ndarray, sample: dict[str, Any], tolerance: float = 2.0) -> float:
    distance = distance_transform_edt(~np.asarray(prediction, dtype=bool))
    values = []
    for centerline in np.asarray(sample["branch_centerlines"], dtype=bool):
        values.append(_ratio(int((centerline & (distance <= tolerance)).sum()), int(centerline.sum()), 1.0))
    return float(np.mean(values)) if values else 1.0


def _cldice(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_skeleton = skeletonize_mask(prediction)
    target_skeleton = skeletonize_mask(target)
    precision = _ratio(int((pred_skeleton & target).sum()), int(pred_skeleton.sum()), 1.0 if not target_skeleton.any() else 0.0)
    recall = _ratio(int((target_skeleton & prediction).sum()), int(target_skeleton.sum()), 1.0)
    return _ratio(2.0 * precision * recall, precision + recall, 1.0)


def _skeleton_f1(prediction: np.ndarray, target: np.ndarray, tolerance: float = 2.0) -> float:
    pred = skeletonize_mask(prediction)
    truth = skeletonize_mask(target)
    precision = _ratio(int((pred & (distance_transform_edt(~truth) <= tolerance)).sum()), int(pred.sum()), 1.0 if not truth.any() else 0.0)
    recall = _ratio(int((truth & (distance_transform_edt(~pred) <= tolerance)).sum()), int(truth.sum()), 1.0)
    return _ratio(2.0 * precision * recall, precision + recall, 1.0)


def _component_count(mask: np.ndarray, minimum_pixels: int = 3) -> int:
    components, count = label(skeletonize_mask(mask), structure=CONNECTIVITY)
    return sum(int((components == component_id).sum()) >= minimum_pixels for component_id in range(1, count + 1))


def _fragmentation(prediction: np.ndarray, target: np.ndarray) -> float:
    target_components = _component_count(target)
    return float(max(_component_count(prediction) - target_components, 0) / max(target_components, 1))


def sample_metrics(probability: np.ndarray, sample: dict[str, Any], threshold: float) -> dict[str, Any]:
    prediction = np.asarray(probability) >= float(threshold)
    target = np.asarray(sample["visible_fault_mask"], dtype=bool)
    intersection = int((prediction & target).sum())
    pred_count = int(prediction.sum())
    target_count = int(target.sum())
    false_bridge = int(false_bridge_event(prediction, sample))
    positive_connected = int(positive_continuation_event(prediction, sample))
    row = {
        "split": sample["split"],
        "index": int(sample["index"]),
        "case": sample["case"],
        "dice": _ratio(2 * intersection, pred_count + target_count, 1.0),
        "precision": _ratio(intersection, pred_count, 0.0 if target_count else 1.0),
        "recall": _ratio(intersection, target_count, 1.0),
        "cldice": _cldice(prediction, target),
        "skeleton_f1": _skeleton_f1(prediction, target),
        "fragmentation": _fragmentation(prediction, target),
        "branch_recall": branch_recall(prediction, sample),
        "false_bridge": false_bridge,
        "negative_event_count": 1,
        "positive_connected": positive_connected,
        "positive_event_count": 1,
        "predicted_foreground_fraction": float(prediction.mean()),
    }
    if not all(math.isfinite(float(value)) for key, value in row.items() if key not in {"split", "case"}):
        raise ValueError("non-finite H3 metric")
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty H3 rows")
    means = ("dice", "precision", "recall", "cldice", "skeleton_f1", "fragmentation", "branch_recall", "predicted_foreground_fraction")
    overall = {name: float(np.mean([float(row[name]) for row in rows])) for name in means}
    false_count = int(sum(int(row["false_bridge"]) for row in rows))
    negative_count = int(sum(int(row["negative_event_count"]) for row in rows))
    positive_count = int(sum(int(row["positive_connected"]) for row in rows))
    positive_total = int(sum(int(row["positive_event_count"]) for row in rows))
    overall.update({
        "false_bridge_rate": _ratio(false_count, negative_count, 0.0),
        "false_bridge_count": false_count,
        "negative_event_count": negative_count,
        "positive_continuation_rate": _ratio(positive_count, positive_total, 1.0),
        "positive_connected_count": positive_count,
        "positive_event_count": positive_total,
    })
    by_case: dict[str, Any] = {}
    for case in sorted({str(row["case"]) for row in rows}):
        selected = [row for row in rows if row["case"] == case]
        by_case[case] = aggregate(selected)["overall"] if len(selected) != len(rows) else overall
        by_case[case]["sample_count"] = len(selected)
    return {"overall": overall, "by_case": by_case, "sample_count": len(rows)}


def evaluate(probabilities: list[np.ndarray], samples: list[dict[str, Any]], threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(probabilities) != len(samples):
        raise ValueError("probability/sample count mismatch")
    rows = [sample_metrics(probability, sample, threshold) for probability, sample in zip(probabilities, samples, strict=True)]
    return aggregate(rows), rows


def select_recall95_threshold(curve: list[dict[str, Any]], minimum_recall: float = 0.95) -> dict[str, Any]:
    eligible = [row for row in curve if float(row["branch_recall"]) >= minimum_recall]
    if not eligible:
        return {**max(curve, key=lambda row: (row["branch_recall"], row["threshold"])), "recall95_achieved": False}
    return {**max(eligible, key=lambda row: row["threshold"]), "recall95_achieved": True}


def select_matched_threshold(curve: list[dict[str, Any]], metric: str, target: float) -> dict[str, Any]:
    if metric not in {"dice", "precision"}:
        raise ValueError("matching is allowed only for Dice or precision")
    return min(curve, key=lambda row: (abs(float(row[metric]) - float(target)), -float(row["branch_recall"]), -float(row["threshold"])))
