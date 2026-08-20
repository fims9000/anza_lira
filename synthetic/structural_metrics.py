"""Claim-safe observed-segmentation and latent-completion metrics."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import label

from trace_extraction.metrics import compute_trace_metrics
from trace_extraction.skeleton import skeletonize_mask


def _safe_ratio(numerator: float, denominator: float, *, empty_value: float) -> float:
    return float(numerator / denominator) if denominator else float(empty_value)


def _segmentation_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    intersection = int(np.logical_and(pred, target).sum())
    pred_count = int(pred.sum())
    target_count = int(target.sum())
    union = int(np.logical_or(pred, target).sum())
    return {
        "visible_dice": _safe_ratio(2 * intersection, pred_count + target_count, empty_value=1.0),
        "visible_iou": _safe_ratio(intersection, union, empty_value=1.0),
        "visible_precision": _safe_ratio(intersection, pred_count, empty_value=1.0 if target_count == 0 else 0.0),
        "visible_recall": _safe_ratio(intersection, target_count, empty_value=1.0),
        "visible_cldice": _cldice(pred, target),
    }


def _cldice(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred_skeleton = skeletonize_mask(pred_mask)
    target_skeleton = skeletonize_mask(target_mask)
    topology_precision = _safe_ratio(
        int(np.logical_and(pred_skeleton, target_mask).sum()),
        int(pred_skeleton.sum()),
        empty_value=1.0 if not target_skeleton.any() else 0.0,
    )
    topology_sensitivity = _safe_ratio(
        int(np.logical_and(target_skeleton, pred_mask).sum()),
        int(target_skeleton.sum()),
        empty_value=1.0,
    )
    return _safe_ratio(
        2.0 * topology_precision * topology_sensitivity,
        topology_precision + topology_sensitivity,
        empty_value=1.0,
    )


def _connected_instance_masks(mask: np.ndarray) -> np.ndarray:
    components, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros((0, *mask.shape), dtype=bool)
    return np.stack([components == index for index in range(1, count + 1)], axis=0)


def _instance_errors(
    predicted_instances: np.ndarray,
    target_instances: np.ndarray,
    overlap_threshold: float,
) -> tuple[float, float, float]:
    predicted = np.asarray(predicted_instances, dtype=bool)
    target = np.asarray(target_instances, dtype=bool)
    if predicted.ndim != 3 or target.ndim != 3 or predicted.shape[1:] != target.shape[1:]:
        raise ValueError("Predicted and latent target instance masks must be NxHxW and share HxW")
    significant = np.zeros((len(predicted), len(target)), dtype=bool)
    for pred_index, pred_mask in enumerate(predicted):
        for target_index, target_mask in enumerate(target):
            overlap = int(np.logical_and(pred_mask, target_mask).sum())
            significant[pred_index, target_index] = overlap / max(int(target_mask.sum()), 1) >= overlap_threshold
    matched_predicted = significant.any(axis=1) if len(predicted) else np.zeros(0, dtype=bool)
    merged = int(np.sum((significant.sum(axis=1) > 1) & matched_predicted)) if len(predicted) else 0
    false_merge_rate = _safe_ratio(merged, int(matched_predicted.sum()), empty_value=0.0)
    split_counts = significant.sum(axis=0) if len(predicted) else np.zeros(len(target), dtype=int)
    excess_splits = np.maximum(split_counts - 1, 0)
    false_split_rate = float(excess_splits.mean()) if len(target) else 0.0
    return false_merge_rate, false_split_rate, false_split_rate


def _eligible_relation_pairs(target: Mapping[str, Any]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    branch_ids = [int(value) for value in target["branch_ids"]]
    branch_index = {branch_id: index for index, branch_id in enumerate(branch_ids)}
    eligible: set[tuple[int, int]] = set()
    x_eligible: set[tuple[int, int]] = set()
    for junction in target["junctions"]:
        local = [branch_index[int(value)] for value in junction["incident_branch_ids"]]
        pairs = {tuple(sorted(pair)) for pair in combinations(local, 2)}
        eligible.update(pairs)
        if junction["junction_type"] == "x_crossing":
            x_eligible.update(pairs)
    return eligible, x_eligible


def _relation_metrics(predicted: np.ndarray | None, target: Mapping[str, Any]) -> dict[str, float | int]:
    truth = np.asarray(target["continuation_relation_matrix"], dtype=bool)
    eligible, x_eligible = _eligible_relation_pairs(target)
    true_pairs = {tuple(map(int, pair)) for pair in np.argwhere(np.triu(truth, k=1))}
    if predicted is None:
        selected_pairs: set[tuple[int, int]] = set()
    else:
        scores = np.asarray(predicted)
        if scores.shape != truth.shape or not np.isfinite(scores).all():
            raise ValueError("Predicted continuation scores must be finite and match the target relation matrix")
        selected_pairs = {
            pair for pair in eligible if float(scores[pair[0], pair[1]]) >= 0.5
        }
    true_positive = len(selected_pairs & true_pairs)
    false_positive = len(selected_pairs - true_pairs)
    precision = _safe_ratio(true_positive, len(selected_pairs), empty_value=1.0 if not true_pairs else 0.0)
    recall = _safe_ratio(true_positive, len(true_pairs), empty_value=1.0)
    f1 = _safe_ratio(
        2.0 * precision * recall,
        precision + recall,
        empty_value=1.0 if not true_pairs else 0.0,
    )
    identity_switch = _safe_ratio(false_positive, len(true_pairs), empty_value=0.0)
    x_true = true_pairs & x_eligible
    x_selected = selected_pairs & x_eligible
    x_accuracy = _safe_ratio(len(x_true & x_selected), len(x_true), empty_value=1.0)
    return {
        "continuation_precision": precision,
        "continuation_recall": recall,
        "branch_continuation_f1": f1,
        "continuation_relation_count": len(true_pairs),
        "branch_pairing_accuracy": x_accuracy,
        "branch_pairing_count": len(x_true),
        "identity_switch_rate": identity_switch,
    }


def _disk_labels(components: np.ndarray, point_xy: list[float], radius: int = 3) -> set[int]:
    x, y = int(round(point_xy[0])), int(round(point_xy[1]))
    yy, xx = np.ogrid[: components.shape[0], : components.shape[1]]
    selected = (yy - y) ** 2 + (xx - x) ** 2 <= radius**2
    return {int(value) for value in np.unique(components[selected]) if int(value) != 0}


def _false_bridge_rate(
    predicted_completion: np.ndarray,
    target: Mapping[str, Any],
    coverage_threshold: float,
) -> tuple[float, int, int]:
    records = [record for record in target["gaps"] if record["gap_type"] == "negative"]
    masks = np.asarray(target["negative_gap_masks"], dtype=bool)
    if len(records) != len(masks):
        raise ValueError("Negative gap records and masks must have equal length")
    components, _ = label(predicted_completion, structure=np.ones((3, 3), dtype=np.uint8))
    bridge_count = 0
    for record, gap_mask in zip(records, masks):
        coverage = _safe_ratio(
            int(np.logical_and(predicted_completion, gap_mask).sum()),
            int(gap_mask.sum()),
            empty_value=0.0,
        )
        endpoints = record["endpoint_xy"]
        shared_components = _disk_labels(components, endpoints[0]) & _disk_labels(components, endpoints[1])
        bridge_count += int(coverage >= coverage_threshold and bool(shared_components))
    return _safe_ratio(bridge_count, len(records), empty_value=0.0), bridge_count, len(records)


def _orientation_error(
    predicted_orientation: np.ndarray | None,
    branch_centerlines: np.ndarray,
    target_cos2: np.ndarray,
    target_sin2: np.ndarray,
) -> float:
    if predicted_orientation is None:
        return 90.0 if np.any(branch_centerlines) else 0.0
    predicted = np.asarray(predicted_orientation, dtype=np.float64)
    if predicted.ndim == 2:
        if predicted.shape != branch_centerlines.shape[1:]:
            raise ValueError("Scalar predicted orientation must be HxW")
        predicted = np.broadcast_to(predicted, branch_centerlines.shape)
    if predicted.shape != branch_centerlines.shape or not np.isfinite(predicted).all():
        raise ValueError("Mode-resolved predicted orientation must be finite and match BxHxW branches")
    errors: list[np.ndarray] = []
    for predicted_branch, mask, cos2, sin2 in zip(predicted, branch_centerlines, target_cos2, target_sin2):
        selected = np.asarray(mask, dtype=bool)
        if selected.any():
            target_angle = 0.5 * np.arctan2(sin2[selected], cos2[selected])
            delta = predicted_branch[selected] - target_angle
            axial = 0.5 * np.arccos(np.clip(np.cos(2.0 * delta), -1.0, 1.0))
            errors.append(np.degrees(axial))
    return float(np.median(np.concatenate(errors))) if errors else 0.0


def compute_structural_metrics(
    predicted_visible_mask: np.ndarray,
    target: Mapping[str, Any],
    *,
    predicted_completion_mask: np.ndarray | None = None,
    predicted_instance_masks: np.ndarray | None = None,
    predicted_continuation_scores: np.ndarray | None = None,
    predicted_orientation: np.ndarray | None = None,
    overlap_threshold: float = 0.20,
    bridge_coverage_threshold: float = 0.50,
) -> dict[str, float | int]:
    """Evaluate observed evidence and latent structure without conflating them."""
    visible_pred = np.asarray(predicted_visible_mask, dtype=bool)
    visible_truth = np.asarray(target["visible_fault_mask"], dtype=bool)
    latent_truth = np.asarray(target["latent_fault_mask"], dtype=bool)
    completion_pred = visible_pred if predicted_completion_mask is None else np.asarray(predicted_completion_mask, dtype=bool)
    if visible_pred.ndim != 2 or visible_pred.shape != visible_truth.shape:
        raise ValueError("Visible prediction and visible target must be matching HxW arrays")
    if completion_pred.ndim != 2 or completion_pred.shape != latent_truth.shape:
        raise ValueError("Completion prediction and latent target must be matching HxW arrays")

    completion_skeleton = skeletonize_mask(completion_pred)
    latent_skeleton = skeletonize_mask(latent_truth)
    trace = compute_trace_metrics(completion_skeleton, latent_skeleton, tolerance=2.0, node_tolerance=3.0)
    instances = (
        _connected_instance_masks(completion_skeleton)
        if predicted_instance_masks is None
        else np.asarray(predicted_instance_masks, dtype=bool)
    )
    false_merge, false_split, fragmentation = _instance_errors(
        instances,
        np.asarray(target["instance_masks"], dtype=bool),
        overlap_threshold,
    )
    positive_gap = np.asarray(target["positive_gap_mask"], dtype=bool)
    gap_recovery = _safe_ratio(
        int(np.logical_and(completion_pred, positive_gap).sum()),
        int(positive_gap.sum()),
        empty_value=1.0,
    )
    false_bridge, false_bridge_count, negative_gap_count = _false_bridge_rate(
        completion_pred,
        target,
        bridge_coverage_threshold,
    )
    metrics: dict[str, float | int] = {
        **_segmentation_metrics(visible_pred, visible_truth),
        "latent_cldice": _cldice(completion_pred, latent_truth),
        "latent_skeleton_f1_2px": float(trace["trace_f1"]),
        "orientation_error_median_deg": _orientation_error(
            predicted_orientation,
            np.asarray(target["branch_centerlines"], dtype=bool),
            np.asarray(target["branch_tangent_cos2"], dtype=np.float32),
            np.asarray(target["branch_tangent_sin2"], dtype=np.float32),
        ),
        "junction_f1": float(trace["junction_f1"]),
        "endpoint_f1": float(trace["endpoint_f1"]),
        **_relation_metrics(predicted_continuation_scores, target),
        "false_merge_rate": false_merge,
        "false_split_rate": false_split,
        "gap_recovery_rate": gap_recovery,
        "positive_gap_count": len(np.asarray(target["positive_gap_masks"])),
        "false_bridge_rate": false_bridge,
        "false_bridge_count": false_bridge_count,
        "negative_gap_count": negative_gap_count,
        "fragmentation": fragmentation,
        "symmetric_skeleton_distance": float(trace["symmetric_skeleton_distance"]),
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("Structural metrics produced NaN or Inf")
    return metrics
