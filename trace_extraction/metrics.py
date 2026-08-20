"""Pixel-distance, node, orientation, and length metrics for trace skeletons."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .geometry import axial_distance, local_pca_orientation
from .graph import extract_trace_graph


def _safe_f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _distance_precision_recall(pred: np.ndarray, target: np.ndarray, tolerance: float) -> tuple[float, float]:
    pred_count, target_count = int(pred.sum()), int(target.sum())
    if pred_count == 0 and target_count == 0:
        return 1.0, 1.0
    if target_count:
        pred_hits = int((pred & (distance_transform_edt(~target) <= tolerance)).sum())
        precision = pred_hits / pred_count if pred_count else 0.0
    else:
        precision = 0.0
    if pred_count:
        target_hits = int((target & (distance_transform_edt(~pred) <= tolerance)).sum())
        recall = target_hits / target_count if target_count else 0.0
    else:
        recall = 0.0
    return precision, recall


def _point_f1(pred_points: np.ndarray, target_points: np.ndarray, tolerance: float) -> float:
    if len(pred_points) == 0 and len(target_points) == 0:
        return 1.0
    if len(pred_points) == 0 or len(target_points) == 0:
        return 0.0
    distances = np.linalg.norm(pred_points[:, None, :] - target_points[None, :, :], axis=2)
    pred_indices, target_indices = linear_sum_assignment(distances)
    matches = int(np.sum(distances[pred_indices, target_indices] <= tolerance))
    precision = matches / len(pred_points)
    recall = matches / len(target_points)
    return _safe_f1(precision, recall)


def _junction_centers(graph, *, evaluable_only: bool = False) -> np.ndarray:
    return np.asarray(
        [
            [np.mean([point[0] for point in component]), np.mean([point[1] for point in component])]
            for index, component in enumerate(graph.junctions)
            if not evaluable_only or not graph.junction_border_truncated[index]
        ],
        dtype=np.float64,
    ).reshape(-1, 2)


def _symmetric_distance(pred: np.ndarray, target: np.ndarray) -> float:
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float(math.hypot(*pred.shape))
    return 0.5 * (float(distance_transform_edt(~target)[pred].mean()) + float(distance_transform_edt(~pred)[target].mean()))


def _total_length(graph) -> float:
    return float(sum(segment.pixel_length for segment in graph.segments))


def compute_trace_metrics(
    predicted_skeleton: np.ndarray,
    target_skeleton: np.ndarray,
    *,
    pred_orientation: np.ndarray | None = None,
    tolerance: float = 2.0,
    node_tolerance: float = 3.0,
    pca_radius: int = 5,
    border_margin: int = 5,
) -> dict[str, float | int]:
    pred = np.asarray(predicted_skeleton, dtype=bool)
    target = np.asarray(target_skeleton, dtype=bool)
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError(f"Trace skeleton shapes must match in 2-D, got {pred.shape} and {target.shape}")
    pred_graph = extract_trace_graph(pred, border_margin=border_margin)
    target_graph = extract_trace_graph(target, border_margin=border_margin)
    precision, recall = _distance_precision_recall(pred, target, tolerance)
    pred_endpoints = np.asarray(
        [point for point, truncated in zip(pred_graph.endpoints, pred_graph.endpoint_border_truncated) if not truncated],
        dtype=np.float64,
    ).reshape(-1, 2)
    target_endpoints = np.asarray(
        [point for point, truncated in zip(target_graph.endpoints, target_graph.endpoint_border_truncated) if not truncated],
        dtype=np.float64,
    ).reshape(-1, 2)

    gt_orientation = local_pca_orientation(target, radius=pca_radius)
    matched_errors: list[float] = []
    if pred_orientation is not None and pred.any() and target.any():
        pred_orientation = np.asarray(pred_orientation, dtype=np.float64)
        if pred_orientation.shape != pred.shape or not np.isfinite(pred_orientation).all():
            raise ValueError("Predicted orientation must be finite and match the skeleton shape")
        target_points = np.argwhere(target)
        tree = cKDTree(target_points)
        for y, x in np.argwhere(pred):
            distance, target_index = tree.query((y, x), k=1)
            if distance <= tolerance:
                ty, tx = target_points[int(target_index)]
                matched_errors.append(math.degrees(float(axial_distance(pred_orientation[y, x], gt_orientation[ty, tx]))))
    if matched_errors:
        error_mean = float(np.mean(matched_errors))
        error_median = float(np.median(matched_errors))
        error_p90 = float(np.percentile(matched_errors, 90))
    else:
        # Axial error has a bounded worst case of 90 degrees. This explicit
        # convention keeps summary artifacts finite when no pixels are matched.
        error_mean = error_median = error_p90 = 0.0 if not pred.any() and not target.any() else 90.0

    pred_length = _total_length(pred_graph)
    target_length = _total_length(target_graph)
    length_error = abs(pred_length - target_length) / max(target_length, 1e-8) if target_length else (0.0 if not pred_length else 1.0)
    metrics: dict[str, float | int] = {
        "trace_precision": precision,
        "trace_recall": recall,
        "trace_f1": _safe_f1(precision, recall),
        "endpoint_f1": _point_f1(pred_endpoints, target_endpoints, node_tolerance),
        "junction_f1": _point_f1(
            _junction_centers(pred_graph, evaluable_only=True),
            _junction_centers(target_graph, evaluable_only=True),
            node_tolerance,
        ),
        "symmetric_skeleton_distance": _symmetric_distance(pred, target),
        "orientation_error_mean_deg": error_mean,
        "orientation_error_median_deg": error_median,
        "orientation_error_p90_deg": error_p90,
        "orientation_matched_pixel_count": len(matched_errors),
        "trace_length_error": float(length_error),
        "predicted_trace_count": len(pred_graph.segments),
        "target_trace_count": len(target_graph.segments),
        "predicted_endpoint_total": len(pred_graph.endpoints),
        "predicted_endpoint_evaluable": len(pred_endpoints),
        "predicted_endpoint_border_truncated": sum(pred_graph.endpoint_border_truncated),
        "target_endpoint_total": len(target_graph.endpoints),
        "target_endpoint_evaluable": len(target_endpoints),
        "target_endpoint_border_truncated": sum(target_graph.endpoint_border_truncated),
        "predicted_junction_total": len(pred_graph.junctions),
        "predicted_junction_evaluable": sum(not item for item in pred_graph.junction_border_truncated),
        "predicted_junction_border_truncated": sum(pred_graph.junction_border_truncated),
        "target_junction_total": len(target_graph.junctions),
        "target_junction_evaluable": sum(not item for item in target_graph.junction_border_truncated),
        "target_junction_border_truncated": sum(target_graph.junction_border_truncated),
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("Trace metric computation produced a non-finite value")
    return metrics
