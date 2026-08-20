"""Zero-training scalar-vs-mode-state oracle continuation evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation
import torch

from anza2.eval.low_fpr import low_fpr_metrics, select_threshold
from anza2.forensics.component_replacement import oracle_field_from_sample
from models.anza2.affinity import ANZA2StructuralAffinity, LOCAL8_OFFSETS
from structural.widest_path import domain_restricted_widest_path
from synthetic.crossing_trace_bench_v4 import SPLIT_SIZES_V4, generate_sample_v4

from .endpoint_modes import compatible_endpoint_modes
from .mode_state_graph import mode_state_edge_weights
from .mode_state_widest_path import mode_state_widest_path


METHODS = ("G0_scalar", "G1_mode_state")
PRIMARY_CASES = {
    "fault_with_gap", "negative_gap", "curved_fault", "near_parallel",
    "x_junction", "t_junction", "y_junction",
}


def _branch_endpoint(sample: dict[str, Any], branch_index: int, reference_xy: tuple[float, float]) -> tuple[int, int]:
    centerline = np.asarray(sample["branch_centerlines"], dtype=bool)[branch_index]
    ys, xs = np.nonzero(centerline)
    if not len(ys):
        raise ValueError("empty branch centerline")
    distance = (xs - float(reference_xy[0])) ** 2 + (ys - float(reference_xy[1])) ** 2
    selected = int(np.argmax(distance))
    return int(ys[selected]), int(xs[selected])


def _branch_endpoints(sample: dict[str, Any], branch_index: int) -> tuple[tuple[int, int], tuple[int, int]]:
    centerline = np.asarray(sample["branch_centerlines"], dtype=bool)[branch_index]
    ys, xs = np.nonzero(centerline)
    points = np.stack((ys, xs), axis=1)
    if not len(points):
        raise ValueError("empty branch centerline")
    # Farthest pair is deterministic and adequate for open generator curves.
    distances = ((points[:, None] - points[None, :]) ** 2).sum(axis=2)
    first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
    return tuple(map(int, points[first])), tuple(map(int, points[second]))


def _branch_tangent(sample: dict[str, Any], branch_index: int, point: tuple[int, int]) -> np.ndarray:
    theta = float(np.asarray(sample["gt_branch_theta"])[branch_index, point[0], point[1]])
    return np.asarray((np.cos(2.0 * theta), np.sin(2.0 * theta)), dtype=np.float64)


def _gap_tangent(sample: dict[str, Any]) -> np.ndarray:
    endpoints = np.asarray(sample["gaps"][0]["endpoint_xy"], dtype=np.float64)
    delta = endpoints[1] - endpoints[0]
    theta = float(np.arctan2(delta[1], delta[0]))
    return np.asarray((np.cos(2.0 * theta), np.sin(2.0 * theta)), dtype=np.float64)


def _gap_endpoints(sample: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int]]:
    endpoints = np.asarray(sample["gaps"][0]["endpoint_xy"], dtype=np.float64)
    return (
        (int(round(endpoints[0, 1])), int(round(endpoints[0, 0]))),
        (int(round(endpoints[1, 1])), int(round(endpoints[1, 0]))),
    )


def _pair_scores(
    field,
    scalar_relation: np.ndarray,
    state_edges: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    start_tangent: np.ndarray,
    goal_tangent: np.ndarray,
    domain: np.ndarray,
) -> dict[str, tuple[float, int]]:
    scalar_score, scalar_path = domain_restricted_widest_path(
        scalar_relation, start, goal, domain=domain, offsets=LOCAL8_OFFSETS
    )
    start_modes = compatible_endpoint_modes(field, start, start_tangent)
    goal_modes = compatible_endpoint_modes(field, goal, goal_tangent)
    if not start_modes or not goal_modes:
        raise AssertionError("oracle endpoint has no compatible active mode")
    state_score, state_path = mode_state_widest_path(
        state_edges,
        [(start[0], start[1], mode) for mode in start_modes],
        [(goal[0], goal[1], mode) for mode in goal_modes],
        domain=domain,
        offsets=LOCAL8_OFFSETS,
    )
    return {
        "G0_scalar": (float(scalar_score), len(scalar_path)),
        "G1_mode_state": (float(state_score), len(state_path)),
    }


def _append_pair(
    rows: list[dict[str, Any]],
    *,
    sample: dict[str, Any],
    task: str,
    label: int,
    pair_id: str,
    scores: dict[str, tuple[float, int]],
) -> None:
    for method, (score, path_length) in scores.items():
        rows.append({
            "split": sample["split"], "index": int(sample["index"]), "case": sample["case"],
            "task": task, "label": int(label), "pair_id": pair_id,
            "method": method, "score": score, "path_length": path_length,
        })


def _sample_rows(sample: dict[str, Any], *, device: torch.device) -> list[dict[str, Any]]:
    field, _valid = oracle_field_from_sample(sample, device=device)
    with torch.inference_mode():
        scalar = ANZA2StructuralAffinity()(field)[0].cpu().numpy().astype(np.float32)
        state = mode_state_edge_weights(field)[0].cpu().numpy().astype(np.float32)
    rows: list[dict[str, Any]] = []
    case = sample["case"]
    if case in {"fault_with_gap", "negative_gap"}:
        start, goal = _gap_endpoints(sample); tangent = _gap_tangent(sample)
        support = np.asarray(sample["latent_fault_mask"], dtype=bool) | np.asarray(sample["negative_gap_mask"], dtype=bool)
        domain = binary_dilation(support, iterations=1)
        _append_pair(
            rows, sample=sample,
            task="positive_gap" if case == "fault_with_gap" else "negative_gap",
            label=1 if case == "fault_with_gap" else 0,
            pair_id="gap:0",
            scores=_pair_scores(field, scalar, state, start, goal, tangent, tangent, domain),
        )
    elif case == "curved_fault":
        start, goal = _branch_endpoints(sample, 0)
        domain = binary_dilation(np.asarray(sample["branch_masks"])[0], iterations=1)
        _append_pair(
            rows, sample=sample, task="curved_continuation", label=1, pair_id="branch:0",
            scores=_pair_scores(
                field, scalar, state, start, goal,
                _branch_tangent(sample, 0, start), _branch_tangent(sample, 0, goal), domain,
            ),
        )
    elif case == "near_parallel":
        masks = np.asarray(sample["branch_masks"], dtype=bool)
        domain = binary_dilation(masks.any(axis=0), iterations=4)
        endpoints = [_branch_endpoints(sample, index) for index in range(2)]
        for branch_index, (start, goal) in enumerate(endpoints):
            _append_pair(
                rows, sample=sample, task="parallel_correct", label=1, pair_id=f"branch:{branch_index}",
                scores=_pair_scores(
                    field, scalar, state, start, goal,
                    _branch_tangent(sample, branch_index, start), _branch_tangent(sample, branch_index, goal), domain,
                ),
            )
        for side in range(2):
            start, goal = endpoints[0][side], endpoints[1][side]
            _append_pair(
                rows, sample=sample, task="parallel_wrong", label=0, pair_id=f"cross:{side}",
                scores=_pair_scores(
                    field, scalar, state, start, goal,
                    _branch_tangent(sample, 0, start), _branch_tangent(sample, 1, goal), domain,
                ),
            )
    elif case in {"x_junction", "t_junction", "y_junction"}:
        branch_ids = [int(value) for value in sample["branch_ids"]]
        by_id = {branch_id: index for index, branch_id in enumerate(branch_ids)}
        junction = sample["junctions"][0]
        center_xy = tuple(junction["point_xy"])
        incident = [int(value) for value in junction["incident_branch_ids"]]
        endpoints = {branch_id: _branch_endpoint(sample, by_id[branch_id], center_xy) for branch_id in incident}
        domain = binary_dilation(np.asarray(sample["branch_masks"])[[by_id[value] for value in incident]].any(axis=0), iterations=1)
        paired = set()
        for first, second in junction["continuation_relation"]:
            first, second = int(first), int(second); paired.add(frozenset((first, second)))
            _append_pair(
                rows, sample=sample,
                task="x_correct" if case == "x_junction" else "ty_continuation",
                label=1, pair_id=f"{first}->{second}",
                scores=_pair_scores(
                    field, scalar, state, endpoints[first], endpoints[second],
                    _branch_tangent(sample, by_id[first], endpoints[first]),
                    _branch_tangent(sample, by_id[second], endpoints[second]), domain,
                ),
            )
        if case == "x_junction":
            for first_position, first in enumerate(incident):
                for second in incident[first_position + 1:]:
                    if frozenset((first, second)) in paired:
                        continue
                    _append_pair(
                        rows, sample=sample, task="x_wrong_turn", label=0, pair_id=f"{first}->{second}",
                        scores=_pair_scores(
                            field, scalar, state, endpoints[first], endpoints[second],
                            _branch_tangent(sample, by_id[first], endpoints[first]),
                            _branch_tangent(sample, by_id[second], endpoints[second]), domain,
                        ),
                    )
    return rows


def oracle_rows(split: str, *, image_size: int = 64, device: str = "cpu") -> list[dict[str, Any]]:
    if split not in {"train", "validation"}:
        raise PermissionError("oracle metrics are limited to train/validation; confirm remains unopened")
    device_obj = torch.device(device); rows = []
    for index in range(SPLIT_SIZES_V4[split]):
        sample = generate_sample_v4(split, index, image_size=image_size)
        if sample["case"] in PRIMARY_CASES:
            rows.extend(_sample_rows(sample, device=device_obj))
    return rows


def calibrate_thresholds(train_rows: list[dict[str, Any]]) -> dict[str, float]:
    thresholds = {}
    for method in METHODS:
        negatives = np.asarray([row["score"] for row in train_rows if row["method"] == method and row["label"] == 0])
        thresholds[method] = select_threshold(negatives, max_fpr=0.05)
    return thresholds


def evaluate_oracle_rows(
    validation_rows: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    output = {}
    for method in METHODS:
        rows = [row for row in validation_rows if row["method"] == method]
        positive = np.asarray([row["score"] for row in rows if row["label"] == 1])
        negative = np.asarray([row["score"] for row in rows if row["label"] == 0])
        threshold = thresholds[method]
        def rate(task: str, label: int) -> float | None:
            values = np.asarray([row["score"] for row in rows if row["task"] == task and row["label"] == label])
            return float(np.mean(values >= threshold)) if len(values) else None
        low_fpr = low_fpr_metrics(positive, negative, max_fpr=0.05)
        output[method] = {
            "calibrated_train_threshold": float(threshold),
            "validation_positive_continuation_recall": float(np.mean(positive >= threshold)),
            "validation_overall_negative_fpr": float(np.mean(negative >= threshold)),
            "x_correct_recall": rate("x_correct", 1),
            "x_wrong_turn_fpr": rate("x_wrong_turn", 0),
            "curved_continuation_recall": rate("curved_continuation", 1),
            "parallel_correct_recall": rate("parallel_correct", 1),
            "parallel_false_bridge": rate("parallel_wrong", 0),
            "positive_gap_recovery": rate("positive_gap", 1),
            "negative_gap_false_bridge": rate("negative_gap", 0),
            "ty_continuation_recall": rate("ty_continuation", 1),
            "low_false_bridge_recovery_area_normalized": low_fpr["low_fpr_pauc_normalized"],
            "ranking_probability": low_fpr["ranking_probability"],
            "positive_count": int(len(positive)), "negative_count": int(len(negative)),
        }
    scalar, state = output["G0_scalar"], output["G1_mode_state"]
    x_relative_reduction = (
        (scalar["x_wrong_turn_fpr"] - state["x_wrong_turn_fpr"]) / scalar["x_wrong_turn_fpr"]
        if scalar["x_wrong_turn_fpr"] and scalar["x_wrong_turn_fpr"] > 0 else 0.0
    )
    gate_checks = {
        "positive_noninferiority": state["validation_positive_continuation_recall"] >= scalar["validation_positive_continuation_recall"] - 0.01,
        "x_wrong_turn_relative_reduction_at_least_50pct": x_relative_reduction >= 0.50,
        "parallel_false_bridge_noninferiority": state["parallel_false_bridge"] <= scalar["parallel_false_bridge"],
        "negative_gap_false_bridge_noninferiority": state["negative_gap_false_bridge"] <= scalar["negative_gap_false_bridge"],
        "curved_continuation_noninferiority": state["curved_continuation_recall"] >= scalar["curved_continuation_recall"] - 0.01,
    }
    return {
        "methods": output,
        "x_wrong_turn_relative_reduction": float(x_relative_reduction),
        "gate_checks": gate_checks,
        "gate_pass": all(gate_checks.values()),
        "status": "PHASE3D_ORACLE_MODE_STATE_PASS" if all(gate_checks.values()) else "FINAL_STOP_MODE_STATE_ORACLE_NO_VALUE",
        "training_performed": False,
        "confirm_evaluation_opened": False,
        "cracks_data_accessed": False,
        "expert_data_accessed": False,
    }
