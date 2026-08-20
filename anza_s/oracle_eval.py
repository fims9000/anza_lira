"""O0--O4 zero-training oracle comparison for ANZA-S."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from anza2.eval.low_fpr import low_fpr_metrics, select_threshold
from anza2.forensics.component_replacement import oracle_field_from_sample
from anza2.phase3d.mode_state_graph import mode_state_edge_weights
from anza2.phase3d.oracle_graph_eval import _pair_scores
from models.anza2.affinity import ANZA2StructuralAffinity

from .cases import ContinuationCandidate, candidate_stream
from .oracle_field import OracleCocycleField, geometry_for_sample
from .rollout import rollout
from .shadowing import terminal_meeting_score, two_sided_shadowing


METHODS = ("O0_scalar_anza", "O1_mode_state", "O2_tangent_streamline", "O3_cocycle_rollout", "O4_cocycle_shadowing")
NEGATIVE_TASKS = ("x_wrong_turn", "parallel_wrong", "negative_gap")
PRIMARY_POSITIVE_TASKS = ("straight_gap", "curved_gap", "x_correct", "parallel_correct")


def _pixel(xy: np.ndarray, size: int) -> tuple[int, int]:
    x, y = np.rint(xy).astype(int)
    return int(np.clip(y, 0, size - 1)), int(np.clip(x, 0, size - 1))


def _axial(direction: np.ndarray) -> np.ndarray:
    theta = float(np.arctan2(direction[1], direction[0]))
    return np.asarray((np.cos(2.0 * theta), np.sin(2.0 * theta)), dtype=np.float64)


def _candidate_scores(
    sample: dict[str, Any], candidate: ContinuationCandidate, *, device: torch.device,
    prepared: tuple[Any, np.ndarray, np.ndarray, OracleCocycleField] | None = None,
) -> tuple[dict[str, float], dict[str, tuple]]:
    if prepared is None:
        prepared = _prepare_sample(sample, device=device)
    field, scalar, state, oracle = prepared
    size = int(sample["image_size"])
    old = _pair_scores(
        field, scalar, state, _pixel(candidate.start_xy, size), _pixel(candidate.goal_xy, size),
        _axial(candidate.start_direction), _axial(candidate.goal_direction), candidate.domain,
    )
    tangent_left = rollout(oracle, candidate.start_xy, candidate.start_direction, steps=3, cocycle=False)
    tangent_right = rollout(oracle, candidate.goal_xy, candidate.goal_direction, steps=3, cocycle=False)
    cocycle_left = rollout(oracle, candidate.start_xy, candidate.start_direction, steps=3, cocycle=True)
    cocycle_right = rollout(oracle, candidate.goal_xy, candidate.goal_direction, steps=3, cocycle=True)
    shadow_energy, shadow_score, meeting = two_sided_shadowing(cocycle_left, cocycle_right)
    scores = {
        "O0_scalar_anza": float(old["G0_scalar"][0]),
        "O1_mode_state": float(old["G1_mode_state"][0]),
        "O2_tangent_streamline": terminal_meeting_score(tangent_left, tangent_right),
        "O3_cocycle_rollout": terminal_meeting_score(cocycle_left, cocycle_right),
        "O4_cocycle_shadowing": shadow_score,
    }
    details = {
        "O2_tangent_streamline": (tangent_left, tangent_right, None, None),
        "O3_cocycle_rollout": (cocycle_left, cocycle_right, None, None),
        "O4_cocycle_shadowing": (cocycle_left, cocycle_right, shadow_energy, meeting),
    }
    return scores, details


def _prepare_sample(sample: dict[str, Any], *, device: torch.device) -> tuple[Any, np.ndarray, np.ndarray, OracleCocycleField]:
    field, _valid = oracle_field_from_sample(sample, device=device)
    with torch.inference_mode():
        scalar = ANZA2StructuralAffinity()(field)[0].cpu().numpy().astype(np.float32)
        state = mode_state_edge_weights(field)[0].cpu().numpy().astype(np.float32)
    return field, scalar, state, OracleCocycleField(geometry_for_sample(sample))


def oracle_rows(split: str, *, image_size: int = 64, device: str = "cpu") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if split not in {"train", "validation"}:
        raise PermissionError("ANZA-S oracle is limited to train calibration and validation; confirm/test remain closed")
    rows, trajectories = [], []
    device_obj = torch.device(device)
    prepared_key = None; prepared = None
    for sample, candidate in candidate_stream(split, image_size=image_size):
        key = (sample["split"], sample["index"])
        if key != prepared_key:
            prepared = _prepare_sample(sample, device=device_obj); prepared_key = key
        scores, details = _candidate_scores(sample, candidate, device=device_obj, prepared=prepared)
        base = {
            "split": split, "index": candidate.index, "case": candidate.case,
            "task": candidate.task, "label": candidate.label, "pair_id": candidate.pair_id,
        }
        for method, score in scores.items():
            row = {**base, "method": method, "score": float(score)}
            if method == "O4_cocycle_shadowing":
                row["shadowing_energy"] = float(details[method][2])
                row["meeting_left_step"], row["meeting_right_step"] = details[method][3]
            rows.append(row)
        for method, (left, right, energy, meeting) in details.items():
            for side, path in (("left", left), ("right", right)):
                for point in path:
                    trajectories.append({
                        **base, "method": method, "side": side, "step": point.step,
                        "x": point.x, "y": point.y, "ux": point.ux, "uy": point.uy,
                        "branch_id": point.branch_id, "membership": point.membership,
                        "curvature": point.curvature,
                        "shadowing_energy": energy if energy is not None else "",
                        "meeting_left_step": meeting[0] if meeting is not None else "",
                        "meeting_right_step": meeting[1] if meeting is not None else "",
                    })
    return rows, trajectories


def calibrate_thresholds(train_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Maximum of task-wise inclusive thresholds enforces FPR<=.05 for every negative stratum."""

    output = {}
    for method in METHODS:
        selected = [row for row in train_rows if row["method"] == method]
        by_task = {}
        for task in NEGATIVE_TASKS:
            values = np.asarray([row["score"] for row in selected if row["task"] == task and row["label"] == 0])
            if not len(values):
                raise ValueError(f"missing train negatives for {method}/{task}")
            selected_threshold = float(select_threshold(values, max_fpr=0.05))
            if not np.isfinite(selected_threshold):
                # A tied maximum can occupy >5% of a stratum. The only valid
                # zero-FPR operating point is immediately above that maximum.
                selected_threshold = float(np.nextafter(values.max(), np.inf))
            by_task[task] = selected_threshold
        threshold = max(by_task.values())
        if not np.isfinite(threshold):
            raise AssertionError("calibration must produce a finite operating point")
        fpr = {
            task: float(np.mean([row["score"] >= threshold for row in selected if row["task"] == task and row["label"] == 0]))
            for task in NEGATIVE_TASKS
        }
        if any(value > 0.05 for value in fpr.values()):
            raise AssertionError("inclusive per-stratum FPR calibration failed")
        output[method] = {"threshold": float(threshold), "task_thresholds": by_task, "train_fpr": fpr}
    return output


def evaluate(rows: list[dict[str, Any]], thresholds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    methods = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        threshold = thresholds[method]["threshold"]
        def rate(task: str, label: int) -> float | None:
            values = [row["score"] for row in selected if row["task"] == task and row["label"] == label]
            return float(np.mean(np.asarray(values) >= threshold)) if values else None
        task_rates = {task: rate(task, 1) for task in PRIMARY_POSITIVE_TASKS}
        positives = np.asarray([row["score"] for row in selected if row["label"] == 1])
        negatives = np.asarray([row["score"] for row in selected if row["label"] == 0])
        low_fpr = low_fpr_metrics(positives, negatives, max_fpr=0.05)
        if not np.isfinite(low_fpr["threshold"]):
            finite_threshold = float(np.nextafter(negatives.max(), np.inf))
            low_fpr.update({
                "threshold": finite_threshold,
                "tpr_at_fpr_0_05": float(np.mean(positives >= finite_threshold)),
                "fpr": float(np.mean(negatives >= finite_threshold)),
            })
        methods[method] = {
            "threshold_from_train": threshold,
            "positive_recall_row_weighted": float(np.mean(positives >= threshold)),
            "positive_recall_macro_primary": float(np.mean(list(task_rates.values()))),
            "positive_recall_by_task": task_rates,
            "x_wrong_turn_fpr": rate("x_wrong_turn", 0),
            "parallel_false_bridge": rate("parallel_wrong", 0),
            "negative_gap_false_bridge": rate("negative_gap", 0),
            "ty_continuation_recall": rate("ty_continuation", 1),
            "low_fpr": low_fpr,
            "positive_count": int(len(positives)), "negative_count": int(len(negatives)),
        }
    o4 = methods["O4_cocycle_shadowing"]
    baseline_checks = {}
    for baseline_name in ("O0_scalar_anza", "O2_tangent_streamline"):
        baseline = methods[baseline_name]
        baseline_checks[baseline_name] = {
            "macro_positive_noninferiority": o4["positive_recall_macro_primary"] >= baseline["positive_recall_macro_primary"] - 0.01,
            "x_wrong_turn_relative_reduction_at_least_50pct": (
                baseline["x_wrong_turn_fpr"] > 0
                and o4["x_wrong_turn_fpr"] <= 0.5 * baseline["x_wrong_turn_fpr"]
            ),
            "parallel_false_bridge_noninferiority": o4["parallel_false_bridge"] <= baseline["parallel_false_bridge"],
            "negative_gap_false_bridge_noninferiority": o4["negative_gap_false_bridge"] <= baseline["negative_gap_false_bridge"],
            "curved_recall_noninferiority": o4["positive_recall_by_task"]["curved_gap"] >= baseline["positive_recall_by_task"]["curved_gap"] - 0.01,
        }
    gate_pass = all(all(checks.values()) for checks in baseline_checks.values())
    o2_scores = [row["score"] for row in rows if row["method"] == "O2_tangent_streamline"]
    o3_scores = [row["score"] for row in rows if row["method"] == "O3_cocycle_rollout"]
    if len(o2_scores) != len(o3_scores) or not o2_scores:
        raise AssertionError("paired O2/O3 candidate scores are required")
    o2_o3_max_diff = max(abs(first - second) for first, second in zip(o2_scores, o3_scores, strict=True))
    return {
        "methods": methods, "baseline_gate_checks": baseline_checks, "gate_pass": gate_pass,
        "status": "ANZA_S_ORACLE_GATE_A_PASS" if gate_pass else "FINAL_STOP_ANOSOV_DYNAMICAL_ARCHITECTURE",
        "causal_diagnostics": {
            "o2_o3_max_absolute_score_difference": float(o2_o3_max_diff),
            "cocycle_rollout_incremental_effect_established": bool(o2_o3_max_diff > 1e-12),
            "generic_tangent_plus_shadowing_control_included": False,
            "claim_boundary": "Formal Gate A tests O4 as specified, but cannot attribute its gain to the cocycle when O2 and O3 coincide and a generic tangent+shadowing control is absent.",
        },
        "training_performed": False, "confirm_opened": False,
        "cracks_data_accessed": False, "expert_data_accessed": False,
    }
