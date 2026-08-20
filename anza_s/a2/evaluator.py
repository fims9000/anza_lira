"""Causal A1/A2/A3 evaluation on fixed ANZA-S trajectories."""

from __future__ import annotations

from typing import Any

import numpy as np

from anza2.eval.low_fpr import low_fpr_metrics, select_threshold
from anza_s.rollout import rollout
from anza_s.shadowing import terminal_meeting_score
from anza_s.oracle_field import OracleCocycleField, geometry_for_sample

from .cases import a2_candidate_stream, curved_comparability
from .covariance_transport import covariance_sequence, trajectory_diagnostics
from .shadowing import hyperbolic_shadowing


METHODS = (
    "A0_tangent_terminal",
    "A1_isotropic_shadowing",
    "A2_local_anisotropic_reset",
    "A3_cocycle_cg_lambda0",
    "A3_cocycle_cg_lambda035",
)
TASKS = {
    "P1_x": ("x_correct", "x_wrong_turn"),
    "P2_parallel": ("parallel_correct", "parallel_wrong"),
    "P3_curved": ("curved_gap", "curved_confuser"),
}


def _score(method: str, left: tuple, right: tuple) -> tuple[float, float | None, tuple[int, int] | None]:
    if method == "A0_tangent_terminal":
        return terminal_meeting_score(left, right), None, None
    if method == "A1_isotropic_shadowing":
        mode, value = "isotropic", 0.0
    elif method == "A2_local_anisotropic_reset":
        mode, value = "local_reset", 0.35
    elif method == "A3_cocycle_cg_lambda0":
        mode, value = "composed", 0.0
    elif method == "A3_cocycle_cg_lambda035":
        mode, value = "composed", 0.35
    else:
        raise ValueError(method)
    left_cov = covariance_sequence(left, mode=mode, hyperbolicity=value)
    right_cov = covariance_sequence(right, mode=mode, hyperbolicity=value)
    energy, score, meeting, _matrix = hyperbolic_shadowing(left, right, left_cov, right_cov)
    return score, energy, meeting


def oracle_rows(split: str, *, image_size: int = 64) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stream = a2_candidate_stream(split, image_size=image_size)
    comparability = curved_comparability(stream)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for sample, candidate in stream:
        field = OracleCocycleField(geometry_for_sample(sample))
        left = rollout(field, candidate.start_xy, candidate.start_direction, steps=3, cocycle=False)
        right = rollout(field, candidate.goal_xy, candidate.goal_direction, steps=3, cocycle=False)
        base = {
            "split": split, "index": candidate.index, "case": candidate.case,
            "task": candidate.task, "label": candidate.label, "pair_id": candidate.pair_id,
        }
        for method in METHODS:
            score, energy, meeting = _score(method, left, right)
            rows.append({
                **base, "method": method, "score": score,
                "shadowing_energy": "" if energy is None else energy,
                "meeting_left_step": "" if meeting is None else meeting[0],
                "meeting_right_step": "" if meeting is None else meeting[1],
            })
        for side, trajectory in (("left", left), ("right", right)):
            for value in (0.0, 0.35):
                for item in trajectory_diagnostics(trajectory, hyperbolicity=value):
                    diagnostics.append({**base, "side": side, "lambda": value, **item})
    return rows, diagnostics, comparability


def calibrate(train_rows: list[dict[str, Any]], *, p3_primary: bool) -> dict[str, Any]:
    task_names = ("P1_x", "P2_parallel", "P3_curved") if p3_primary else ("P1_x", "P2_parallel")
    output: dict[str, Any] = {}
    for method in METHODS:
        output[method] = {}
        selected = [row for row in train_rows if row["method"] == method]
        for name in task_names:
            positive_task, negative_task = TASKS[name]
            negatives = np.asarray([row["score"] for row in selected if row["task"] == negative_task], dtype=np.float64)
            threshold = float(select_threshold(negatives, max_fpr=0.05))
            if not np.isfinite(threshold):
                threshold = float(np.nextafter(negatives.max(), np.inf))
            output[method][name] = {
                "threshold": threshold,
                "train_fpr": float(np.mean(negatives >= threshold)),
                "positive_task": positive_task, "negative_task": negative_task,
            }
    return {"primary_tasks": list(task_names), "methods": output}


def _task_metrics(rows: list[dict[str, Any]], method: str, task: str, threshold: float) -> dict[str, float]:
    positive_task, negative_task = TASKS[task]
    selected = [row for row in rows if row["method"] == method]
    positives = np.asarray([row["score"] for row in selected if row["task"] == positive_task], dtype=np.float64)
    negatives = np.asarray([row["score"] for row in selected if row["task"] == negative_task], dtype=np.float64)
    metrics = low_fpr_metrics(positives, negatives, max_fpr=0.05)
    return {
        "positive_count": int(positives.size), "negative_count": int(negatives.size),
        "threshold_from_train": threshold,
        "tpr": float(np.mean(positives >= threshold)), "fpr": float(np.mean(negatives >= threshold)),
        "auroc": metrics["auroc"], "ranking_probability": metrics["ranking_probability"],
        "pauc_fpr_0_05": metrics["low_fpr_pauc_normalized"],
    }


def _bootstrap_delta(
    rows: list[dict[str, Any]], freeze: dict[str, Any], *, repetitions: int = 10_000, seed: int = 20260818,
) -> dict[str, float]:
    rng = np.random.default_rng(seed); by_task = {}
    for task in freeze["primary_tasks"]:
        positive_task, _ = TASKS[task]
        records: dict[int, dict[str, list[float]]] = {}
        for row in rows:
            if row["task"] != positive_task or row["method"] not in {"A2_local_anisotropic_reset", "A3_cocycle_cg_lambda035"}:
                continue
            records.setdefault(int(row["index"]), {}).setdefault(row["method"], []).append(float(row["score"]))
        deltas = []
        for pair in records.values():
            if len(pair) != 2:
                raise AssertionError("paired A2/A3 scores required")
            a2 = np.mean(np.asarray(pair["A2_local_anisotropic_reset"]) >= freeze["methods"]["A2_local_anisotropic_reset"][task]["threshold"])
            a3 = np.mean(np.asarray(pair["A3_cocycle_cg_lambda035"]) >= freeze["methods"]["A3_cocycle_cg_lambda035"][task]["threshold"])
            deltas.append(float(a3 - a2))
        by_task[task] = np.asarray(deltas, dtype=np.float64)
    samples = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        samples[index] = np.mean([
            float(np.mean(values[rng.integers(0, len(values), len(values))])) for values in by_task.values()
        ])
    return {
        "repetitions": repetitions, "seed": seed,
        "estimate": float(np.mean([values.mean() for values in by_task.values()])),
        "ci95_low": float(np.quantile(samples, 0.025)), "ci95_high": float(np.quantile(samples, 0.975)),
        "unit": "synthetic sample index cluster within predeclared task, then equal-weight task macro",
    }


def evaluate(rows: list[dict[str, Any]], freeze: dict[str, Any]) -> dict[str, Any]:
    methods = {}
    for method in METHODS:
        tasks = {name: _task_metrics(rows, method, name, freeze["methods"][method][name]["threshold"]) for name in freeze["primary_tasks"]}
        methods[method] = {
            "tasks": tasks,
            "macro_tpr": float(np.mean([value["tpr"] for value in tasks.values()])),
            "macro_fpr": float(np.mean([value["fpr"] for value in tasks.values()])),
            "macro_ranking": float(np.mean([value["ranking_probability"] for value in tasks.values()])),
            "macro_pauc_fpr_0_05": float(np.mean([value["pauc_fpr_0_05"] for value in tasks.values()])),
        }
    a1, a2, a3, null = (methods[name] for name in (
        "A1_isotropic_shadowing", "A2_local_anisotropic_reset",
        "A3_cocycle_cg_lambda035", "A3_cocycle_cg_lambda0",
    ))
    x_a2 = a2["tasks"]["P1_x"]["tpr"]; x_a3 = a3["tasks"]["P1_x"]["tpr"]
    ceiling = x_a2 >= 0.95 and x_a3 >= x_a2
    bootstrap = _bootstrap_delta(rows, freeze)
    non_ceiling = [name for name in freeze["primary_tasks"] if null["tasks"][name]["tpr"] < 0.95]
    lambda_gains = {name: a3["tasks"][name]["tpr"] - null["tasks"][name]["tpr"] for name in freeze["primary_tasks"]}
    lambda_causal = bool(non_ceiling and max(lambda_gains[name] for name in non_ceiling) > 0.0)
    gates = {
        "x_gain_or_ceiling": bool((x_a3 - x_a2 >= 0.10) or ceiling),
        "macro_tpr_gain_at_least_0_08": bool(a3["macro_tpr"] - a2["macro_tpr"] >= 0.08),
        "macro_ranking_improves": bool(a3["macro_ranking"] > a2["macro_ranking"]),
        "parallel_safety": bool(a3["tasks"]["P2_parallel"]["fpr"] <= a2["tasks"]["P2_parallel"]["fpr"]),
        "paired_ci_lower_above_zero": bool(bootstrap["ci95_low"] > 0.0),
        "lambda_0_35_non_inert": lambda_causal,
    }
    score_differences = []
    paired = {}
    for row in rows:
        if row["method"] in {"A3_cocycle_cg_lambda0", "A3_cocycle_cg_lambda035"}:
            paired.setdefault((row["index"], row["task"], row["pair_id"]), {})[row["method"]] = float(row["score"])
    for pair in paired.values():
        score_differences.append(abs(pair["A3_cocycle_cg_lambda035"] - pair["A3_cocycle_cg_lambda0"]))
    hyperbolicity_inert = max(score_differences, default=0.0) <= 1e-12
    unsafe = not gates["parallel_safety"]
    all_ceiling = all(a2["tasks"][name]["tpr"] >= 0.95 for name in freeze["primary_tasks"])
    if unsafe:
        status = "ANOSOV_COCYCLE_UNSAFE"
    elif hyperbolicity_inert:
        status = "ANOSOV_HYPERBOLICITY_INERT"
    elif all(gates.values()):
        status = "ANOSOV_COCYCLE_CAUSAL_ORACLE_PASS"
    elif all_ceiling:
        status = "ANOSOV_COCYCLE_REDUNDANT_AT_ORACLE"
    elif a2["macro_tpr"] > a1["macro_tpr"] and a3["macro_tpr"] - a2["macro_tpr"] < 0.08:
        status = "LOCAL_ANISOTROPY_GAIN_COCYCLE_REDUNDANT"
    else:
        status = "SHADOWING_ONLY_NO_ANOSOV_GAIN"
    return {
        "status": status, "gate_pass": status == "ANOSOV_COCYCLE_CAUSAL_ORACLE_PASS",
        "methods": methods, "gates": gates, "paired_bootstrap": bootstrap,
        "lambda_intervention": {"task_tpr_gains_lambda035_minus_lambda0": lambda_gains,
                                "max_absolute_score_difference": max(score_differences, default=0.0),
                                "hyperbolicity_inert": hyperbolicity_inert},
        "training_performed": False, "confirm_opened": False, "test_opened": False,
        "cracks_data_accessed": False, "expert_data_accessed": False,
    }


def gap_identifiability_control(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for method in METHODS:
        positives = np.asarray([row["score"] for row in rows if row["method"] == method and row["task"] == "straight_gap"])
        negatives = np.asarray([row["score"] for row in rows if row["method"] == method and row["task"] == "negative_gap"])
        metrics = low_fpr_metrics(positives, negatives)
        if not np.isfinite(metrics["threshold"]):
            threshold = float(np.nextafter(negatives.max(), np.inf))
            metrics.update({
                "threshold": threshold,
                "tpr_at_fpr_0_05": float(np.mean(positives >= threshold)),
                "fpr": float(np.mean(negatives >= threshold)),
            })
        output[method] = {**metrics, "role": "leakage control only; excluded from gate"}
    return output
