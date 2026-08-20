"""Truth-blind candidate scoring and lineage evaluation for all S0 controls."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ..synthetic3d.families import END_INDEX, CaseBatch, generate_batch
from ..transport.core import common_mean, gaussian_log_score, initial_covariance, propagate_covariance


@dataclass
class TrackingRows:
    rows: list[dict[str, Any]]
    transition_correct: np.ndarray
    switch: np.ndarray
    margin: np.ndarray


def _track_batch(method: str, params: dict[str, float], batch: CaseBatch, split_name: str) -> TrackingRows:
    n, steps, candidate_count, _ = batch.candidate_points.shape
    covariance = None if method == "G0_euclidean" else initial_covariance(method, batch.candidate_theta[:, 0, 0], params)
    theta = batch.candidate_theta[:, 0, 0].copy()
    last = batch.candidate_points[:, 0, 0].copy(); previous = np.zeros_like(last)
    has_previous = np.zeros(n, dtype=bool); last_k = np.zeros(n, dtype=np.int16); active = np.ones(n, dtype=bool)
    correctness = np.full((n, steps), np.nan); margins = np.full((n, steps), np.nan); positional = np.full((n, steps), np.nan)
    for k in range(1, steps):
        hidden = active & ~batch.observed[:, k]
        if hidden.any() and method not in ("G0_euclidean", "G1_local_reset"):
            covariance[hidden] = propagate_covariance(method, covariance[hidden], theta[hidden], theta[hidden], params)
        visible = active & batch.observed[:, k]
        if not visible.any():
            continue
        indices = np.flatnonzero(visible); elapsed = k - last_k[indices]
        predicted = np.where(has_previous[indices, None], common_mean(last[indices], previous[indices], elapsed), last[indices])
        scores = np.full((len(indices), candidate_count), -np.inf, dtype=np.float64)
        candidate_covariances = None if method == "G0_euclidean" else np.empty((len(indices), candidate_count, 2, 2), dtype=np.float64)
        for candidate in range(candidate_count):
            valid = batch.candidate_valid[indices, k, candidate]
            if candidate == END_INDEX:
                scores[valid, candidate] = 0.0
                if candidate_covariances is not None:
                    candidate_covariances[:, candidate] = covariance[indices]
                continue
            delta = batch.candidate_points[indices, k, candidate] - predicted
            if method == "G0_euclidean":
                scores[valid, candidate] = -np.sum(delta[valid] ** 2, axis=-1)
            else:
                local_covariance = propagate_covariance(
                    method, covariance[indices], theta[indices], batch.candidate_theta[indices, k, candidate], params,
                )
                candidate_covariances[:, candidate] = local_covariance
                scores[valid, candidate] = gaussian_log_score(delta[valid], local_covariance[valid])
        order = np.argsort(scores, axis=1); selected = order[:, -1]
        best = scores[np.arange(len(indices)), selected]; second = scores[np.arange(len(indices)), order[:, -2]]
        finite_second = np.isfinite(second); local_margin = np.where(finite_second, best - second, np.inf)
        truth = batch.truth_index[indices, k]; correct = selected == truth
        correctness[indices, k] = correct.astype(float); margins[indices, k] = local_margin
        point_selected = selected != END_INDEX
        local = np.flatnonzero(point_selected); global_indices = indices[local]; chosen = selected[local]
        chosen_points = batch.candidate_points[global_indices, k, chosen]
        chosen_theta = batch.candidate_theta[global_indices, k, chosen]
        positional[global_indices, k] = np.linalg.norm(chosen_points - batch.true_points[global_indices, k], axis=-1)
        previous[global_indices] = last[global_indices]; last[global_indices] = chosen_points
        has_previous[global_indices] = True; last_k[global_indices] = k; theta[global_indices] = chosen_theta
        if covariance is not None and local.size:
            covariance[global_indices] = candidate_covariances[local, chosen]
        ended = selected == END_INDEX; active[indices[ended]] = False

    rows = []; per_case_accuracy = np.nanmean(correctness, axis=1); switched = np.nanmin(correctness, axis=1) < 1
    case_margin = np.nanmin(margins, axis=1)
    for index in range(n):
        decisions = correctness[index, np.isfinite(correctness[index])]
        correct_position = positional[index, correctness[index] == 1]
        row = {
            "split": split_name, "method": method, "scene_id": int(batch.scene_id[index]), "family": str(batch.family[index]),
            "top1": float(per_case_accuracy[index]), "switch": int(switched[index]),
            "survival_3": float(np.all(decisions[:3] == 1)), "survival_7": float(np.all(decisions[:7] == 1)),
            "survival_15": float(np.all(decisions[:15] == 1)),
            "mean_position_error_correct_lineage": float(np.mean(correct_position)) if correct_position.size else np.nan,
            "margin": float(case_margin[index]), "decision_count": int(decisions.size),
        }
        rows.append(row)
    return TrackingRows(rows, per_case_accuracy, switched.astype(float), case_margin)


def evaluate_method(method: str, params: dict[str, float], split_name: str, *, total: int = 10_000, batch_size: int = 1000) -> TrackingRows:
    rows = []; correct = []; switches = []; margins = []
    for start in range(0, total, batch_size):
        batch = generate_batch(split_name, start, min(batch_size, total - start)); result = _track_batch(method, params, batch, split_name)
        rows.extend(result.rows); correct.append(result.transition_correct); switches.append(result.switch); margins.append(result.margin)
    return TrackingRows(rows, np.concatenate(correct), np.concatenate(switches), np.concatenate(margins))


def evaluate_method_batches(method: str, params: dict[str, float], split_name: str, batches: list[CaseBatch]) -> TrackingRows:
    rows = []; correct = []; switches = []; margins = []
    for batch in batches:
        result = _track_batch(method, params, batch, split_name); rows.extend(result.rows)
        correct.append(result.transition_correct); switches.append(result.switch); margins.append(result.margin)
    return TrackingRows(rows, np.concatenate(correct), np.concatenate(switches), np.concatenate(margins))


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    output = {}
    for key in ("top1", "switch", "survival_3", "survival_7", "survival_15", "mean_position_error_correct_lineage"):
        values = np.asarray([row[key] for row in rows], dtype=float)
        output[key] = float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else math.nan
    output["scene_count"] = len(rows)
    return output


def summarize_strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for family in sorted({row["family"] for row in rows}):
        local = [row for row in rows if row["family"] == family]
        output.append({"split": local[0]["split"], "method": local[0]["method"], "family": family, **summarize(local)})
    return output
