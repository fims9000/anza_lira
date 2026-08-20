"""Corrected threshold-free routing metrics for CrossingTraceBench.

This module intentionally does not replace :mod:`synthetic.structural_metrics`.
It keeps pixel/trace quality (Family A), learned routing (Family B), and the
generator-geometry heuristic (Family C) as separate evidence families.
"""

from __future__ import annotations

from itertools import combinations
import math
from typing import Any, Mapping, Sequence

import numpy as np

from synthetic.evaluation import minimum_angle_continuation_scores
from synthetic.structural_metrics import compute_structural_metrics


FAMILY_A_KEYS = (
    "visible_dice",
    "visible_iou",
    "visible_precision",
    "visible_recall",
    "visible_cldice",
    "latent_cldice",
    "latent_skeleton_f1_2px",
    "junction_f1",
    "endpoint_f1",
    "false_merge_rate",
    "false_split_rate",
    "fragmentation",
    "symmetric_skeleton_distance",
    "gap_recovery_rate",
    "positive_gap_count",
    "false_bridge_rate",
    "false_bridge_count",
    "negative_gap_count",
    "orientation_error_median_deg",
)

ROUTE_METRIC_KEYS = (
    "route_top1_hit",
    "route_true_probability_mass",
    "route_mrr",
    "route_average_precision",
    "route_entropy_normalized",
    "chance_top1",
    "route_excess_over_chance",
    "topology_constrained_pairing_score",
    "x_pairing_exact_accuracy",
    "x_pair_f1",
    "t_pairing_exact_accuracy",
    "t_pair_f1",
    "y_pairing_exact_accuracy",
    "y_pair_f1",
    "legacy_threshold_0_5_pair_metric",
)


def _pair(first: int, second: int) -> tuple[int, int]:
    return tuple(sorted((int(first), int(second))))


def _pair_f1(predicted: set[tuple[int, int]], truth: set[tuple[int, int]]) -> float:
    true_positive = len(predicted & truth)
    precision = true_positive / len(predicted) if predicted else float(not truth)
    recall = true_positive / len(truth) if truth else 1.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _symmetric_pair_score(
    probabilities: np.ndarray,
    first_index: int,
    second_index: int,
    *,
    epsilon: float,
) -> float:
    return math.sqrt(
        (float(probabilities[first_index, second_index]) + epsilon)
        * (float(probabilities[second_index, first_index]) + epsilon)
    )


def topology_constrained_assignment(
    probabilities: np.ndarray,
    branch_ids: Sequence[int],
    junction_type: str,
    incident_branch_ids: Sequence[int],
    *,
    epsilon: float = 1e-12,
) -> set[tuple[int, int]]:
    """Choose an X/T/Y assignment using scores and public topology only.

    The function deliberately has no ground-truth continuation argument, so a
    true relation cannot leak into prediction construction.
    """

    scores = np.asarray(probabilities, dtype=np.float64)
    identifiers = [int(value) for value in branch_ids]
    incident = [int(value) for value in incident_branch_ids]
    if scores.shape != (len(identifiers), len(identifiers)) or not np.isfinite(scores).all():
        raise ValueError("Routing probabilities must be a finite square branch matrix")
    if len(set(identifiers)) != len(identifiers) or not set(incident).issubset(identifiers):
        raise ValueError("Incident branches must be unique members of branch_ids")
    index = {branch_id: position for position, branch_id in enumerate(identifiers)}

    def score(pair: tuple[int, int]) -> float:
        return _symmetric_pair_score(scores, index[pair[0]], index[pair[1]], epsilon=epsilon)

    if junction_type == "x_crossing":
        if len(incident) != 4:
            raise ValueError("X-crossing requires exactly four incident branches")
        a, b, c, d = incident
        candidates = (
            {_pair(a, b), _pair(c, d)},
            {_pair(a, c), _pair(b, d)},
            {_pair(a, d), _pair(b, c)},
        )
        return max(candidates, key=lambda pairs: (sum(math.log(score(pair) + epsilon) for pair in pairs), sorted(pairs)))
    if junction_type == "t_intersection":
        if len(incident) != 3:
            raise ValueError("T-intersection requires exactly three incident branches")
        return {max((_pair(a, b) for a, b in combinations(incident, 2)), key=lambda pair: (score(pair), pair))}
    if junction_type == "y_branch":
        if len(incident) != 3:
            raise ValueError("Y-branch requires exactly three incident branches")
        hub = max(
            incident,
            key=lambda candidate: (
                sum(score(_pair(candidate, other)) for other in incident if other != candidate),
                candidate,
            ),
        )
        return {_pair(hub, other) for other in incident if other != hub}
    raise ValueError(f"Unsupported junction topology: {junction_type}")


def _validate_route_matrices(
    probabilities: np.ndarray,
    target: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = np.asarray(probabilities, dtype=np.float64)
    eligible = np.asarray(target["continuation_eligible_matrix"], dtype=bool)
    truth = np.asarray(target["continuation_relation_matrix"], dtype=bool)
    if probability.shape != eligible.shape or truth.shape != eligible.shape or probability.ndim != 2:
        raise ValueError("Probability, eligibility, and truth must be matching branch matrices")
    if not np.isfinite(probability).all() or np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("Routing probabilities must be finite values in [0, 1]")
    if np.any(probability[~eligible] > 1e-7):
        raise ValueError("Ineligible destinations must have zero probability")
    if np.any(truth & ~eligible):
        raise ValueError("Every true continuation must be eligible")
    valid_rows = eligible.any(axis=1)
    if valid_rows.any() and not np.allclose(probability[valid_rows].sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Eligible routing rows must be row-stochastic")
    return probability, eligible, truth


def unavailable_route_metrics() -> dict[str, Any]:
    """Explicit NA Family B result for a model without transport output."""

    return {
        "family": "B_ROUTE_MECHANISM",
        "available": False,
        "reason": "MODEL_HAS_NO_TRANSPORT_OUTPUT",
        **{name: None for name in ROUTE_METRIC_KEYS},
        "route_row_count": 0,
        "topology_junction_count": 0,
        "legacy_readout_invalid_for_primary_claim": True,
    }


def compute_route_metrics(
    probabilities: np.ndarray,
    target: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute threshold-free Family B metrics for a real routing output."""

    probability, eligible, truth = _validate_route_matrices(probabilities, target)
    row_values: dict[str, list[float]] = {
        "route_top1_hit": [],
        "route_true_probability_mass": [],
        "route_mrr": [],
        "route_average_precision": [],
        "route_entropy_normalized": [],
        "chance_top1": [],
        "route_excess_over_chance": [],
    }
    for row_index in np.flatnonzero(truth.any(axis=1)):
        destinations = np.flatnonzero(eligible[row_index])
        true_destinations = set(np.flatnonzero(truth[row_index]).tolist())
        ranked = sorted(destinations.tolist(), key=lambda value: (-probability[row_index, value], value))
        hits = np.asarray([value in true_destinations for value in ranked], dtype=bool)
        top1 = float(hits[0])
        true_mass = float(probability[row_index, list(true_destinations)].sum())
        first_true_rank = int(np.flatnonzero(hits)[0]) + 1
        hit_ranks = np.flatnonzero(hits) + 1
        average_precision = float(np.mean(np.cumsum(hits)[hits] / hit_ranks))
        row_probability = probability[row_index, destinations]
        entropy = float(-(row_probability * np.log(np.clip(row_probability, 1e-12, None))).sum())
        normalized_entropy = entropy / math.log(len(destinations)) if len(destinations) > 1 else 0.0
        chance = len(true_destinations) / len(destinations)
        row_values["route_top1_hit"].append(top1)
        row_values["route_true_probability_mass"].append(true_mass)
        row_values["route_mrr"].append(1.0 / first_true_rank)
        row_values["route_average_precision"].append(average_precision)
        row_values["route_entropy_normalized"].append(normalized_entropy)
        row_values["chance_top1"].append(chance)
        row_values["route_excess_over_chance"].append(top1 - chance)

    topology: dict[str, list[tuple[float, float]]] = {
        "x_crossing": [],
        "t_intersection": [],
        "y_branch": [],
    }
    branch_ids = [int(value) for value in target["branch_ids"]]
    predicted_all: set[tuple[int, int]] = set()
    true_all: set[tuple[int, int]] = set()
    for junction in target["junctions"]:
        junction_type = str(junction["junction_type"])
        predicted = topology_constrained_assignment(
            probability,
            branch_ids,
            junction_type,
            junction["incident_branch_ids"],
        )
        expected = {_pair(*pair) for pair in junction["continuation_relation"]}
        topology[junction_type].append((float(predicted == expected), _pair_f1(predicted, expected)))
        predicted_all.update(predicted)
        true_all.update(expected)

    legacy_selected = {
        _pair(branch_ids[first], branch_ids[second])
        for first, second in np.argwhere(np.triu(eligible, k=1))
        if probability[first, second] >= 0.5
    }
    values: dict[str, Any] = {
        "family": "B_ROUTE_MECHANISM",
        "available": True,
        **{name: float(np.mean(items)) if items else None for name, items in row_values.items()},
        "route_row_count": len(row_values["route_top1_hit"]),
        "topology_constrained_pairing_score": _pair_f1(predicted_all, true_all) if true_all else None,
        "topology_junction_count": sum(len(items) for items in topology.values()),
        "legacy_threshold_0_5_pair_metric": _pair_f1(legacy_selected, true_all) if true_all else None,
        "legacy_readout_invalid_for_primary_claim": True,
    }
    prefixes = {"x_crossing": "x", "t_intersection": "t", "y_branch": "y"}
    for junction_type, items in topology.items():
        prefix = prefixes[junction_type]
        values[f"{prefix}_pairing_exact_accuracy"] = float(np.mean([item[0] for item in items])) if items else None
        values[f"{prefix}_pair_f1"] = float(np.mean([item[1] for item in items])) if items else None
        values[f"{prefix}_junction_count"] = len(items)
    for key, value in values.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Corrected routing metric is not finite: {key}")
    return values


def compute_family_a_metrics(
    predicted_visible_mask: np.ndarray,
    target: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Return only model-comparable pixel/trace metrics from the legacy core."""

    metrics = compute_structural_metrics(predicted_visible_mask, target, **kwargs)
    return {"family": "A_SEGMENTATION_TRACE", **{key: metrics[key] for key in FAMILY_A_KEYS}}


def geometry_only_minimum_angle_heuristic(target: Mapping[str, Any]) -> dict[str, Any]:
    """Family C diagnostic; generator geometry is explicit in the metadata."""

    scores = minimum_angle_continuation_scores(dict(target))
    # Convert deterministic selections to row-stochastic scores for the common
    # topology assignment while keeping this diagnostic outside model results.
    eligible = np.asarray(target["continuation_eligible_matrix"], dtype=bool)
    probability = np.zeros_like(scores, dtype=np.float64)
    for row in range(len(scores)):
        selected = eligible[row] & (scores[row] > 0)
        if selected.any():
            probability[row, selected] = 1.0 / int(selected.sum())
        elif eligible[row].any():
            probability[row, eligible[row]] = 1.0 / int(eligible[row].sum())
    return {
        "family": "C_GEOMETRY_DIAGNOSTIC",
        "diagnostic_id": "geometry_only_minimum_angle_heuristic",
        "uses_generator_branch_geometry": True,
        "is_model_specific": False,
        "eligible_for_model_superiority_claim": False,
        "metrics": compute_route_metrics(probability, target),
    }
