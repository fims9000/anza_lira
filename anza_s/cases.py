"""Frozen local continuation candidates for the ANZA-S oracle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation

from synthetic.crossing_trace_bench_v4 import generate_sample_v4

from .oracle_field import geometry_for_sample


PRIMARY_CASES = {"fault_with_gap", "negative_gap", "curved_fault", "near_parallel", "x_junction", "t_junction", "y_junction"}


@dataclass(frozen=True)
class ContinuationCandidate:
    split: str
    index: int
    case: str
    task: str
    label: int
    pair_id: str
    start_xy: np.ndarray
    goal_xy: np.ndarray
    start_direction: np.ndarray
    goal_direction: np.ndarray
    domain: np.ndarray


def _unit(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _candidate(sample: dict[str, Any], task: str, label: int, pair_id: str, start, goal, start_direction, goal_direction, domain) -> ContinuationCandidate:
    return ContinuationCandidate(
        str(sample["split"]), int(sample["index"]), str(sample["case"]), task, int(label), pair_id,
        np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64),
        _unit(start_direction), _unit(goal_direction), np.asarray(domain, dtype=bool),
    )


def _branch_anchor(points: np.ndarray, center: np.ndarray, distance: float = 4.0) -> np.ndarray:
    values = np.linalg.norm(np.asarray(points) - center, axis=1)
    return np.asarray(points[int(np.argmin(abs(values - distance)))], dtype=np.float64)


def candidates_for_sample(sample: dict[str, Any]) -> list[ContinuationCandidate]:
    geometry = geometry_for_sample(sample)
    case = str(sample["case"]); output = []
    if case in {"fault_with_gap", "negative_gap"}:
        start, goal = geometry.gaps[0].points_xy[[0, -1]]
        support = np.asarray(sample["latent_fault_mask"], dtype=bool) | np.asarray(sample["negative_gap_mask"], dtype=bool)
        domain = binary_dilation(support, iterations=1)
        output.append(_candidate(
            sample, "straight_gap" if case == "fault_with_gap" else "negative_gap",
            1 if case == "fault_with_gap" else 0, "gap:0", start, goal, goal - start, start - goal, domain,
        ))
    elif case == "curved_fault":
        branch = geometry.branches[0]; start, goal = branch.points_xy[43], branch.points_xy[53]
        domain = binary_dilation(np.asarray(sample["branch_masks"])[0], iterations=1)
        output.append(_candidate(sample, "curved_gap", 1, "derived:center", start, goal, goal - start, start - goal, domain))
    elif case == "near_parallel":
        domain = binary_dilation(np.asarray(sample["branch_masks"], dtype=bool).any(axis=0), iterations=4)
        for branch_index, branch in enumerate(geometry.branches):
            start, goal = branch.points_xy[43], branch.points_xy[53]
            output.append(_candidate(sample, "parallel_correct", 1, f"branch:{branch_index}", start, goal, goal - start, start - goal, domain))
        first, second = geometry.branches
        for reverse in (False, True):
            source, destination = (first, second) if not reverse else (second, first)
            start, goal = source.points_xy[43], destination.points_xy[53]
            output.append(_candidate(sample, "parallel_wrong", 0, f"cross:{int(reverse)}", start, goal,
                                     source.points_xy[53] - start, destination.points_xy[43] - goal, domain))
    elif case in {"x_junction", "t_junction", "y_junction"}:
        junction = geometry.junctions[0]; center = np.asarray(junction.point_xy, dtype=np.float64)
        by_id = {branch.branch_id: branch for branch in geometry.branches}
        anchors = {branch_id: _branch_anchor(by_id[branch_id].points_xy, center) for branch_id in junction.incident_branch_ids}
        target_pairs = {frozenset(pair) for pair in junction.continuation_relation}
        involved = [list(sample["branch_ids"]).index(branch_id) for branch_id in junction.incident_branch_ids]
        domain = binary_dilation(np.asarray(sample["branch_masks"])[involved].any(axis=0), iterations=1)
        for first, second in junction.continuation_relation:
            output.append(_candidate(
                sample, "x_correct" if case == "x_junction" else "ty_continuation", 1,
                f"{first}->{second}", anchors[first], anchors[second], center - anchors[first], center - anchors[second], domain,
            ))
        if case == "x_junction":
            incident = list(junction.incident_branch_ids)
            for position, first in enumerate(incident):
                for second in incident[position + 1:]:
                    if frozenset((first, second)) in target_pairs:
                        continue
                    output.append(_candidate(
                        sample, "x_wrong_turn", 0, f"{first}->{second}", anchors[first], anchors[second],
                        center - anchors[first], center - anchors[second], domain,
                    ))
    return output


def candidate_stream(split: str, *, image_size: int = 64) -> list[tuple[dict[str, Any], ContinuationCandidate]]:
    output = []
    for index in range(512):
        sample = generate_sample_v4(split, index, image_size=image_size)
        if sample["case"] in PRIMARY_CASES:
            output.extend((sample, candidate) for candidate in candidates_for_sample(sample))
    return output
