"""Frozen identifiable tasks and geometry-only leakage controls for Phase A2."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation

from anza_s.cases import ContinuationCandidate, candidate_stream
from anza_s.oracle_field import geometry_for_sample
from synthetic.crossing_trace_bench_v4 import generate_sample_v4


def _unit(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _curved_confusers(split: str, image_size: int) -> list[tuple[dict[str, Any], ContinuationCandidate]]:
    output = []
    for index in range(256, 512):
        sample = generate_sample_v4(split, index, image_size=image_size)
        if sample["case"] != "close_non_intersecting":
            continue
        geometry = geometry_for_sample(sample)
        first, second = geometry.branches
        domain = binary_dilation(np.asarray(sample["branch_masks"], dtype=bool).any(axis=0), iterations=4)
        for reverse in (False, True):
            source, destination = (first, second) if not reverse else (second, first)
            start = np.asarray(source.points_xy[43], dtype=np.float64)
            goal = np.asarray(destination.points_xy[53], dtype=np.float64)
            candidate = ContinuationCandidate(
                split=split, index=index, case="close_non_intersecting", task="curved_confuser",
                label=0, pair_id=f"curve-cross:{int(reverse)}", start_xy=start, goal_xy=goal,
                start_direction=_unit(source.points_xy[53] - start),
                goal_direction=_unit(destination.points_xy[43] - goal), domain=np.asarray(domain, dtype=bool),
            )
            output.append((sample, candidate))
    return output


def a2_candidate_stream(split: str, *, image_size: int = 64) -> list[tuple[dict[str, Any], ContinuationCandidate]]:
    if split not in {"train", "validation"}:
        raise PermissionError("Phase A2 may access only synthetic train and validation")
    allowed = {
        "x_correct", "x_wrong_turn", "parallel_correct", "parallel_wrong",
        "curved_gap", "straight_gap", "negative_gap",
    }
    primary = [(sample, candidate) for sample, candidate in candidate_stream(split, image_size=image_size) if candidate.task in allowed]
    return primary + _curved_confusers(split, image_size)


def descriptor(candidate: ContinuationCandidate) -> dict[str, float]:
    distance = float(np.linalg.norm(candidate.goal_xy - candidate.start_xy))
    axial = float(abs(candidate.start_direction @ candidate.goal_direction))
    return {"endpoint_distance": distance, "endpoint_axial_agreement": axial}


def curved_comparability(rows: list[tuple[dict[str, Any], ContinuationCandidate]]) -> dict[str, Any]:
    positive = [descriptor(candidate) for _, candidate in rows if candidate.task == "curved_gap"]
    negative = [descriptor(candidate) for _, candidate in rows if candidate.task == "curved_confuser"]
    if not positive or not negative:
        raise ValueError("curved task requires positive and negative cases")
    pos_distance = float(np.median([item["endpoint_distance"] for item in positive]))
    neg_distance = float(np.median([item["endpoint_distance"] for item in negative]))
    ratio = neg_distance / pos_distance
    pos_axis = float(np.median([item["endpoint_axial_agreement"] for item in positive]))
    neg_axis = float(np.median([item["endpoint_axial_agreement"] for item in negative]))
    passed = 0.5 <= ratio <= 2.0 and abs(pos_axis - neg_axis) <= 0.25
    return {
        "positive_count": len(positive), "negative_count": len(negative),
        "positive_median_endpoint_distance": pos_distance,
        "negative_median_endpoint_distance": neg_distance,
        "negative_to_positive_distance_ratio": ratio,
        "positive_median_axial_agreement": pos_axis,
        "negative_median_axial_agreement": neg_axis,
        "absolute_axial_agreement_difference": abs(pos_axis - neg_axis),
        "frozen_rule": "distance ratio in [0.5,2.0] and axial-agreement difference <=0.25",
        "primary_eligible": bool(passed),
    }
