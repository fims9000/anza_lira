"""Generator-lineage mechanism observations for frozen ANZA-2 relations."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation

from models.anza2.affinity import LOCAL8_OFFSETS
from structural.widest_path import domain_restricted_widest_path
from synthetic.affinity_targets import _shift, build_affinity_targets


def _junction_category(junction_type: str) -> str:
    if junction_type == "x_crossing":
        return "X"
    if junction_type == "t_intersection":
        return "T"
    if junction_type == "y_branch":
        return "Y"
    return junction_type


def branch_observations(sample: dict[str, Any], relation: np.ndarray) -> list[dict[str, Any]]:
    branch_ids = list(sample.get("branch_ids", range(1, len(sample["branch_centerlines"]) + 1)))
    by_id = {int(branch_id): index for index, branch_id in enumerate(branch_ids)}
    rows = []
    for junction_index, junction in enumerate(sample.get("junctions", [])):
        center_x, center_y = junction["point_xy"]
        center = (int(round(center_y)), int(round(center_x)))
        category = _junction_category(str(junction["junction_type"]))
        for branch_id in junction["incident_branch_ids"]:
            centerline = np.asarray(sample["branch_centerlines"][by_id[int(branch_id)]], dtype=bool)
            ys, xs = np.nonzero(centerline)
            candidates = []
            for y, x in zip(ys, xs):
                dy, dx = int(y - center[0]), int(x - center[1])
                distance = max(abs(dx), abs(dy))
                if 1 <= distance <= 4:
                    candidates.append((distance, int(y), int(x)))
            if not candidates:
                continue
            _distance, y, x = min(candidates)
            dx, dy = int(np.sign(x - center[1])), int(np.sign(y - center[0]))
            if (dx, dy) not in LOCAL8_OFFSETS:
                continue
            channel = LOCAL8_OFFSETS.index((dx, dy))
            rows.append({
                "kind": "branch", "subtype": category, "junction_index": junction_index,
                "branch_id": int(branch_id), "score": float(relation[channel, center[0], center[1]]),
            })
    return rows


def _gap_score(sample: dict[str, Any], relation: np.ndarray) -> float | None:
    gaps = sample.get("gaps", [])
    if not gaps:
        return None
    endpoints_xy = np.asarray(gaps[0]["endpoint_xy"], dtype=np.float64)
    start = (int(round(endpoints_xy[0, 1])), int(round(endpoints_xy[0, 0])))
    goal = (int(round(endpoints_xy[1, 1])), int(round(endpoints_xy[1, 0])))
    support = np.asarray(sample["latent_fault_mask"], dtype=bool) | np.asarray(sample["negative_gap_mask"], dtype=bool)
    domain = binary_dilation(support, iterations=1)
    score, _path = domain_restricted_widest_path(
        relation, start, goal, domain=domain, offsets=LOCAL8_OFFSETS
    )
    return float(score)


def _centerline_edges(sample: dict[str, Any]) -> np.ndarray:
    branches = np.asarray(sample["branch_centerlines"], dtype=bool)
    rows = []
    for dx, dy in LOCAL8_OFFSETS:
        shifted, valid = _shift(branches, dx, dy)
        rows.append(np.any(branches & shifted, axis=0) & valid)
    return np.stack(rows)


def mechanism_observations(sample: dict[str, Any], relation: np.ndarray) -> list[dict[str, Any]]:
    rows = branch_observations(sample, relation)
    gap_score = _gap_score(sample, relation)
    if gap_score is not None:
        rows.append({
            "kind": "gap_path", "subtype": "positive" if sample["case"] == "fault_with_gap" else "negative",
            "score": gap_score,
        })
    targets = build_affinity_targets(sample, LOCAL8_OFFSETS)
    hard = targets["affinity_hard_negative"]
    if hard.any() and ("crossing" in sample["case"] or "junction" in sample["case"]):
        for value in relation[hard]:
            rows.append({"kind": "crossing_negative_edge", "subtype": sample["case"], "score": float(value)})
    if sample["case"] == "near_parallel" and hard.any():
        for value in relation[hard]:
            rows.append({"kind": "parallel_negative_edge", "subtype": "near_parallel", "score": float(value)})
    if sample["case"] == "curved_fault":
        for value in relation[_centerline_edges(sample)]:
            rows.append({"kind": "curved_trace_edge", "subtype": "curved_fault", "score": float(value)})
    return rows


def aggregate_mechanism(rows: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    def selected(kind: str, subtype: str | None = None) -> list[float]:
        return [
            float(row["score"]) for row in rows
            if row["kind"] == kind and (subtype is None or row.get("subtype") == subtype)
        ]

    def recall(kind: str, subtype: str | None = None) -> float | None:
        values = selected(kind, subtype)
        return float(np.mean(np.asarray(values) >= threshold)) if values else None

    branch = selected("branch")
    positive_gap = selected("gap_path", "positive")
    negative_gap = selected("gap_path", "negative")
    crossing_negative = selected("crossing_negative_edge")
    parallel_negative = selected("parallel_negative_edge")
    return {
        "absolute_threshold": float(threshold),
        "overall_branch_recall": float(np.mean(np.asarray(branch) >= threshold)) if branch else None,
        "x_branch_recall": recall("branch", "X"),
        "t_branch_recall": recall("branch", "T"),
        "y_branch_recall": recall("branch", "Y"),
        "straight_gap_recovery": float(np.mean(np.asarray(positive_gap) >= threshold)) if positive_gap else None,
        "curved_gap_recovery": None,
        "curved_gap_status": "NOT_AVAILABLE_IN_FROZEN_V4_DEVELOPMENT; curved-trace edge recall reported instead",
        "curved_trace_edge_recall": recall("curved_trace_edge"),
        "parallel_fault_false_bridge": float(np.mean(np.asarray(negative_gap) >= threshold)) if negative_gap else None,
        "near_parallel_negative_edge_fpr": float(np.mean(np.asarray(parallel_negative) >= threshold)) if parallel_negative else None,
        "crossing_false_bridge_edge_fpr": float(np.mean(np.asarray(crossing_negative) >= threshold)) if crossing_negative else None,
        "branch_observations": len(branch), "positive_gap_observations": len(positive_gap),
        "negative_gap_observations": len(negative_gap), "crossing_negative_edges": len(crossing_negative),
    }
