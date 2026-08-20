"""Calibration-only selection of one system-level NONE threshold."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..protocol import PROTOCOL
from .metrics import relation_metrics, source_decisions


def calibration_curve(sources: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = np.asarray([float(row["score"]) for row in candidates], dtype=float)
    thresholds = np.unique(np.concatenate(([0.0], scores, [np.nextafter(scores.max(), np.inf) if len(scores) else 1.0])))
    rows = []
    for threshold in thresholds:
        metrics = relation_metrics(source_decisions(sources, candidates, float(threshold)))
        rows.append({"threshold": float(threshold), **metrics})
    return rows


def calibrate_threshold(sources: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    curve = calibration_curve(sources, candidates)
    selector = PROTOCOL["selector"]
    feasible = [row for row in curve if row["FalseBridge"] <= float(selector["false_bridge_max"]) and row["WrongBranch"] <= float(selector["wrong_branch_max"])]
    if not feasible:
        return {"status": "STOP_P0_OPERATING_POINT_INFEASIBLE", "selected": None, "curve": curve}
    selected = max(feasible, key=lambda row: (row["RelationRecovery"], -row["FalseBridge"], -row["WrongBranch"], row["threshold"]))
    return {"status": "CALIBRATION_FEASIBLE", "selected": selected, "curve": curve}
