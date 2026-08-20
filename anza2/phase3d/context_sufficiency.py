"""Compare the frozen encoder receptive field with generator structure scales."""

from __future__ import annotations

from typing import Any

import numpy as np


ENCODER_EFFECTIVE_RF_PX = 11


def _summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "q50": None, "q90": None, "max": None}
    return {
        "count": int(len(array)),
        "q50": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def context_sufficiency(rows: list[dict[str, Any]], *, split: str = "train") -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    gap = _summary([float(row["gap_length_px"]) for row in selected if row["gap_length_px"] not in (None, "")])
    junction = _summary([float(row["junction_diameter_px"]) for row in selected if row["junction_diameter_px"] not in (None, "")])
    parallel = _summary([
        float(row["parallel_separation_px"]) for row in selected
        if row["case"] in {"near_parallel", "close_non_intersecting"} and row["parallel_separation_px"] not in (None, "")
    ])
    curvature = _summary([
        float(row["curvature_dispersion"]) for row in selected
        if "curved" in str(row["case"]) and row["curvature_dispersion"] not in (None, "")
    ])
    required_q90 = max(value for value in (gap["q90"], junction["q90"], parallel["q90"]) if value is not None)
    sufficient = ENCODER_EFFECTIVE_RF_PX >= required_q90
    return {
        "split": split,
        "encoder_effective_receptive_field_px": ENCODER_EFFECTIVE_RF_PX,
        "gap_length_px": gap,
        "junction_diameter_px": junction,
        "parallel_separation_px": parallel,
        "curvature_dispersion_dimensionless": curvature,
        "required_context_q90_px": required_q90,
        "context_sufficient_for_primary_local_scales": sufficient,
        "architecture_action": "UNCHANGED" if sufficient else "ONE_SHARED_CONTEXT_BLOCK_AUTHORIZED_BEFORE_TRAINING",
        "interpretation": "RF sufficiency is geometric scale coverage, not proof that hidden-gap local membership is observable.",
    }

