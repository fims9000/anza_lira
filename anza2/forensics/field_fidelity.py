"""Membership, axial, scale, hyperbolicity, and derived-geometry fidelity."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from models.anza2.field import ANZA2FieldOutput
from .component_replacement import REFERENCE_BASE_SCALE, REFERENCE_HYPERBOLICITY


def _quantiles(values: torch.Tensor) -> dict[str, float | None]:
    array = values.detach().cpu().numpy().astype(np.float64)
    if array.size == 0:
        return {name: None for name in ("min", "q10", "q25", "median", "q75", "q90", "max", "mean")}
    return {
        "min": float(array.min()), "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)), "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)), "q90": float(np.quantile(array, 0.90)),
        "max": float(array.max()), "mean": float(array.mean()),
    }


def field_fidelity_row(
    sample: dict[str, Any],
    learned: ANZA2FieldOutput,
    oracle: ANZA2FieldOutput,
    oracle_valid: torch.Tensor,
    *,
    seed: int,
    sample_index: int,
) -> dict[str, Any]:
    active = learned.membership >= 0.5
    target = oracle_valid.bool()
    tp = int((active & target).sum()); fp = int((active & ~target).sum()); fn = int((~active & target).sum())
    target_count = target.sum(dim=1)
    predicted_count = active.sum(dim=1)
    target_pixels = target_count > 0
    crossing_pixels = target_count >= 2

    similarity = (learned.orientation * oracle.orientation).sum(dim=2).clamp(-1.0, 1.0)
    orientation_error = (1.0 - similarity)[target]
    learned_base = learned.base_scale[target]
    learned_h = learned.hyperbolicity[target]

    # Evaluate the learned ellipse along and perpendicular to the generator axis
    # without reconstructing theta from the doubled-angle representation.
    target_c = oracle.orientation[:, :, 0]
    target_s = oracle.orientation[:, :, 1]
    learned_c = learned.orientation[:, :, 0]
    learned_s = learned.orientation[:, :, 1]
    dot2 = learned_c * target_c + learned_s * target_s
    lambda_parallel = learned.sigma_parallel.reciprocal().square()
    lambda_perpendicular = learned.sigma_perpendicular.reciprocal().square()
    m0 = 0.5 * (lambda_parallel + lambda_perpendicular)
    m1 = 0.5 * (lambda_parallel - lambda_perpendicular)
    g_along = torch.exp(-0.5 * (m0 + m1 * dot2))[target]
    g_perpendicular = torch.exp(-0.5 * (m0 - m1 * dot2))[target]
    ratio = (g_along + 1e-8) / (g_perpendicular + 1e-8)

    active_membership = learned.membership[target]
    inactive_membership = learned.membership[~target]
    base_stats = _quantiles(learned_base)
    h_stats = _quantiles(learned_h)
    orientation_stats = _quantiles(orientation_error)
    return {
        "seed": seed, "sample_index": sample_index, "case": sample["case"],
        "membership_tp": tp, "membership_fp": fp, "membership_fn": fn,
        "active_mode_recall": tp / max(tp + fn, 1),
        "active_mode_precision": tp / max(tp + fp, 1),
        "mean_active_mode_count_target_pixels": float(predicted_count[target_pixels].float().mean()) if target_pixels.any() else None,
        "one_mode_collapse_fraction_crossing": float((predicted_count[crossing_pixels] <= 1).float().mean()) if crossing_pixels.any() else None,
        "all_zero_fraction_target_pixels": float((predicted_count[target_pixels] == 0).float().mean()) if target_pixels.any() else None,
        "all_one_saturation_fraction_all_pixels": float((predicted_count == learned.num_modes).float().mean()),
        "membership_active_median": _quantiles(active_membership)["median"],
        "membership_inactive_median": _quantiles(inactive_membership)["median"],
        "orientation_error_median": orientation_stats["median"],
        "orientation_error_q90": orientation_stats["q90"],
        "orientation_catastrophic_fraction": float((orientation_error > 0.5).float().mean()) if orientation_error.numel() else None,
        "base_scale_mean": base_stats["mean"], "base_scale_median": base_stats["median"],
        "base_scale_q10": base_stats["q10"], "base_scale_q90": base_stats["q90"],
        "base_scale_cv": float(learned_base.std(unbiased=False) / learned_base.mean().clamp_min(1e-8)) if learned_base.numel() else None,
        "base_scale_reference_abs_deviation": float((learned_base - REFERENCE_BASE_SCALE).abs().mean()) if learned_base.numel() else None,
        "hyperbolicity_mean": h_stats["mean"], "hyperbolicity_median": h_stats["median"],
        "hyperbolicity_q10": h_stats["q10"], "hyperbolicity_q90": h_stats["q90"],
        "hyperbolicity_near_zero_fraction": float((learned_h <= 0.05).float().mean()) if learned_h.numel() else None,
        "hyperbolicity_near_max_fraction": float((learned_h >= 1.20).float().mean()) if learned_h.numel() else None,
        "hyperbolicity_reference_abs_deviation": float((learned_h - REFERENCE_HYPERBOLICITY).abs().mean()) if learned_h.numel() else None,
        "anisotropy_ratio_mean": float(torch.exp(2.0 * learned_h).mean()) if learned_h.numel() else None,
        "g_along_mean": float(g_along.mean()) if g_along.numel() else None,
        "g_perpendicular_mean": float(g_perpendicular.mean()) if g_perpendicular.numel() else None,
        "g_ratio_mean": float(ratio.mean()) if ratio.numel() else None,
    }


def aggregate_fidelity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = [key for key in rows[0] if key not in {"seed", "sample_index", "case"}]

    def aggregate(group: list[dict[str, Any]]) -> dict[str, float | None]:
        output = {}
        for key in numeric:
            values = [row[key] for row in group if row[key] is not None]
            output[key] = float(np.mean(values)) if values else None
        return output

    by_case = {case: aggregate([row for row in rows if row["case"] == case]) for case in sorted({row["case"] for row in rows})}
    return {"overall": aggregate(rows), "by_case": by_case, "sample_rows": len(rows)}
