"""Compact, claim-neutral diagnostics for ANZA-2 fields."""

from __future__ import annotations

from typing import Any

import torch

from .field import ANZA2FieldOutput


@torch.inference_mode()
def summarize_field(field: ANZA2FieldOutput, *, active_threshold: float = 0.5) -> dict[str, Any]:
    membership = field.membership
    active = (membership >= float(active_threshold)).sum(dim=1).float()
    anisotropy = field.sigma_parallel / field.sigma_perpendicular.clamp_min(1e-8)
    return {
        "membership_mean": float(membership.mean()),
        "membership_min": float(membership.min()),
        "membership_max": float(membership.max()),
        "active_mode_mean": float(active.mean()),
        "base_scale_mean": float(field.base_scale.mean()),
        "hyperbolicity_mean": float(field.hyperbolicity.mean()),
        "anisotropy_ratio_mean": float(anisotropy.mean()),
        "all_finite": bool(all(torch.isfinite(value).all() for value in (
            membership, field.orientation, field.base_scale, field.hyperbolicity,
            field.sigma_parallel, field.sigma_perpendicular,
        ))),
    }
