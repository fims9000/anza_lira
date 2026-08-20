"""Separate observable local-mode targets from privileged latent continuation."""

from __future__ import annotations

from typing import Any

import numpy as np


def split_visible_and_latent_targets(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    theta = np.asarray(sample["gt_theta_set"], dtype=np.float32)
    valid = np.asarray(sample["gt_theta_valid"], dtype=bool)
    visible = np.asarray(sample["visible_fault_mask"], dtype=bool)
    positive_gap = np.asarray(sample["positive_gap_mask"], dtype=bool)
    if valid.shape[1:] != visible.shape or theta.shape != valid.shape:
        raise ValueError("mode targets and visible mask must share spatial shape")

    visible_valid = valid & visible[None]
    latent_gap_valid = valid & positive_gap[None]
    privileged_local_valid = valid & ~visible[None]
    visible_count = visible_valid.sum(axis=0).astype(np.uint8)
    latent_count = latent_gap_valid.sum(axis=0).astype(np.uint8)
    if np.any(visible_valid & latent_gap_valid):
        raise AssertionError("visible local and latent gap targets must be disjoint")
    if np.any(latent_gap_valid & ~privileged_local_valid):
        raise AssertionError("latent gap targets must be outside visible evidence")
    return {
        "visible_theta_set": theta.copy(),
        "visible_theta_valid": visible_valid,
        "visible_mode_count": visible_count,
        "latent_continuation_theta_set": theta.copy(),
        "latent_continuation_theta_valid": latent_gap_valid,
        "latent_continuation_mode_count": latent_count,
        "privileged_local_theta_valid": privileged_local_valid,
        "unobserved_non_gap_theta_valid": privileged_local_valid & ~positive_gap[None],
    }


def target_audit_row(sample: dict[str, Any]) -> dict[str, int | bool]:
    targets = split_visible_and_latent_targets(sample)
    original = np.asarray(sample["gt_theta_valid"], dtype=bool)
    return {
        "original_target_axes": int(original.sum()),
        "visible_target_axes": int(targets["visible_theta_valid"].sum()),
        "latent_gap_target_axes": int(targets["latent_continuation_theta_valid"].sum()),
        "unobserved_non_gap_target_axes": int(targets["unobserved_non_gap_theta_valid"].sum()),
        "visible_latent_overlap_axes": int((targets["visible_theta_valid"] & targets["latent_continuation_theta_valid"]).sum()),
        "privileged_gap_local_supervision_removed": bool(
            not np.any(targets["visible_theta_valid"] & np.asarray(sample["positive_gap_mask"])[None])
        ),
    }

