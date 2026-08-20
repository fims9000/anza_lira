"""Fixed axial orientation bank and generator-derived fuzzy targets."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F


def axial_angles(num_orientations: int = 8) -> np.ndarray:
    if num_orientations < 2:
        raise ValueError("at least two axial bins are required")
    return np.arange(num_orientations, dtype=np.float32) * (math.pi / num_orientations)


def orientation_bank_targets(
    sample: dict[str, Any], *, num_orientations: int = 8, sigma_theta: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Return MxHxW fuzzy targets and a reliable visible-axis mask."""

    if sigma_theta <= 0:
        raise ValueError("sigma_theta must be positive")
    masks = np.asarray(sample["branch_masks"], dtype=bool)
    centerlines = np.asarray(sample["branch_centerlines"], dtype=bool)
    cos2 = np.asarray(sample["branch_tangent_cos2"], dtype=np.float32)
    sin2 = np.asarray(sample["branch_tangent_sin2"], dtype=np.float32)
    if masks.shape != centerlines.shape or masks.shape != cos2.shape or masks.shape != sin2.shape:
        raise ValueError("branch target tensors must have matching BxHxW shapes")
    height, width = masks.shape[1:]
    target = np.zeros((num_orientations, height, width), dtype=np.float32)
    angles = axial_angles(num_orientations)
    bank_cos2 = np.cos(2.0 * angles); bank_sin2 = np.sin(2.0 * angles)
    for branch_mask, centerline, branch_cos2, branch_sin2 in zip(masks, centerlines, cos2, sin2, strict=True):
        if not centerline.any():
            continue
        _, indices = distance_transform_edt(~centerline, return_indices=True)
        nearest_cos2 = branch_cos2[indices[0], indices[1]]
        nearest_sin2 = branch_sin2[indices[0], indices[1]]
        axial_similarity = (
            bank_cos2[:, None, None] * nearest_cos2[None]
            + bank_sin2[:, None, None] * nearest_sin2[None]
        )
        distance = 1.0 - np.clip(axial_similarity, -1.0, 1.0)
        branch_target = np.exp(-distance / (2.0 * sigma_theta**2)).astype(np.float32)
        target[:, branch_mask] = np.maximum(target[:, branch_mask], branch_target[:, branch_mask])
    valid = masks.any(axis=0)
    target[:, ~valid] = 0.0
    return target, valid


def orientation_bank_loss(
    logits: list[torch.Tensor], target: torch.Tensor, valid: torch.Tensor, *, background_weight: float = 0.25,
) -> torch.Tensor:
    if not logits:
        return target.new_zeros(())
    if target.ndim != 4 or valid.ndim != 4 or valid.shape[1] != 1:
        raise ValueError("target must be BxMxHxW and valid Bx1xHxW")
    losses = []
    for prediction in logits:
        local_target = F.interpolate(target, size=prediction.shape[-2:], mode="bilinear", align_corners=False)
        local_valid = F.interpolate(valid.float(), size=prediction.shape[-2:], mode="nearest").bool()
        element = F.binary_cross_entropy_with_logits(prediction, local_target, reduction="none")
        positive = element.masked_select(local_valid.expand_as(element))
        background = element.masked_select((~local_valid).expand_as(element))
        positive_loss = positive.mean() if positive.numel() else element.new_zeros(())
        background_loss = background.mean() if background.numel() else element.new_zeros(())
        losses.append(positive_loss + float(background_weight) * background_loss)
    return torch.stack(losses).mean()
