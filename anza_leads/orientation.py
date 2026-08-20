"""Crowd-positive local-PCA axial targets and equal auxiliary loss."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def crowd_orientation_targets(
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    radius: int = 5,
    num_orientations: int = 8,
    sigma_theta: float = 0.20,
    min_neighbors: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local PCA only at explicit blue/green positive pixels.

    Inputs are Ax1xHxW. Returned tensors are AxMxHxW fuzzy axial targets
    and Ax1xHxW confidence weights. Orange and white are always invalid.
    """
    if targets.ndim != 4 or targets.shape != weights.shape or targets.shape[1] != 1:
        raise ValueError("orientation inputs must be matching Ax1xHxW tensors")
    if radius <= 0 or min_neighbors < 2:
        raise ValueError("invalid local PCA parameters")
    positive = ((targets > 0.5) & (weights > 0)).to(dtype=torch.float32)
    size = 2 * radius + 1
    coordinates = torch.arange(-radius, radius + 1, dtype=targets.dtype, device=targets.device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")

    def local_sum(kernel: torch.Tensor) -> torch.Tensor:
        return F.conv2d(positive, kernel.view(1, 1, size, size), padding=radius)

    ones = torch.ones_like(xx)
    count = local_sum(ones).clamp_min(1.0)
    mean_x = local_sum(xx) / count
    mean_y = local_sum(yy) / count
    covariance_xx = local_sum(xx * xx) / count - mean_x.square()
    covariance_yy = local_sum(yy * yy) / count - mean_y.square()
    covariance_xy = local_sum(xx * yy) / count - mean_x * mean_y
    delta = covariance_xx - covariance_yy
    denominator = torch.sqrt(delta.square() + 4.0 * covariance_xy.square()).clamp_min(1e-6)
    cos2 = delta / denominator
    sin2 = 2.0 * covariance_xy / denominator
    angles = torch.arange(num_orientations, dtype=targets.dtype, device=targets.device) * (math.pi / num_orientations)
    similarity = (
        torch.cos(2.0 * angles)[None, :, None, None] * cos2
        + torch.sin(2.0 * angles)[None, :, None, None] * sin2
    ).clamp(-1.0, 1.0)
    bank = torch.exp(-(1.0 - similarity) / (2.0 * sigma_theta**2))
    valid = positive.bool() & (count >= float(min_neighbors))
    confidence = torch.where(valid, weights, torch.zeros_like(weights))
    return bank * valid.to(bank.dtype), confidence


def crowd_orientation_loss(
    logits: list[torch.Tensor], target: torch.Tensor, confidence: torch.Tensor
) -> torch.Tensor:
    """Mean independently normalized positive-trace loss over stages/annotators."""
    if not logits:
        raise ValueError("equal orientation auxiliary requires stage logits")
    if target.ndim != 4 or confidence.ndim != 4 or confidence.shape[1] != 1:
        raise ValueError("invalid orientation target shapes")
    stage_losses = []
    for prediction in logits:
        local_target = F.interpolate(target, size=prediction.shape[-2:], mode="bilinear", align_corners=False)
        local_weight = F.interpolate(confidence, size=prediction.shape[-2:], mode="nearest")
        annotator_losses = []
        for ann_target, ann_weight in zip(local_target, local_weight, strict=True):
            expanded = ann_weight.expand_as(ann_target)
            if float(expanded.sum().detach()) <= 0:
                continue
            element = F.binary_cross_entropy_with_logits(prediction, ann_target.unsqueeze(0), reduction="none")[0]
            annotator_losses.append((element * expanded).sum() / expanded.sum().clamp_min(1e-8))
        if not annotator_losses:
            stage_losses.append(prediction.sum() * 0.0)
        else:
            stage_losses.append(torch.stack(annotator_losses).mean())
    return torch.stack(stage_losses).mean()
