"""Separate structural supervision terms for controlled synthetic truth."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.azconv_v2 import axial_distance


def visible_segmentation_loss(
    logits: torch.Tensor,
    visible_target: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    dice_weight: float = 1.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """BCE + Dice against observed evidence only, never latent geometry."""
    target = torch.as_tensor(visible_target, device=logits.device, dtype=logits.dtype)
    if logits.shape != target.shape:
        raise ValueError("Visible logits and target must have equal shape")
    valid = torch.ones_like(target) if valid_mask is None else torch.as_tensor(
        valid_mask, device=logits.device, dtype=logits.dtype
    )
    if valid.shape != target.shape:
        raise ValueError("valid_mask must match visible target")
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(epsilon)
    probability = torch.sigmoid(logits)
    intersection = (probability * target * valid).sum()
    denominator = (probability * valid).sum() + (target * valid).sum()
    dice_loss = 1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)
    return bce + float(dice_weight) * dice_loss


def structural_gap_loss(
    completion_probability: torch.Tensor,
    positive_gap_mask: torch.Tensor,
    negative_gap_mask: torch.Tensor,
    *,
    negative_weight: float = 1.0,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Positive completion plus matched-negative false-bridge control."""
    probability = torch.as_tensor(completion_probability)
    positive = torch.as_tensor(positive_gap_mask, device=probability.device, dtype=torch.bool)
    negative = torch.as_tensor(negative_gap_mask, device=probability.device, dtype=torch.bool)
    if probability.shape != positive.shape or probability.shape != negative.shape:
        raise ValueError("Completion probability and gap masks must have equal shape")
    if not torch.isfinite(probability).all() or torch.any((probability < 0) | (probability > 1)):
        raise ValueError("Completion probability must be finite and in [0,1]")
    zero = probability.sum() * 0.0
    positive_loss = -torch.log(probability[positive].clamp_min(epsilon)).mean() if positive.any() else zero
    negative_loss = (
        -torch.log((1.0 - probability[negative]).clamp_min(epsilon)).mean()
        if negative.any()
        else zero
    )
    total = positive_loss + float(negative_weight) * negative_loss
    return total, {
        "positive_gap_loss": positive_loss,
        "negative_gap_loss": negative_loss,
    }


def routing_supervision_loss(
    continuation_logits: torch.Tensor,
    continuation_target: torch.Tensor,
    eligible: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Multi-positive branch continuation cross-entropy.

    Each source branch is supervised only when it has at least one lineage
    continuation. A Y parent may have two correct destinations; their target
    probability is shared uniformly. Ineligible relations never enter the
    normalization.
    """
    logits = torch.as_tensor(continuation_logits)
    target = torch.as_tensor(continuation_target, device=logits.device, dtype=torch.bool)
    eligible_mask = torch.as_tensor(eligible, device=logits.device, dtype=torch.bool)
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("Continuation logits must be a square branch matrix")
    if target.shape != logits.shape or eligible_mask.shape != logits.shape:
        raise ValueError("Continuation target and eligibility must match logits")
    if not torch.isfinite(logits).all():
        raise ValueError("Continuation logits must be finite")
    if torch.any(target & ~eligible_mask):
        raise ValueError("Every target continuation must be eligible at a declared junction")

    valid_source = target.any(dim=-1)
    if not valid_source.any():
        return logits.sum() * 0.0
    masked_logits = logits.masked_fill(~eligible_mask, torch.finfo(logits.dtype).min)
    log_probability = torch.log_softmax(masked_logits[valid_source], dim=-1)
    target_rows = target[valid_source].to(dtype=logits.dtype)
    target_distribution = target_rows / target_rows.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    return -(target_distribution * log_probability).sum(dim=-1).mean()


def branch_transition_logits(
    transport: torch.Tensor,
    branch_masks: torch.Tensor,
    *,
    variant: str,
    kernel_size: int = 3,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Aggregate pixel/mode transport into differentiable branch-pair logits."""
    transition = torch.as_tensor(transport)
    masks = torch.as_tensor(branch_masks, device=transition.device, dtype=transition.dtype)
    if masks.ndim != 3:
        raise ValueError("branch_masks must be NxHxW for one generated sample")
    branches, height, width = masks.shape
    locations = height * width
    patch_area = kernel_size**2
    if variant == "v2a":
        if transition.ndim != 5:
            raise ValueError("V2A transport must be BxRxRxKxL")
        marginal = transition.sum(dim=(1, 2))
    elif variant == "v2b":
        if transition.ndim != 7:
            raise ValueError("V2B transport must be BxRx2xRx2xKxL")
        marginal = transition.sum(dim=(1, 2, 3, 4))
    else:
        raise ValueError("variant must be v2a or v2b")
    if transition.shape[0] != 1 or marginal.shape != (1, patch_area, locations):
        raise ValueError("Branch aggregation currently requires one sample and matching spatial size")
    source = F.unfold(
        masks.unsqueeze(1), kernel_size, padding=kernel_size // 2
    ).reshape(branches, patch_area, locations)
    destination = masks.reshape(branches, locations)
    pair_support = source[:, None] * destination[None, :, None, :]
    score = (pair_support * marginal[0].unsqueeze(0).unsqueeze(0)).sum(dim=(2, 3))
    denominator = pair_support.sum(dim=(2, 3)).clamp_min(epsilon)
    return torch.log((score / denominator).clamp_min(epsilon))


def cone_half_angle(
    junction_score: torch.Tensor,
    *,
    alpha_straight: float,
    alpha_junction: float,
) -> torch.Tensor:
    if alpha_straight < 0 or alpha_junction < alpha_straight:
        raise ValueError("Cone angles require 0 <= alpha_straight <= alpha_junction")
    score = torch.as_tensor(junction_score)
    if not torch.isfinite(score).all() or torch.any((score < 0) | (score > 1)):
        raise ValueError("Junction score must be finite and in [0,1]")
    return float(alpha_straight) + (float(alpha_junction) - float(alpha_straight)) * score


def cone_consistency_values(
    membership: torch.Tensor,
    theta: torch.Tensor,
    junction_score: torch.Tensor,
    *,
    kernel_size: int = 3,
    alpha_straight: float = 0.17453292519943295,
    alpha_junction: float = 0.7853981633974483,
    kappa_cone: float = 4.0,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return c_r(p,q) and a valid-neighbor mask as BxRxKxHxW."""
    if membership.shape != theta.shape or membership.ndim != 4:
        raise ValueError("membership and theta must be matching BxRxHxW tensors")
    if kernel_size <= 0 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be positive and odd")
    batch, modes, height, width = membership.shape
    locations = height * width
    patch_area = kernel_size**2
    normalized = membership / membership.sum(dim=1, keepdim=True).clamp_min(epsilon)
    neighbor_membership = F.unfold(
        normalized,
        kernel_size,
        padding=kernel_size // 2,
    ).reshape(batch, modes, patch_area, locations)
    neighbor_theta = F.unfold(
        theta,
        kernel_size,
        padding=kernel_size // 2,
    ).reshape(batch, modes, patch_area, locations)
    center_theta = theta.reshape(batch, modes, 1, 1, locations)
    neighbor_theta = neighbor_theta.reshape(batch, 1, modes, patch_area, locations)
    distance = axial_distance(center_theta, neighbor_theta)
    alpha = cone_half_angle(
        junction_score,
        alpha_straight=alpha_straight,
        alpha_junction=alpha_junction,
    ).reshape(batch, 1, 1, 1, locations)
    compatibility = torch.exp(
        -float(kappa_cone) * torch.relu(distance - alpha).square()
    )
    values = (
        neighbor_membership.reshape(batch, 1, modes, patch_area, locations) * compatibility
    ).sum(dim=2)
    valid = F.unfold(
        torch.ones(batch, 1, height, width, device=membership.device, dtype=membership.dtype),
        kernel_size,
        padding=kernel_size // 2,
    ).reshape(batch, 1, patch_area, height, width)
    return values.reshape(batch, modes, patch_area, height, width), valid.expand(-1, modes, -1, -1, -1)


def cone_consistency_loss(
    membership: torch.Tensor,
    theta: torch.Tensor,
    junction_score: torch.Tensor,
    fault_probability: torch.Tensor,
    *,
    geometry_weight: torch.Tensor | None = None,
    kernel_size: int = 3,
    alpha_straight: float = 0.17453292519943295,
    alpha_junction: float = 0.7853981633974483,
    kappa_cone: float = 4.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    values, valid = cone_consistency_values(
        membership,
        theta,
        junction_score,
        kernel_size=kernel_size,
        alpha_straight=alpha_straight,
        alpha_junction=alpha_junction,
        kappa_cone=kappa_cone,
        epsilon=epsilon,
    )
    batch, modes, patch_area, height, width = values.shape
    if fault_probability.shape != (batch, 1, height, width):
        raise ValueError("fault_probability must be Bx1xHxW")
    neighbor_fault = F.unfold(
        fault_probability,
        kernel_size,
        padding=kernel_size // 2,
    ).reshape(batch, 1, patch_area, height, width)
    normalized = membership / membership.sum(dim=1, keepdim=True).clamp_min(epsilon)
    weights = (
        fault_probability.unsqueeze(2)
        * neighbor_fault
        * normalized.unsqueeze(2)
        * valid
    )
    if geometry_weight is not None:
        geometry = torch.as_tensor(geometry_weight, device=weights.device, dtype=weights.dtype)
        if geometry.shape != weights.shape:
            raise ValueError("geometry_weight must match BxRxKxHxW cone values")
        weights = weights * geometry
    return -(weights * torch.log(values.clamp_min(epsilon))).sum() / weights.sum().clamp_min(epsilon)
