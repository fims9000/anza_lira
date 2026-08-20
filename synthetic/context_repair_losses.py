"""Frozen supervision terms for the B1-B3 context-repair cycle."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .structural_losses import structural_gap_loss


def contextual_gate_loss(
    gate_probability: torch.Tensor,
    gate_target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    positive_weight: float = 4.0,
    focal_gamma: float = 2.0,
    dice_weight: float = 0.25,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    probability = torch.as_tensor(gate_probability)
    target = torch.as_tensor(gate_target, device=probability.device, dtype=probability.dtype)
    valid = torch.as_tensor(valid_mask, device=probability.device, dtype=torch.bool)
    if probability.shape != target.shape or probability.shape != valid.shape:
        raise ValueError("gate probability, target, and valid mask must have equal shape")
    if not torch.isfinite(probability).all() or torch.any((probability < 0) | (probability > 1)):
        raise ValueError("gate probability must be finite and in [0,1]")
    selected_probability = probability[valid].clamp(epsilon, 1.0 - epsilon)
    selected_target = target[valid]
    if selected_probability.numel() == 0:
        return probability.sum() * 0.0
    positive = selected_target
    negative = 1.0 - selected_target
    focal = -(
        float(positive_weight) * positive * (1.0 - selected_probability).pow(focal_gamma) * selected_probability.log()
        + negative * selected_probability.pow(focal_gamma) * (1.0 - selected_probability).log()
    ).mean()
    intersection = (selected_probability * selected_target).sum()
    dice = 1.0 - (2.0 * intersection + epsilon) / (
        selected_probability.sum() + selected_target.sum() + epsilon
    )
    return focal + float(dice_weight) * dice


def matched_negative_route_loss(
    continuation_logits: torch.Tensor,
    continuation_target: torch.Tensor,
    eligible: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Mode-specific multi-positive InfoNCE over declared matched negatives."""
    if float(temperature) not in {0.1, 0.2}:
        raise ValueError("frozen route temperature must be 0.1 or 0.2")
    logits = torch.as_tensor(continuation_logits)
    target = torch.as_tensor(continuation_target, device=logits.device, dtype=torch.bool)
    eligible_mask = torch.as_tensor(eligible, device=logits.device, dtype=torch.bool)
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("continuation logits must be square")
    if target.shape != logits.shape or eligible_mask.shape != logits.shape:
        raise ValueError("route target and eligibility must match logits")
    if torch.any(target & ~eligible_mask):
        raise ValueError("positive continuation outside eligible set")
    rows = target.any(dim=-1) & (eligible_mask & ~target).any(dim=-1)
    if not rows.any():
        return logits.sum() * 0.0
    scaled = logits[rows] / float(temperature)
    scaled = scaled.masked_fill(~eligible_mask[rows], torch.finfo(logits.dtype).min)
    log_probability = torch.log_softmax(scaled, dim=-1)
    positives = target[rows].to(dtype=logits.dtype)
    positives /= positives.sum(dim=-1, keepdim=True)
    return -(positives * log_probability).sum(dim=-1).mean()


def paired_gap_logit_loss(
    completion_logits: torch.Tensor,
    positive_gap_mask: torch.Tensor,
    negative_gap_mask: torch.Tensor,
    *,
    negative_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return structural_gap_loss(
        torch.sigmoid(completion_logits),
        positive_gap_mask,
        negative_gap_mask,
        negative_weight=negative_weight,
    )


def effective_mode_count(membership: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    mu = membership / membership.sum(dim=1, keepdim=True).clamp_min(epsilon)
    return torch.exp(-(mu * mu.clamp_min(epsilon).log()).sum(dim=1))


def mode_cardinality_diagnostics(
    membership: torch.Tensor,
    gt_mode_count: torch.Tensor,
) -> dict[str, torch.Tensor]:
    effective = effective_mode_count(membership)
    truth = torch.as_tensor(gt_mode_count, device=membership.device)
    if truth.ndim == 4 and truth.shape[1] == 1:
        truth = truth[:, 0]
    if effective.shape != truth.shape:
        raise ValueError("membership and gt mode count must share batch/spatial shape")
    selected = truth > 0
    zero = effective.sum() * 0.0
    if not selected.any():
        return {"mode_count_accuracy": zero, "neff_mae": zero}
    predicted_count = effective.round().clamp(1, membership.shape[1]).to(dtype=truth.dtype)
    return {
        "mode_count_accuracy": (predicted_count[selected] == truth[selected]).float().mean(),
        "neff_mae": (effective[selected] - truth[selected].float()).abs().mean(),
    }


def normalized_routing_entropy(probabilities: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
    probability = torch.as_tensor(probabilities)
    mask = torch.as_tensor(eligible, device=probability.device, dtype=torch.bool)
    rows = mask.sum(dim=-1) > 1
    if not rows.any():
        return probability.sum() * 0.0
    selected = probability[rows].clamp_min(1e-8)
    selected_mask = mask[rows]
    entropy = -(selected * selected.log() * selected_mask).sum(dim=-1)
    scale = selected_mask.sum(dim=-1).float().log().clamp_min(math.log(2.0))
    return (entropy / scale).mean()
