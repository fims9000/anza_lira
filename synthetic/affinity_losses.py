"""Balanced edge supervision and the frozen-stage helpers for affinity repair."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.azconv_affinity import StructuralAffinityAZConv2d


def balanced_affinity_bce(
    logits: torch.Tensor,
    positive: torch.Tensor,
    hard_negative: torch.Tensor,
) -> torch.Tensor:
    positive = positive.bool()
    hard_negative = hard_negative.bool()
    losses: list[torch.Tensor] = []
    if positive.any():
        losses.append(F.binary_cross_entropy_with_logits(logits[positive], torch.ones_like(logits[positive])))
    if hard_negative.any():
        losses.append(F.binary_cross_entropy_with_logits(logits[hard_negative], torch.zeros_like(logits[hard_negative])))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def matched_hard_negative_ranking(
    logits: torch.Tensor,
    positive: torch.Tensor,
    hard_negative: torch.Tensor,
    *,
    margin: float = 0.5,
) -> torch.Tensor:
    pos = logits[positive.bool()]
    neg = logits[hard_negative.bool()]
    count = min(pos.numel(), neg.numel())
    if count == 0:
        return logits.sum() * 0.0
    return F.relu(float(margin) - pos[:count] + neg[:count]).mean()


def configure_affinity_stage1(module: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Freeze the full base path and return (trainable affinity, frozen base)."""

    for parameter in module.parameters():
        parameter.requires_grad_(False)
    trainable: list[nn.Parameter] = []
    for layer in module.modules():
        if isinstance(layer, StructuralAffinityAZConv2d):
            for child in (layer.context_encoder, layer.affinity_mlp):
                for parameter in child.parameters():
                    parameter.requires_grad_(True)
                    trainable.append(parameter)
            layer.raw_beta.requires_grad_(True)
            trainable.append(layer.raw_beta)
    frozen = [parameter for parameter in module.parameters() if not parameter.requires_grad]
    return trainable, frozen

