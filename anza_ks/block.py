"""Minimal residual ANZA-KS readout contract for later K2, not trained in K0/K1."""

from __future__ import annotations

import torch
from torch import nn


class ANZAKSResidualReadout(nn.Module):
    """Shared orientation readout over precomputed symbolic features."""

    def __init__(self, channels: int, feature_width: int, orientation_count: int = 8) -> None:
        super().__init__()
        self.orientation_count = int(orientation_count)
        self.shared_readout = nn.Linear(feature_width, channels, bias=False)
        self.output_projection = nn.Linear(channels, channels, bias=False)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, symbolic: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        if symbolic.shape[-2] != self.orientation_count or evidence.shape[-1] != self.orientation_count:
            raise ValueError("orientation bank mismatch")
        responses = self.shared_readout(symbolic)
        weights = evidence.sigmoid().unsqueeze(-1)
        aggregated = (weights * responses).sum(dim=-2) / (weights.sum(dim=-2) + 1e-6)
        correction = self.output_projection(aggregated)
        return x + self.gamma * correction
