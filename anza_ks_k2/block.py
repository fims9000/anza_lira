"""Single-scale residual symbolic block used by all K2 structured controls."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .dense_features import FEATURE_WIDTH, dense_orientation_features


class SymbolicFeatureBlock(nn.Module):
    def __init__(self, channels: int, method: str, *, feature_mean: torch.Tensor | None = None, feature_std: torch.Tensor | None = None) -> None:
        super().__init__()
        self.method = method
        self.structural_projection = nn.Conv2d(channels, 1, 1)
        self.orientation_head = nn.Conv2d(channels, 8, 1)
        self.occupancy_head = nn.Conv2d(channels, 1, 1)
        self.readout = nn.Sequential(nn.Linear(FEATURE_WIDTH, 32), nn.GELU(), nn.Linear(32, 16))
        self.output_projection = nn.Conv2d(16, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(()))
        mean = torch.zeros(FEATURE_WIDTH) if feature_mean is None else torch.as_tensor(feature_mean, dtype=torch.float32)
        std = torch.ones(FEATURE_WIDTH) if feature_std is None else torch.as_tensor(feature_std, dtype=torch.float32)
        if mean.shape != (FEATURE_WIDTH,) or std.shape != (FEATURE_WIDTH,) or torch.any(std <= 0):
            raise ValueError("feature normalization must be positive 104-vectors")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        structural = self.structural_projection(x)
        features = dense_orientation_features(structural, self.method)
        normalized = (features - self.feature_mean) / self.feature_std
        responses = self.readout(normalized)
        orientation_logits = self.orientation_head(x)
        occupancy_logits = self.occupancy_head(x)
        evidence = torch.sigmoid(orientation_logits).permute(0, 2, 3, 1)[..., None]
        aggregate = (evidence * responses).sum(dim=-2) / evidence.sum(dim=-2).clamp_min(1e-6)
        aggregate = aggregate * torch.sigmoid(occupancy_logits).permute(0, 2, 3, 1)
        correction = self.output_projection(aggregate.permute(0, 3, 1, 2))
        output = x + self.gamma * correction
        if not return_aux:
            return output
        return output, {
            "structural_map": structural,
            "orientation_logits": orientation_logits,
            "occupancy_logits": occupancy_logits,
            "gamma": self.gamma,
            "feature_method": self.method,
        }
