"""Residual ANZA-2 block with a zero-initialized geometric branch."""

from __future__ import annotations

import torch
from torch import nn

from .aggregation import ANZA2Aggregation, ANZA2AggregationOutput
from .field import ANZA2Field, ANZA2FieldConfig, ANZA2FieldOutput


class ANZA2Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        *,
        config: ANZA2FieldConfig | None = None,
        tau0: float = 1.0,
    ) -> None:
        super().__init__()
        out_channels = int(out_channels or in_channels)
        self.field = ANZA2Field(in_channels, config=config)
        self.aggregation = ANZA2Aggregation(
            in_channels,
            out_channels,
            num_modes=self.field.config.num_modes,
            tau0=tau0,
        )
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(
        self, features: torch.Tensor, *, return_diagnostics: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, ANZA2FieldOutput, ANZA2AggregationOutput]:
        field = self.field(features)
        aggregation = self.aggregation(features, field)
        output = self.residual(features) + self.gamma * aggregation.projected
        if return_diagnostics:
            return output, field, aggregation
        return output
