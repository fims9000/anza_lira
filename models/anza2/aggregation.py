"""Per-mode self-mass aggregation for ANZA-2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .affinity import LOCAL8_OFFSETS, _shift_neighbor, shift_field
from .field import ANZA2FieldOutput
from .geometry import directed_geometry, directed_step_support


@dataclass(frozen=True)
class ANZA2AggregationOutput:
    projected: torch.Tensor
    mode_features: torch.Tensor
    self_mass: torch.Tensor
    neighbor_mass: torch.Tensor


def aggregate_modes(
    values: torch.Tensor,
    field: ANZA2FieldOutput,
    *,
    tau0: float = 1.0,
    offsets: Iterable[tuple[int, int]] = LOCAL8_OFFSETS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mode features and exactly normalized self/neighbor masses."""

    if values.ndim != 4:
        raise ValueError("values must be BxCxHxW")
    if tau0 <= 0:
        raise ValueError("tau0 must be positive")
    offset_list = tuple((int(dx), int(dy)) for dx, dy in offsets)
    transitions = []
    neighbor_values = []
    for dx, dy in offset_list:
        neighbor_field, valid = shift_field(field, dx, dy)
        center_geometry = directed_geometry(field, (dx, dy))
        reverse_support = directed_step_support(neighbor_field, (-dx, -dy)).unsqueeze(1)
        transitions.append(center_geometry * reverse_support * valid.to(center_geometry.dtype))
        shifted_values, value_valid = _shift_neighbor(values, dx, dy)
        neighbor_values.append(shifted_values * value_valid.to(shifted_values.dtype))
    transition = torch.stack(transitions, dim=2)  # B,R,K,H,W
    neighbors = torch.stack(neighbor_values, dim=2)  # B,C,K,H,W
    denominator = float(tau0) + transition.sum(dim=2)
    self_mass = float(tau0) / denominator
    neighbor_mass = transition / denominator.unsqueeze(2)
    transported = torch.einsum("brkhw,bckhw->brchw", neighbor_mass, neighbors)
    mode_features = field.membership.unsqueeze(2) * (
        self_mass.unsqueeze(2) * values.unsqueeze(1) + transported
    )
    return mode_features, self_mass, neighbor_mass


class ANZA2Aggregation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_modes: int = 4,
        tau0: float = 1.0,
        offsets: Iterable[tuple[int, int]] = LOCAL8_OFFSETS,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, num_modes) < 1:
            raise ValueError("channel and mode counts must be positive")
        if tau0 <= 0:
            raise ValueError("tau0 must be positive")
        self.num_modes = int(num_modes)
        self.tau0 = float(tau0)
        self.offsets = tuple(offsets)
        self.value = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.projection = nn.Conv2d(num_modes * out_channels, out_channels, kernel_size=1)

    def forward(self, features: torch.Tensor, field: ANZA2FieldOutput) -> ANZA2AggregationOutput:
        if field.num_modes != self.num_modes:
            raise ValueError("field mode count does not match aggregation")
        values = self.value(features)
        modes, self_mass, neighbor_mass = aggregate_modes(
            values, field, tau0=self.tau0, offsets=self.offsets
        )
        batch, mode_count, channels, height, width = modes.shape
        projected = self.projection(modes.reshape(batch, mode_count * channels, height, width))
        return ANZA2AggregationOutput(projected, modes, self_mass, neighbor_mass)
