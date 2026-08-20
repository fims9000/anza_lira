"""Learned multimodal axial hyperbolic fuzzy field for ANZA-2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ANZA2FieldConfig:
    num_modes: int = 4
    ell_min: float = 0.25
    h_max: float = 1.25
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.num_modes < 1:
            raise ValueError("num_modes must be positive")
        if self.ell_min <= 0:
            raise ValueError("ell_min must be positive")
        if self.h_max < 0:
            raise ValueError("h_max must be nonnegative")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


@dataclass(frozen=True)
class ANZA2FieldOutput:
    membership: torch.Tensor
    orientation: torch.Tensor
    base_scale: torch.Tensor
    hyperbolicity: torch.Tensor
    sigma_parallel: torch.Tensor
    sigma_perpendicular: torch.Tensor

    @property
    def num_modes(self) -> int:
        return int(self.membership.shape[1])


def normalize_doubled_angle(raw: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Normalize ``[..., 2, H, W]`` doubled-angle vectors with a safe axis fallback."""

    if raw.ndim < 3 or raw.shape[-3] != 2:
        raise ValueError("orientation tensor must have a size-2 component axis")
    norm = torch.linalg.vector_norm(raw, dim=-3, keepdim=True)
    fallback = torch.zeros_like(raw)
    fallback.select(-3, 0).fill_(1.0)
    return torch.where(norm > eps, raw / norm.clamp_min(eps), fallback)


def field_from_raw(
    membership_logits: torch.Tensor,
    raw_orientation: torch.Tensor,
    raw_base_scale: torch.Tensor,
    raw_hyperbolicity: torch.Tensor,
    *,
    config: ANZA2FieldConfig,
) -> ANZA2FieldOutput:
    """Apply the exact ANZA-2 field parameterization to caller-provided tensors."""

    if membership_logits.ndim != 4:
        raise ValueError("membership logits must be BxRxHxW")
    expected = membership_logits.shape
    if raw_orientation.shape != (expected[0], expected[1], 2, expected[2], expected[3]):
        raise ValueError("raw orientation must be BxRx2xHxW")
    if raw_base_scale.shape != expected or raw_hyperbolicity.shape != expected:
        raise ValueError("scale and hyperbolicity tensors must match memberships")
    membership = torch.sigmoid(membership_logits)
    orientation = normalize_doubled_angle(raw_orientation, eps=config.eps)
    base_scale = float(config.ell_min) + F.softplus(raw_base_scale)
    hyperbolicity = float(config.h_max) * torch.sigmoid(raw_hyperbolicity)
    sigma_parallel = base_scale * torch.exp(hyperbolicity)
    sigma_perpendicular = base_scale * torch.exp(-hyperbolicity)
    return ANZA2FieldOutput(
        membership=membership,
        orientation=orientation,
        base_scale=base_scale,
        hyperbolicity=hyperbolicity,
        sigma_parallel=sigma_parallel,
        sigma_perpendicular=sigma_perpendicular,
    )


class ANZA2Field(nn.Module):
    """Predict independent memberships, doubled axes, and reciprocal scales."""

    def __init__(self, in_channels: int, config: ANZA2FieldConfig | None = None) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        self.config = config or ANZA2FieldConfig()
        modes = self.config.num_modes
        self.membership_head = nn.Conv2d(in_channels, modes, kernel_size=1)
        self.orientation_head = nn.Conv2d(in_channels, 2 * modes, kernel_size=1)
        self.base_scale_head = nn.Conv2d(in_channels, modes, kernel_size=1)
        self.hyperbolicity_head = nn.Conv2d(in_channels, modes, kernel_size=1)
        self._initialize_orientation_bias()

    def _initialize_orientation_bias(self) -> None:
        modes = self.config.num_modes
        with torch.no_grad():
            bias = self.orientation_head.bias.view(modes, 2)
            angles = torch.arange(modes, dtype=bias.dtype, device=bias.device) * torch.pi / modes
            bias[:, 0] = torch.cos(2.0 * angles)
            bias[:, 1] = torch.sin(2.0 * angles)

    def forward(self, features: torch.Tensor) -> ANZA2FieldOutput:
        if features.ndim != 4:
            raise ValueError("features must be BxCxHxW")
        batch, _channels, height, width = features.shape
        modes = self.config.num_modes
        raw_orientation = self.orientation_head(features).view(batch, modes, 2, height, width)
        return field_from_raw(
            self.membership_head(features),
            raw_orientation,
            self.base_scale_head(features),
            self.hyperbolicity_head(features),
            config=self.config,
        )
