"""Capacity-matched fixed-bank local orientation operators."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class OrientationBankConv(nn.Module):
    """Residual fixed-bank convolution with isolated scale parameterization."""

    def __init__(
        self, channels: int, *, kind: str, num_orientations: int = 8, kernel_size: int = 9,
        base_scale: float = 1.5, hyperbolicity: float = 0.35,
    ) -> None:
        super().__init__()
        if kind not in {"isotropic", "generic", "hyperbolic"}:
            raise ValueError(f"unknown bank-convolution kind: {kind}")
        if kernel_size <= 0 or kernel_size % 2 != 1 or num_orientations < 2 or base_scale <= 0:
            raise ValueError("invalid orientation-bank configuration")
        self.channels = int(channels); self.kind = kind; self.num_orientations = int(num_orientations)
        self.kernel_size = int(kernel_size); self.base_scale = float(base_scale); self.hyperbolicity = float(hyperbolicity)
        angles = torch.arange(num_orientations, dtype=torch.float32) * (math.pi / num_orientations)
        self.register_buffer("angles", angles)
        radius = kernel_size // 2
        yy, xx = torch.meshgrid(torch.arange(-radius, radius + 1, dtype=torch.float32), torch.arange(-radius, radius + 1, dtype=torch.float32), indexing="ij")
        self.register_buffer("offset_x", xx); self.register_buffer("offset_y", yy)
        initial_u = base_scale * math.exp(hyperbolicity); initial_s = base_scale * math.exp(-hyperbolicity)
        if kind == "generic":
            floor = 0.20
            self.raw_sigma_u = nn.Parameter(torch.full((num_orientations,), _inverse_softplus(initial_u - floor)))
            self.raw_sigma_s = nn.Parameter(torch.full((num_orientations,), _inverse_softplus(initial_s - floor)))
        else:
            self.register_parameter("raw_sigma_u", None); self.register_parameter("raw_sigma_s", None)
        self.evidence_head = nn.Conv2d(channels, num_orientations, kernel_size=3, padding=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.fuse = nn.Conv2d(channels * num_orientations, channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(()))

    def scales(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kind == "isotropic":
            sigma = self.angles.new_full((self.num_orientations,), self.base_scale)
            return sigma, sigma
        if self.kind == "hyperbolic":
            return (
                self.angles.new_full((self.num_orientations,), self.base_scale * math.exp(self.hyperbolicity)),
                self.angles.new_full((self.num_orientations,), self.base_scale * math.exp(-self.hyperbolicity)),
            )
        return 0.20 + F.softplus(self.raw_sigma_u), 0.20 + F.softplus(self.raw_sigma_s)

    def kernels(self) -> torch.Tensor:
        sigma_u, sigma_s = self.scales()
        cosine = torch.cos(self.angles)[:, None, None]; sine = torch.sin(self.angles)[:, None, None]
        along = self.offset_x[None] * cosine + self.offset_y[None] * sine
        transverse = -self.offset_x[None] * sine + self.offset_y[None] * cosine
        kernel = torch.exp(-0.5 * ((along / sigma_u[:, None, None]) ** 2 + (transverse / sigma_s[:, None, None]) ** 2))
        return kernel / kernel.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("orientation-bank input must be BxCxHxW with configured channels")
        logits = self.evidence_head(x); evidence = torch.sigmoid(logits)
        value = self.value(x); kernels = self.kernels()
        modes = []
        for index in range(self.num_orientations):
            weight = kernels[index].view(1, 1, self.kernel_size, self.kernel_size).expand(self.channels, 1, -1, -1)
            filtered = F.conv2d(value, weight, padding=self.kernel_size // 2, groups=self.channels)
            modes.append(filtered * evidence[:, index : index + 1])
        correction = self.fuse(torch.cat(modes, dim=1))
        return x + self.gamma * correction, logits


class IsotropicOrientConv(OrientationBankConv):
    def __init__(self, channels: int, **kwargs) -> None:
        super().__init__(channels, kind="isotropic", **kwargs)


class GenericAnisoConv(OrientationBankConv):
    def __init__(self, channels: int, **kwargs) -> None:
        super().__init__(channels, kind="generic", **kwargs)


class ANZAHyperbolicConv(OrientationBankConv):
    def __init__(self, channels: int, **kwargs) -> None:
        super().__init__(channels, kind="hyperbolic", **kwargs)
