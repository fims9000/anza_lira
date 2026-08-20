"""Stable/unstable five-lobe local feature operators."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import axial_bank, frozen_foliation_geometry
from .kernels import five_lobe_kernels


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class SharedOrientationTransform(nn.Module):
    """One transform applied to every orientation by folding M into the batch."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        if q.ndim != 5:
            raise ValueError("orientation responses must be BxMx3CxHxW")
        batch, modes, channels3, height, width = q.shape
        result = self.net(q.reshape(batch * modes, channels3, height, width))
        return result.reshape(batch, modes, -1, height, width)


class FoliationConvBase(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        num_orientations: int = 8,
        kernel_size: int = 9,
        base_scale: float = 1.5,
        hyperbolicity: float = 0.35,
        free_geometry: bool,
    ) -> None:
        super().__init__()
        if channels <= 0 or num_orientations < 2 or kernel_size <= 0 or kernel_size % 2 != 1:
            raise ValueError("invalid foliation configuration")
        self.channels = int(channels)
        self.num_orientations = int(num_orientations)
        self.kernel_size = int(kernel_size)
        self.base_scale = float(base_scale)
        self.hyperbolicity = float(hyperbolicity)
        self.free_geometry = bool(free_geometry)
        angles, _, _ = axial_bank(num_orientations)
        self.register_buffer("angles", angles)
        initial = frozen_foliation_geometry(base_scale, hyperbolicity)
        if free_geometry:
            floors = (0.20, 0.20, 0.10, 0.10)
            names = ("raw_sigma_u", "raw_sigma_s", "raw_delta_u", "raw_delta_s")
            for name, value, floor in zip(names, initial, floors, strict=True):
                self.register_parameter(name, nn.Parameter(torch.tensor(_inverse_softplus(value - floor))))
        else:
            for name in ("raw_sigma_u", "raw_sigma_s", "raw_delta_u", "raw_delta_s"):
                self.register_parameter(name, None)
        self.evidence_head = nn.Conv2d(channels, num_orientations, kernel_size=3, padding=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.shared_psi = SharedOrientationTransform(channels)
        self.gamma = nn.Parameter(torch.zeros(()))

    def geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.free_geometry:
            values = frozen_foliation_geometry(self.base_scale, self.hyperbolicity)
            return tuple(self.angles.new_tensor(value) for value in values)  # type: ignore[return-value]
        return (
            0.20 + F.softplus(self.raw_sigma_u),
            0.20 + F.softplus(self.raw_sigma_s),
            0.10 + F.softplus(self.raw_delta_u),
            0.10 + F.softplus(self.raw_delta_s),
        )

    def kernels(self) -> torch.Tensor:
        sigma_u, sigma_s, delta_u, delta_s = self.geometry()
        return five_lobe_kernels(
            self.angles,
            sigma_u=sigma_u,
            sigma_s=sigma_s,
            delta_u=delta_u,
            delta_s=delta_s,
            kernel_size=self.kernel_size,
        )

    def structural_responses(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return C, U-C and C-S as BxMxCxHxW."""
        kernels = self.kernels()
        weight = kernels[None].expand(self.channels, -1, -1, -1, -1).reshape(
            self.channels * self.num_orientations * 5, 1, self.kernel_size, self.kernel_size
        )
        filtered = F.conv2d(value, weight, padding=self.kernel_size // 2, groups=self.channels)
        batch, _, height, width = filtered.shape
        lobes = filtered.reshape(batch, self.channels, self.num_orientations, 5, height, width).permute(0, 2, 1, 3, 4, 5)
        center = lobes[:, :, :, 0]
        unstable = 0.5 * (lobes[:, :, :, 1] + lobes[:, :, :, 2])
        stable = 0.5 * (lobes[:, :, :, 3] + lobes[:, :, :, 4])
        return center, unstable - center, center - stable

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, dict[str, Any]]:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("foliation input must be BxCxHxW with configured channels")
        logits = self.evidence_head(x)
        evidence = torch.sigmoid(logits)
        center, longitudinal, transverse = self.structural_responses(self.value(x))
        q = torch.cat((center, longitudinal, transverse), dim=2)
        per_orientation = self.shared_psi(q)
        correction = (evidence[:, :, None] * per_orientation).sum(dim=1) / evidence.sum(dim=1, keepdim=True).clamp_min(1e-8)
        output = x + self.gamma * correction
        if not return_aux:
            return output, logits
        entropy = -(evidence.clamp(1e-6, 1 - 1e-6) * torch.log(evidence.clamp(1e-6, 1 - 1e-6)) + (1 - evidence).clamp(1e-6) * torch.log((1 - evidence).clamp(1e-6))).mean()
        return output, {
            "orientation_logits": logits,
            "center": center,
            "longitudinal_minus_center": longitudinal,
            "center_minus_transverse": transverse,
            "evidence_entropy": entropy,
            "gamma": self.gamma,
            "geometry": self.geometry(),
        }


class FreeFoliationConv(FoliationConvBase):
    def __init__(self, channels: int, **kwargs: Any) -> None:
        super().__init__(channels, free_geometry=True, **kwargs)


class ANZAFoliationConv(FoliationConvBase):
    def __init__(self, channels: int, **kwargs: Any) -> None:
        super().__init__(channels, free_geometry=False, **kwargs)
