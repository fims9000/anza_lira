"""Frozen axial stable/unstable geometry for ANZA-FS H3."""

from __future__ import annotations

import math

import torch


def axial_bank(num_orientations: int = 8, *, device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_orientations < 2:
        raise ValueError("at least two axial orientations are required")
    angles = torch.arange(num_orientations, device=device, dtype=dtype) * (math.pi / num_orientations)
    unstable = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    stable = torch.stack((-torch.sin(angles), torch.cos(angles)), dim=-1)
    return angles, unstable, stable


def reciprocal_scales(base_scale: float = 1.5, hyperbolicity: float = 0.35) -> tuple[float, float]:
    if base_scale <= 0 or hyperbolicity < 0:
        raise ValueError("base_scale must be positive and hyperbolicity non-negative")
    return base_scale * math.exp(hyperbolicity), base_scale * math.exp(-hyperbolicity)


def frozen_foliation_geometry(base_scale: float = 1.5, hyperbolicity: float = 0.35) -> tuple[float, float, float, float]:
    sigma_u, sigma_s = reciprocal_scales(base_scale, hyperbolicity)
    return sigma_u, sigma_s, 1.5 * sigma_u, 1.5 * sigma_s
