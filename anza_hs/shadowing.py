"""Shadowing-stability primitives reserved for H2 after an H1 pass."""

from __future__ import annotations

import math

import torch


def symmetric_shadow_distance(
    first_xy: torch.Tensor, first_axis: torch.Tensor, second_xy: torch.Tensor, second_axis: torch.Tensor,
    *, sigma_x: float = 1.5, eta_theta: float = 2.0, temperature: float = 0.25,
) -> torch.Tensor:
    if first_xy.ndim != 2 or second_xy.ndim != 2 or first_xy.shape[1] != 2 or second_xy.shape[1] != 2:
        raise ValueError("trajectory positions must be Nx2")
    if first_axis.shape != first_xy.shape or second_axis.shape != second_xy.shape or sigma_x <= 0 or temperature <= 0:
        raise ValueError("trajectory axes must match positions and parameters must be positive")
    delta = first_xy[:, None, :] - second_xy[None, :, :]
    spatial = delta.square().sum(dim=-1) / float(sigma_x**2)
    dot = torch.einsum("ic,jc->ij", first_axis, second_axis)
    cost = spatial + float(eta_theta) * (1.0 - dot.square())
    forward = (-temperature * torch.logsumexp(-cost / temperature, dim=1)).mean()
    backward = (-temperature * torch.logsumexp(-cost / temperature, dim=0)).mean()
    return 0.5 * (forward + backward)


def inverse_map_points(points_xy: torch.Tensor, inverse_affine: torch.Tensor) -> torch.Tensor:
    if points_xy.ndim != 2 or points_xy.shape[1] != 2 or inverse_affine.shape != (3, 3):
        raise ValueError("points must be Nx2 and inverse affine 3x3")
    homogeneous = torch.cat([points_xy, torch.ones_like(points_xy[:, :1])], dim=1)
    mapped = homogeneous @ inverse_affine.T
    return mapped[:, :2] / mapped[:, 2:].clamp_min(1e-8)


def top_axial_peaks(evidence: torch.Tensor, *, max_peaks: int = 2) -> torch.Tensor:
    """Select distinct fixed-bank peaks without assigning persistent mode IDs."""
    if evidence.ndim != 1 or max_peaks not in {1, 2}:
        raise ValueError("evidence must be one bank vector and max_peaks 1 or 2")
    count = evidence.numel(); first = int(torch.argmax(evidence))
    if max_peaks == 1:
        return torch.tensor([first], device=evidence.device)
    distance = torch.arange(count, device=evidence.device)
    circular = torch.minimum((distance - first).abs(), count - (distance - first).abs())
    allowed = circular >= max(1, count // 4)
    masked = evidence.masked_fill(~allowed, -torch.inf)
    return torch.tensor([first, int(torch.argmax(masked))], device=evidence.device)


def bank_axes(indices: torch.Tensor, num_orientations: int) -> torch.Tensor:
    theta = indices.float() * (math.pi / num_orientations)
    return torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
