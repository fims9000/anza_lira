"""Shared differentiable normalized corridor construction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def hyperbolic_distance(offset_yx: torch.Tensor, tangent_yx: torch.Tensor, h: float | torch.Tensor) -> torch.Tensor:
    tangent = tangent_yx / torch.linalg.vector_norm(tangent_yx, dim=-1, keepdim=True).clamp_min(1e-8)
    normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)
    du = (offset_yx * tangent).sum(-1); ds = (offset_yx * normal).sum(-1); h_value = torch.as_tensor(h, dtype=offset_yx.dtype, device=offset_yx.device)
    return torch.exp(-2.0 * h_value) * du.square() + torch.exp(2.0 * h_value) * ds.square()


def corridor_grid(source_yx: torch.Tensor, destination_yx: torch.Tensor, *, image_hw: tuple[int, int], output_hw: tuple[int, int] = (32, 64), cross_extent: float = 16.0, padding: float = 12.0) -> torch.Tensor:
    source = source_yx.to(torch.float32); destination = destination_yx.to(torch.float32); vector = destination - source; distance = torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-6); along = vector / distance; across = torch.stack((-along[:, 1], along[:, 0]), dim=-1); midpoint = 0.5 * (source + destination)
    height, width = output_hw; longitudinal = torch.linspace(-1, 1, width, device=source.device)[None, :] * (0.5 * distance + padding); transverse = torch.linspace(-cross_extent, cross_extent, height, device=source.device)[:, None]
    points = midpoint[:, None, None, :] + longitudinal[:, None, :, None] * along[:, None, None, :] + transverse[None, :, :, None] * across[:, None, None, :]
    image_height, image_width = image_hw; x = 2.0 * points[..., 1] / max(image_width - 1, 1) - 1.0; y = 2.0 * points[..., 0] / max(image_height - 1, 1) - 1.0
    return torch.stack((x, y), dim=-1)


def extract_corridors(dense: torch.Tensor, source_yx: torch.Tensor, destination_yx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return BKx10x32x64 corridors and grids from Bx8xHxW dense maps."""
    batch, candidates = destination_yx.shape[:2]; flat_destination = destination_yx.reshape(-1, 2); flat_source = source_yx[:, None].expand(-1, candidates, -1).reshape(-1, 2)
    grid = corridor_grid(flat_source, flat_destination, image_hw=dense.shape[-2:]); repeated = dense[:, None].expand(-1, candidates, -1, -1, -1).reshape(-1, *dense.shape[1:])
    sampled = F.grid_sample(repeated, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
    height, width = sampled.shape[-2:]; yy, xx = torch.meshgrid(torch.arange(height, device=dense.device), torch.arange(width, device=dense.device), indexing="ij")
    source_marker = torch.exp(-((xx - 12.0) ** 2 + (yy - (height - 1) / 2) ** 2) / 6.0); destination_marker = torch.exp(-((xx - (width - 13.0)) ** 2 + (yy - (height - 1) / 2) ** 2) / 6.0)
    markers = torch.stack((source_marker, destination_marker))[None].expand(len(sampled), -1, -1, -1)
    return torch.cat((sampled, markers), dim=1), grid


def anza_q_bias(source_tangent_yx: torch.Tensor, source_yx: torch.Tensor, destination_yx: torch.Tensor, *, h: float = 0.35, token_hw: tuple[int, int] = (4, 8)) -> torch.Tensor:
    """Frozen hyperbolic Q at corridor-token centers in pair-normalized coordinates."""
    batch, candidates = destination_yx.shape[:2]; tangent = source_tangent_yx[:, None].expand(-1, candidates, -1).reshape(-1, 2); delta = (destination_yx - source_yx[:, None]).reshape(-1, 2); distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-6); pair_axis = delta / distance
    # Axial tangent in corridor frame: dot with along/across.
    across = torch.stack((-pair_axis[:, 1], pair_axis[:, 0]), dim=-1); u_x = (tangent * pair_axis).sum(-1); u_y = (tangent * across).sum(-1); norm = torch.sqrt(u_x.square() + u_y.square()).clamp_min(1e-6); u_x /= norm; u_y /= norm
    ys = torch.linspace(-1, 1, token_hw[0], device=delta.device); xs = torch.linspace(-1, 1, token_hw[1], device=delta.device); yy, xx = torch.meshgrid(ys, xs, indexing="ij"); dx = xx.reshape(1, -1) + 1.0; dy = yy.reshape(1, -1)
    du = dx * u_x[:, None] + dy * u_y[:, None]; ds = -dx * u_y[:, None] + dy * u_x[:, None]
    q = torch.exp(torch.tensor(-2.0 * h, device=delta.device)) * du.square() + torch.exp(torch.tensor(2.0 * h, device=delta.device)) * ds.square()
    return q.reshape(batch, candidates, -1)
