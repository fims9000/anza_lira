"""Five-lobe Gaussian kernels with explicit longitudinal and transverse centers."""

from __future__ import annotations

import torch

from .geometry import axial_bank


LOBE_NAMES = ("center", "unstable_plus", "unstable_minus", "stable_plus", "stable_minus")


def five_lobe_kernels(
    angles: torch.Tensor,
    *,
    sigma_u: torch.Tensor,
    sigma_s: torch.Tensor,
    delta_u: torch.Tensor,
    delta_s: torch.Tensor,
    kernel_size: int = 9,
) -> torch.Tensor:
    """Return normalized kernels with shape Mx5xKxK."""

    if kernel_size <= 0 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be positive and odd")
    modes = int(angles.numel())
    values = []
    for value, name in ((sigma_u, "sigma_u"), (sigma_s, "sigma_s"), (delta_u, "delta_u"), (delta_s, "delta_s")):
        value = torch.as_tensor(value, device=angles.device, dtype=angles.dtype).reshape(-1)
        if value.numel() not in {1, modes} or torch.any(value <= 0):
            raise ValueError(f"{name} must be positive scalar or length-M tensor")
        values.append(value.expand(modes))
    sigma_u, sigma_s, delta_u, delta_s = values
    _, unstable, stable = axial_bank(modes, device=angles.device, dtype=angles.dtype)
    centers = torch.stack(
        (
            torch.zeros_like(unstable),
            delta_u[:, None] * unstable,
            -delta_u[:, None] * unstable,
            delta_s[:, None] * stable,
            -delta_s[:, None] * stable,
        ),
        dim=1,
    )
    radius = kernel_size // 2
    yy, xx = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=angles.device, dtype=angles.dtype),
        torch.arange(-radius, radius + 1, device=angles.device, dtype=angles.dtype),
        indexing="ij",
    )
    dx = xx[None, None] - centers[..., 0, None, None]
    dy = yy[None, None] - centers[..., 1, None, None]
    cosine = torch.cos(angles)[:, None, None, None]
    sine = torch.sin(angles)[:, None, None, None]
    along = dx * cosine + dy * sine
    transverse = -dx * sine + dy * cosine
    kernel = torch.exp(
        -0.5
        * (
            (along / sigma_u[:, None, None, None]) ** 2
            + (transverse / sigma_s[:, None, None, None]) ** 2
        )
    )
    return kernel / kernel.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def kernel_centroids(kernels: torch.Tensor) -> torch.Tensor:
    if kernels.ndim != 4 or kernels.shape[1] != 5 or kernels.shape[-1] != kernels.shape[-2]:
        raise ValueError("kernels must have shape Mx5xKxK")
    radius = kernels.shape[-1] // 2
    yy, xx = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=kernels.device, dtype=kernels.dtype),
        torch.arange(-radius, radius + 1, device=kernels.device, dtype=kernels.dtype),
        indexing="ij",
    )
    return torch.stack(((kernels * xx).sum((-2, -1)), (kernels * yy).sum((-2, -1))), dim=-1)
