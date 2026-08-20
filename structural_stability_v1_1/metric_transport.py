"""Forward-Jacobian SPD congruence under the parent's output-to-input warp."""

from __future__ import annotations

import numpy as np
import torch

from structural_stability_v1_1.matrix_log import spd_matrix_log


def output_to_input_jacobian_xy(displacement_yx: np.ndarray) -> np.ndarray:
    field = np.asarray(displacement_yx, dtype=np.float64)
    if field.ndim != 3 or field.shape[0] != 2:
        raise ValueError("displacement must have shape 2xHxW in y,x order")
    dy_dy, dy_dx = np.gradient(field[0])
    dx_dy, dx_dx = np.gradient(field[1])
    # Matrix coordinates are x,y even though stored displacement is y,x.
    return np.stack(
        (
            np.stack((1.0 + dx_dx, dx_dy), axis=-1),
            np.stack((dy_dx, 1.0 + dy_dy), axis=-1),
        ),
        axis=-2,
    )


def forward_jacobian_xy(displacement_yx: np.ndarray) -> np.ndarray:
    """A=D(clean->warped)=(D phi_output_to_input)^-1 at output coordinates."""
    backward = output_to_input_jacobian_xy(displacement_yx)
    determinant = np.linalg.det(backward)
    if np.any(determinant <= 0):
        raise ValueError("warp has non-positive output-to-input Jacobian")
    return np.linalg.inv(backward)


def area_normalize_jacobian(jacobian: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if jacobian.shape[-2:] != (2, 2):
        raise ValueError("Jacobian must end in 2x2")
    determinant = torch.linalg.det(jacobian)
    if torch.any(determinant <= 0):
        raise ValueError("Jacobian determinant must be positive")
    return jacobian / torch.sqrt(determinant.clamp_min(eps))[..., None, None]


def transport_metric(metric: torch.Tensor, forward_jacobian: torch.Tensor) -> torch.Tensor:
    """Apply C'=Abar C Abar^T with matrices in their final two axes."""
    normalized = area_normalize_jacobian(forward_jacobian)
    return normalized @ metric @ normalized.transpose(-1, -2)


def metric_equivariance_loss(predicted: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Weighted log-Euclidean loss; target is explicitly stop-gradient."""
    difference = spd_matrix_log(predicted) - spd_matrix_log(target.detach())
    squared = torch.sum(difference.square(), dim=(1, 2))
    weights = weight.to(dtype=squared.dtype)
    return torch.sum(weights * squared) / (torch.sum(weights) + eps)
