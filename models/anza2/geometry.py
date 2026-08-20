"""Exact doubled-angle reciprocal-scale geometry used by ANZA-2."""

from __future__ import annotations

import torch

from .field import ANZA2FieldOutput


def hyperbolic_shape_matrix(field: ANZA2FieldOutput) -> torch.Tensor:
    """Return determinant-one shape matrices as ``B,R,H,W,2,2``."""

    c = field.orientation[:, :, 0]
    s = field.orientation[:, :, 1]
    cos_sq = (1.0 + c) / 2.0
    sin_sq = (1.0 - c) / 2.0
    sin_cos = s / 2.0
    stretch = torch.exp(field.hyperbolicity)
    shrink = torch.exp(-field.hyperbolicity)
    h00 = stretch * cos_sq + shrink * sin_sq
    h11 = stretch * sin_sq + shrink * cos_sq
    h01 = (stretch - shrink) * sin_cos
    return torch.stack((h00, h01, h01, h11), dim=-1).reshape(*h00.shape, 2, 2)


def quadratic_form(field: ANZA2FieldOutput, displacement: tuple[float, float] | torch.Tensor) -> torch.Tensor:
    """Evaluate Q for ``(dx, dy)=q-p`` without recovering an angle."""

    if isinstance(displacement, torch.Tensor):
        if displacement.numel() != 2:
            raise ValueError("displacement tensor must contain dx and dy")
        dx, dy = displacement.reshape(-1)[0], displacement.reshape(-1)[1]
    else:
        dx, dy = float(displacement[0]), float(displacement[1])
    lam_parallel = field.base_scale.reciprocal().square() * torch.exp(-2.0 * field.hyperbolicity)
    lam_perpendicular = field.base_scale.reciprocal().square() * torch.exp(2.0 * field.hyperbolicity)
    m0 = (lam_parallel + lam_perpendicular) / 2.0
    m1 = (lam_parallel - lam_perpendicular) / 2.0
    c = field.orientation[:, :, 0]
    s = field.orientation[:, :, 1]
    radial = dx * dx + dy * dy
    axial = c * (dx * dx - dy * dy) + 2.0 * s * dx * dy
    return (m0 * radial + m1 * axial).clamp_min(0.0)


def directed_geometry(field: ANZA2FieldOutput, displacement: tuple[float, float] | torch.Tensor) -> torch.Tensor:
    """Return the literal standard Gaussian ``exp(-Q/2)`` for every mode."""

    return torch.exp(-0.5 * quadratic_form(field, displacement))


def directed_step_support(field: ANZA2FieldOutput, displacement: tuple[float, float] | torch.Tensor) -> torch.Tensor:
    """Zadeh fuzzy OR over local modes for a directed displacement."""

    return (field.membership * directed_geometry(field, displacement)).amax(dim=1)
