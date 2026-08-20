"""Absolute bidirectional ANZA-2 structural relation."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from .field import ANZA2FieldOutput
from .geometry import directed_step_support


LOCAL8_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
)


def _shift_neighbor(tensor: torch.Tensor, dx: int, dy: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return values at q=p+(dx,dy) and a non-wrapping validity mask."""

    shifted = torch.roll(tensor, shifts=(-int(dy), -int(dx)), dims=(-2, -1))
    valid = torch.ones(tensor.shape[-2:], dtype=torch.bool, device=tensor.device)
    if dy > 0:
        valid[-dy:, :] = False
    elif dy < 0:
        valid[:-dy, :] = False
    if dx > 0:
        valid[:, -dx:] = False
    elif dx < 0:
        valid[:, :-dx] = False
    while valid.ndim < tensor.ndim:
        valid = valid.unsqueeze(0)
    return shifted, valid


def shift_field(field: ANZA2FieldOutput, dx: int, dy: int) -> tuple[ANZA2FieldOutput, torch.Tensor]:
    membership, valid = _shift_neighbor(field.membership, dx, dy)
    orientation, _ = _shift_neighbor(field.orientation, dx, dy)
    base_scale, _ = _shift_neighbor(field.base_scale, dx, dy)
    hyperbolicity, _ = _shift_neighbor(field.hyperbolicity, dx, dy)
    sigma_parallel, _ = _shift_neighbor(field.sigma_parallel, dx, dy)
    sigma_perpendicular, _ = _shift_neighbor(field.sigma_perpendicular, dx, dy)
    return ANZA2FieldOutput(
        membership=membership,
        orientation=orientation,
        base_scale=base_scale,
        hyperbolicity=hyperbolicity,
        sigma_parallel=sigma_parallel,
        sigma_perpendicular=sigma_perpendicular,
    ), valid


def structural_affinity_pair(
    field_p: ANZA2FieldOutput,
    field_q: ANZA2FieldOutput,
    displacement: tuple[float, float],
) -> torch.Tensor:
    """Symmetric geometric mean of two independently supported directed steps."""

    forward = directed_step_support(field_p, displacement)
    reverse = directed_step_support(field_q, (-float(displacement[0]), -float(displacement[1])))
    return torch.sqrt((forward * reverse).clamp_min(0.0)).clamp(0.0, 1.0)


class ANZA2StructuralAffinity(nn.Module):
    """Build local absolute edge strengths without convolution normalization."""

    def __init__(self, offsets: Iterable[tuple[int, int]] = LOCAL8_OFFSETS) -> None:
        super().__init__()
        self.offsets = tuple((int(dx), int(dy)) for dx, dy in offsets)
        if not self.offsets or (0, 0) in self.offsets:
            raise ValueError("affinity offsets must be non-empty and exclude self")

    def forward(self, field: ANZA2FieldOutput) -> torch.Tensor:
        edges = []
        for dx, dy in self.offsets:
            neighbor, valid = shift_field(field, dx, dy)
            edge = structural_affinity_pair(field, neighbor, (dx, dy))
            edges.append(edge * valid.reshape(1, *valid.shape[-2:]).to(edge.dtype))
        return torch.stack(edges, dim=1)


class GenericAffinityCombiner(nn.Module):
    """Causal generic-logit + nonnegative ANZA-prior combination."""

    def __init__(self, *, initial_beta: float = 0.0, eps: float = 1e-6) -> None:
        super().__init__()
        if initial_beta < 0:
            raise ValueError("initial_beta must be nonnegative")
        self.eps = float(eps)
        # A very negative raw value makes the prescribed softplus beta
        # numerically zero while retaining the exact generic baseline to test tolerance.
        raw = -12.0 if initial_beta == 0.0 else math.log(math.expm1(initial_beta))
        self.beta_raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.beta_raw)

    def forward(
        self,
        generic_logits: torch.Tensor,
        anza_affinity: torch.Tensor,
        *,
        beta_override: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if generic_logits.shape != anza_affinity.shape:
            raise ValueError("generic logits and ANZA affinity must have equal shape")
        prior = torch.logit(anza_affinity.clamp(self.eps, 1.0 - self.eps))
        beta = self.beta if beta_override is None else torch.as_tensor(
            beta_override, dtype=generic_logits.dtype, device=generic_logits.device
        )
        if bool((beta < 0).any()):
            raise ValueError("beta must be nonnegative")
        return generic_logits + beta.to(generic_logits.dtype) * prior
