"""ANZA-2 mode-preserving spatial state graph."""

from __future__ import annotations

from typing import Iterable

import torch

from models.anza2.affinity import LOCAL8_OFFSETS, shift_field
from models.anza2.field import ANZA2FieldOutput
from models.anza2.geometry import directed_geometry


def mode_state_edge_weights(
    field: ANZA2FieldOutput,
    offsets: Iterable[tuple[int, int]] = LOCAL8_OFFSETS,
) -> torch.Tensor:
    """Return BxCxRxSxHxW spatial edges for nodes ``(pixel, mode)``.

    Offset zero is forbidden, so this tensor cannot encode a free intra-pixel
    transition between modes.
    """

    offset_list = tuple((int(dx), int(dy)) for dx, dy in offsets)
    if not offset_list or (0, 0) in offset_list:
        raise ValueError("mode-state offsets must be non-empty and exclude intra-pixel switches")
    edges = []
    for dx, dy in offset_list:
        neighbor, valid = shift_field(field, dx, dy)
        source_geometry = directed_geometry(field, (dx, dy))
        destination_geometry = directed_geometry(neighbor, (-dx, -dy))
        product = (
            field.membership.unsqueeze(2)
            * neighbor.membership.unsqueeze(1)
            * source_geometry.unsqueeze(2)
            * destination_geometry.unsqueeze(1)
        )
        edge = torch.sqrt(product.clamp_min(0.0)).clamp(0.0, 1.0)
        edges.append(edge * valid.reshape(1, 1, 1, *valid.shape[-2:]).to(edge.dtype))
    return torch.stack(edges, dim=1)


def permute_modes(field: ANZA2FieldOutput, order: torch.Tensor) -> ANZA2FieldOutput:
    """Apply one global mode permutation for invariance tests/audits."""

    if order.ndim != 1 or len(order) != field.num_modes:
        raise ValueError("order must contain exactly one index per mode")
    return ANZA2FieldOutput(
        field.membership[:, order], field.orientation[:, order],
        field.base_scale[:, order], field.hyperbolicity[:, order],
        field.sigma_parallel[:, order], field.sigma_perpendicular[:, order],
    )

