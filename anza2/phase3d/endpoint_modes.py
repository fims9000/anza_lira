"""Endpoint-to-mode compatibility for oracle and learned continuation."""

from __future__ import annotations

import numpy as np

from models.anza2.field import ANZA2FieldOutput


def compatible_endpoint_modes(
    field: ANZA2FieldOutput,
    point_yx: tuple[int, int],
    tangent_doubled: tuple[float, float] | np.ndarray,
    *,
    membership_threshold: float = 0.5,
    axial_similarity_threshold: float = 0.90,
) -> tuple[int, ...]:
    if field.membership.shape[0] != 1:
        raise ValueError("endpoint mode selection expects one field")
    y, x = map(int, point_yx)
    tangent = np.asarray(tangent_doubled, dtype=np.float64)
    if tangent.shape != (2,):
        raise ValueError("tangent_doubled must contain cos(2theta), sin(2theta)")
    norm = np.linalg.norm(tangent)
    if norm <= 0:
        raise ValueError("endpoint tangent must be nonzero")
    tangent = tangent / norm
    membership = field.membership[0, :, y, x].detach().cpu().numpy()
    orientation = field.orientation[0, :, :, y, x].detach().cpu().numpy()
    similarity = orientation @ tangent
    return tuple(int(index) for index in np.flatnonzero(
        (membership >= membership_threshold) & (similarity >= axial_similarity_threshold)
    ))

