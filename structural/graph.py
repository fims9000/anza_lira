"""Local graph validation and domain restriction helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def restrict_relation_to_domain(
    relation: np.ndarray,
    domain: np.ndarray,
    offsets: Iterable[tuple[int, int]],
) -> np.ndarray:
    edges = np.asarray(relation, dtype=np.float32)
    mask = np.asarray(domain, dtype=bool)
    offset_list = tuple((int(dx), int(dy)) for dx, dy in offsets)
    if edges.ndim != 3 or edges.shape[0] != len(offset_list) or edges.shape[1:] != mask.shape:
        raise ValueError("relation/domain/offset shapes are inconsistent")
    restricted = np.zeros_like(edges)
    height, width = mask.shape
    for channel, (dx, dy) in enumerate(offset_list):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        if y0 >= y1 or x0 >= x1:
            continue
        destination = mask[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        allowed = mask[y0:y1, x0:x1] & destination
        restricted[channel, y0:y1, x0:x1] = edges[channel, y0:y1, x0:x1] * allowed
    return restricted
