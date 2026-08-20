"""Exact fuzzy max-min transitive closure on a local relation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from models.azconv_affinity import LOCAL8_OFFSETS, _shift_tensor


def _validate_numpy(seed: np.ndarray, relation: np.ndarray, offsets: tuple[tuple[int, int], ...]) -> None:
    if seed.ndim != 2 or relation.shape != (len(offsets),) + seed.shape:
        raise ValueError("seed must be HxW and relation must be KxHxW")
    if not np.isfinite(seed).all() or not np.isfinite(relation).all():
        raise ValueError("max-min inputs must be finite")
    if np.any((seed < 0) | (seed > 1)) or np.any((relation < 0) | (relation > 1)):
        raise ValueError("max-min inputs must lie in [0, 1]")


def maxmin_closure_reference(
    seed: np.ndarray,
    relation: np.ndarray,
    *,
    offsets: Iterable[tuple[int, int]] = LOCAL8_OFFSETS,
    max_steps: int | None = None,
) -> tuple[np.ndarray, int]:
    """Readable scalar reference for ``max(s0, max_q min(s(q), a(q,p)))``."""

    offset_list = tuple((int(dx), int(dy)) for dx, dy in offsets)
    support0 = np.asarray(seed, dtype=np.float32)
    edges = np.asarray(relation, dtype=np.float32)
    _validate_numpy(support0, edges, offset_list)
    height, width = support0.shape
    limit = int(max_steps if max_steps is not None else height + width)
    state = support0.copy()
    for step in range(1, limit + 1):
        updated = support0.copy()
        for y in range(height):
            for x in range(width):
                value = float(updated[y, x])
                for channel, (dx, dy) in enumerate(offset_list):
                    qx, qy = x + dx, y + dy
                    if 0 <= qx < width and 0 <= qy < height:
                        value = max(value, min(float(state[qy, qx]), float(edges[channel, y, x])))
                updated[y, x] = value
        if np.array_equal(updated, state):
            return updated, step
        state = updated
    return state, limit


def maxmin_closure_torch(
    seed: torch.Tensor,
    relation: torch.Tensor,
    *,
    offsets: Iterable[tuple[int, int]] = LOCAL8_OFFSETS,
    max_steps: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Batched PyTorch implementation exactly matching the scalar reference."""

    offset_list = tuple((int(dx), int(dy)) for dx, dy in offsets)
    if seed.ndim == 3:
        seed = seed.unsqueeze(1)
    if seed.ndim != 4 or seed.shape[1] != 1:
        raise ValueError("seed must be B,H,W or B,1,H,W")
    if relation.shape != (seed.shape[0], len(offset_list), seed.shape[2], seed.shape[3]):
        raise ValueError("relation must be B,K,H,W")
    if not torch.isfinite(seed).all() or not torch.isfinite(relation).all():
        raise ValueError("max-min inputs must be finite")
    if torch.any((seed < 0) | (seed > 1)) or torch.any((relation < 0) | (relation > 1)):
        raise ValueError("max-min inputs must lie in [0, 1]")
    support0 = seed
    state = seed.clone()
    limit = int(max_steps if max_steps is not None else seed.shape[2] + seed.shape[3])
    for step in range(1, limit + 1):
        messages = [support0]
        for channel, (dx, dy) in enumerate(offset_list):
            neighbor, valid = _shift_tensor(state, dx, dy)
            messages.append(torch.minimum(neighbor, relation[:, channel : channel + 1]) * valid)
        updated = torch.stack(messages, dim=0).amax(dim=0)
        if torch.equal(updated, state):
            return updated, step
        state = updated
    return state, limit

