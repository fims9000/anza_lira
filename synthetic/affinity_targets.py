"""Lineage-derived local edge targets with explicit hard negatives."""

from __future__ import annotations

from typing import Iterable, Mapping, Any

import numpy as np


def _shift(array: np.ndarray, dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(array)
    shifted = np.zeros_like(source)
    valid = np.zeros(source.shape[-2:], dtype=bool)
    height, width = source.shape[-2:]
    src_y0, src_y1 = max(0, dy), min(height, height + dy)
    src_x0, src_x1 = max(0, dx), min(width, width + dx)
    dst_y0, dst_y1 = max(0, -dy), min(height, height - dy)
    dst_x0, dst_x1 = max(0, -dx), min(width, width - dx)
    shifted[..., dst_y0:dst_y1, dst_x0:dst_x1] = source[..., src_y0:src_y1, src_x0:src_x1]
    valid[dst_y0:dst_y1, dst_x0:dst_x1] = True
    return shifted, valid


def build_affinity_targets(
    sample: Mapping[str, Any],
    offsets: Iterable[tuple[int, int]],
) -> dict[str, np.ndarray]:
    offset_list = tuple((int(dx), int(dy)) for dx, dy in offsets)
    instances = np.asarray(sample["instance_masks"], dtype=bool)
    latent = np.asarray(sample["latent_fault_mask"], dtype=bool)
    negative_gap = np.asarray(sample["negative_gap_mask"], dtype=bool)
    candidate = latent | negative_gap
    positives: list[np.ndarray] = []
    hard_negatives: list[np.ndarray] = []
    valid_edges: list[np.ndarray] = []
    for dx, dy in offset_list:
        shifted_instances, in_bounds = _shift(instances, dx, dy)
        shifted_candidate, _ = _shift(candidate, dx, dy)
        shifted_latent, _ = _shift(latent, dx, dy)
        shifted_negative_gap, _ = _shift(negative_gap, dx, dy)
        same_identity = np.any(instances & shifted_instances, axis=0) & in_bounds
        both_candidates = candidate & shifted_candidate & in_bounds
        negative_control = (
            (latent & shifted_latent)
            | negative_gap
            | shifted_negative_gap
        )
        hard_negative = both_candidates & ~same_identity & negative_control
        positives.append(same_identity)
        hard_negatives.append(hard_negative)
        valid_edges.append(same_identity | hard_negative)
    positive = np.stack(positives)
    hard = np.stack(hard_negatives)
    valid = np.stack(valid_edges)
    return {
        "affinity_target": positive.astype(np.float32),
        "affinity_positive": positive,
        "affinity_hard_negative": hard,
        "affinity_valid": valid,
        "offsets_xy": np.asarray(offset_list, dtype=np.int16),
    }
