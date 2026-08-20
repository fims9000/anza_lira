"""Crowd agreement weights that preserve T1 partial-label semantics."""

from __future__ import annotations

import numpy as np

from datasets.cracks import BLUE, GREEN, ORANGE, WHITE


def crowd_agreement(masks_rgb: list[np.ndarray]) -> dict[str, np.ndarray]:
    if not masks_rgb:
        raise ValueError("agreement requires at least one nonexpert annotation")
    shape = np.asarray(masks_rgb[0]).shape
    if len(shape) != 3 or shape[-1] != 3:
        raise ValueError("agreement annotations must be HxWx3")
    w_pos = np.zeros(shape[:2], dtype=np.float32)
    w_neg = np.zeros(shape[:2], dtype=np.float32)
    allowed = (BLUE, GREEN, ORANGE, WHITE)
    for mask in masks_rgb:
        rgb = np.asarray(mask, dtype=np.uint8)
        if rgb.shape != shape:
            raise ValueError("agreement annotations have inconsistent shapes")
        known = np.zeros(shape[:2], dtype=bool)
        for color in allowed:
            known |= np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)
        if not known.all():
            raise ValueError("agreement annotation contains an unknown color")
        w_pos += np.all(rgb == np.asarray(BLUE, dtype=np.uint8), axis=-1).astype(np.float32)
        w_pos += 0.5 * np.all(rgb == np.asarray(GREEN, dtype=np.uint8), axis=-1).astype(np.float32)
        w_neg += np.all(rgb == np.asarray(ORANGE, dtype=np.uint8), axis=-1).astype(np.float32)
    labeled = w_pos + w_neg
    probability = np.divide(w_pos, labeled, out=np.zeros_like(w_pos), where=labeled > 0)
    agreement = np.where(labeled > 0, np.abs(2.0 * probability - 1.0) ** 2 * np.minimum(1.0, labeled / 3.0), 0.0)
    return {
        "positive_weight": w_pos,
        "negative_weight": w_neg,
        "labeled_weight": labeled,
        "crowd_probability": probability.astype(np.float32),
        "agreement": agreement.astype(np.float32),
    }

