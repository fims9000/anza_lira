"""Binary-mask to one-pixel skeleton conversion."""

from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D binary mask, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Cannot skeletonize a mask containing NaN or Inf")
    return skeletonize(array > 0).astype(bool, copy=False)
