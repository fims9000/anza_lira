"""Fixed axial charts shared by all K1 methods."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import rotate

from .torus import CAT_MAP


ORIENTATION_COUNT = 8


def orientation_angles(count: int = ORIENTATION_COUNT) -> np.ndarray:
    return np.arange(count, dtype=np.float64) * np.pi / float(count)


def unstable_angle() -> float:
    values, vectors = np.linalg.eig(CAT_MAP.astype(np.float64))
    vector = vectors[:, int(np.argmax(np.abs(values)))].real
    return float(np.arctan2(vector[1], vector[0]) % np.pi)


def align_patch(patch: np.ndarray, orientation: float) -> np.ndarray:
    """Rotate an axial chart so ``orientation`` aligns with Cat's unstable axis."""

    delta = unstable_angle() - (float(orientation) % np.pi)
    return rotate(np.asarray(patch, dtype=np.float64), np.degrees(delta), reshape=False, order=1, mode="wrap", prefilter=False)
