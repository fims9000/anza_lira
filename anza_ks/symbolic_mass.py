"""Image-conditioned probability densities and symbolic word masses."""

from __future__ import annotations

import numpy as np


def image_density(patch: np.ndarray, *, tau: float = 0.5, epsilon: float = 1e-9) -> np.ndarray:
    values = np.asarray(patch, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("density patch must be a square scalar field")
    standardized = (values - np.median(values)) / (values.std() + epsilon)
    scaled = standardized / tau
    positive = np.logaddexp(0.0, scaled) + epsilon
    return positive / positive.sum()


def symbolic_probabilities(density: np.ndarray, word_ids: np.ndarray, word_count: int) -> np.ndarray:
    density = np.asarray(density, dtype=np.float64)
    ids = np.asarray(word_ids, dtype=np.int64)
    if density.shape != ids.shape:
        raise ValueError("density and word IDs must align")
    probabilities = np.bincount(ids.ravel(), weights=density.ravel(), minlength=int(word_count)).astype(np.float64)
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("symbolic probability mass is empty")
    return probabilities / total
