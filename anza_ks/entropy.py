"""Finite-partition entropy features; no exact image KS-entropy claim."""

from __future__ import annotations

import numpy as np


def block_entropy(probabilities: np.ndarray, *, epsilon: float = 1e-12) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positive = probabilities[probabilities > 0]
    if not np.isfinite(probabilities).all() or np.any(probabilities < -epsilon):
        raise ValueError("invalid probability vector")
    return float(max(-np.sum(positive * np.log(positive + epsilon)), 0.0))


def conditional_entropies(block_entropies: list[float]) -> np.ndarray:
    values = np.asarray(block_entropies, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("at least two block entropies are required")
    return np.diff(values)
