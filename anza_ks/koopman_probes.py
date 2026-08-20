"""Fixed probes and exact-permutation correlations for K1 controls."""

from __future__ import annotations

import numpy as np

from .torus import CAT_MAP, exact_permutation


def _normalize(probe: np.ndarray) -> np.ndarray:
    probe = np.asarray(probe, dtype=np.float64) - np.mean(probe)
    norm = np.linalg.norm(probe)
    if norm <= 1e-12:
        raise ValueError("degenerate probe")
    return probe / norm


def fixed_probes(size: int = 17) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, size, endpoint=True)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    central = np.exp(-(x * x + y * y) / (2 * 0.20**2))
    unstable = np.exp(-(y**2) / (2 * 0.12**2))
    stable = np.exp(-(x**2) / (2 * 0.12**2))
    checker = np.sign(x) * np.sign(y)
    return np.stack([_normalize(value) for value in (central, unstable, stable, checker)])


def koopman_correlations(patch: np.ndarray, matrix: np.ndarray = CAT_MAP, *, K: int = 4) -> np.ndarray:
    values = np.asarray(patch, dtype=np.float64)
    values = (values - values.mean()) / (np.linalg.norm(values - values.mean()) + 1e-12)
    probes = fixed_probes(values.shape[0])
    correlations = []
    for lag in range(-K, K + 1):
        transported = exact_permutation(values, matrix, power=lag)
        correlations.extend(float(np.sum(probe * transported)) for probe in probes)
    return np.asarray(correlations, dtype=np.float64)
