"""Frozen second-order/static measurement family used for causal matching."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..constants import FEATURE_WIDTH


STATIC_SIGNATURE_VERSION = "ANZA_KS_STATIC_SIGNATURE_V1"


def _coordinates(size: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, size)
    return np.meshgrid(axis, axis, indexing="xy")


def _mean_zero_unit(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float64) - np.mean(row)
    norm = np.linalg.norm(row)
    return row / norm if norm > 1e-12 else row


@lru_cache(maxsize=4)
def measurement_rows(size: int = 17) -> np.ndarray:
    x, y = _coordinates(size)
    rows = [np.ones((size, size)), x, y, x * x, y * y, x * y]
    angles = np.arange(8) * np.pi / 8.0
    for angle in angles:
        longitudinal = x * np.cos(angle) + y * np.sin(angle)
        transverse = -x * np.sin(angle) + y * np.cos(angle)
        rows.append(_mean_zero_unit(np.exp(-transverse**2 / (2 * 0.10**2)) * np.cos(np.pi * longitudinal)))
    for scale_u, scale_s in ((0.24, 0.08), (0.38, 0.12), (0.54, 0.18)):
        for angle in angles:
            longitudinal = x * np.cos(angle) + y * np.sin(angle)
            transverse = -x * np.sin(angle) + y * np.cos(angle)
            kernel = np.exp(-0.5 * ((longitudinal / scale_u) ** 2 + (transverse / scale_s) ** 2))
            rows.append(_mean_zero_unit(kernel))
    radius2 = x * x + y * y
    for scale in (0.12, 0.20, 0.30):
        log_probe = (radius2 / scale**4 - 2.0 / scale**2) * np.exp(-radius2 / (2 * scale**2))
        rows.append(_mean_zero_unit(log_probe))
    return np.stack([row.ravel() / (np.linalg.norm(row.ravel()) + 1e-12) for row in rows])


@lru_cache(maxsize=4)
def projection_basis(size: int = 17) -> np.ndarray:
    _, singular, vh = np.linalg.svd(measurement_rows(size), full_matrices=False)
    rank = int(np.sum(singular > 1e-10))
    return vh[:rank]


def static_signature(patch: np.ndarray) -> np.ndarray:
    values = np.asarray(patch, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("static signature requires a square patch")
    raw = measurement_rows(values.shape[0]) @ values.ravel()
    energy = float(np.sum(values * values))
    mean = float(values.mean())
    variance = float(values.var())
    derived = np.concatenate((raw, [mean, variance, energy]))
    if len(derived) > FEATURE_WIDTH:
        raise ValueError("static signature exceeds capacity-matched width")
    return np.pad(derived, (0, FEATURE_WIDTH - len(derived)))


def match_in_static_nullspace(positive: np.ndarray, negative: np.ndarray, *, residual_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Match every frozen static measurement and total energy without dynamic features."""

    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if positive.shape != negative.shape or positive.ndim != 2 or positive.shape[0] != positive.shape[1]:
        raise ValueError("candidate patches must be aligned squares")
    basis = projection_basis(positive.shape[0])
    p_flat = positive.ravel()
    n_flat = negative.ravel()
    common = basis.T @ (0.5 * (basis @ p_flat + basis @ n_flat))
    residual_p = p_flat - basis.T @ (basis @ p_flat)
    residual_n = n_flat - basis.T @ (basis @ n_flat)
    norm_p = np.linalg.norm(residual_p)
    norm_n = np.linalg.norm(residual_n)
    if min(norm_p, norm_n) <= 1e-8:
        raise ValueError("candidate lacks a static-nullspace component")
    target = residual_scale * min(norm_p, norm_n)
    matched_p = common + target * residual_p / norm_p
    matched_n = common + target * residual_n / norm_n
    return matched_p.reshape(positive.shape), matched_n.reshape(negative.shape)
