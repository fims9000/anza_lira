"""Finite-time cocycle products and Cauchy--Green diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np


def cocycle_product(matrices: Iterable[np.ndarray]) -> np.ndarray:
    product = np.eye(2, dtype=np.float64)
    for matrix in matrices:
        value = np.asarray(matrix, dtype=np.float64)
        if value.shape != (2, 2) or not np.isfinite(value).all():
            raise ValueError("finite 2x2 cocycle matrices required")
        product = value @ product
    return product


def cauchy_green(product: np.ndarray) -> np.ndarray:
    value = np.asarray(product, dtype=np.float64)
    if value.shape != (2, 2) or not np.isfinite(value).all():
        raise ValueError("finite 2x2 product required")
    tensor = value.T @ value
    eigenvalues = np.linalg.eigvalsh(tensor)
    if eigenvalues[0] <= 0:
        raise ValueError("Cauchy--Green tensor must be positive definite")
    return tensor


def finite_time_diagnostics(product: np.ndarray, elapsed: float) -> dict[str, float]:
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("elapsed time must be positive")
    eigenvalues = np.linalg.eigvalsh(cauchy_green(product))
    singular = np.sqrt(eigenvalues)
    return {
        "sigma_min": float(singular[0]),
        "sigma_max": float(singular[-1]),
        "hyperbolicity_ratio": float(singular[-1] / singular[0]),
        "ftle_min": float(math.log(singular[0]) / elapsed),
        "ftle_max": float(math.log(singular[-1]) / elapsed),
        "determinant": float(np.linalg.det(product)),
    }
