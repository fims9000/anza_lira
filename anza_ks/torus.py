"""Exact integer torus permutations used by ANZA-KS."""

from __future__ import annotations

import numpy as np


CAT_MAP = np.asarray([[2, 1], [1, 1]], dtype=np.int64)
CAT_INVERSE = np.asarray([[1, -1], [-1, 2]], dtype=np.int64)
SHEAR_MAP = np.asarray([[1, 1], [0, 1]], dtype=np.int64)
SHEAR_INVERSE = np.asarray([[1, -1], [0, 1]], dtype=np.int64)


def integer_matrix_power(matrix: np.ndarray, power: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (2, 2):
        raise ValueError("torus matrix must be 2x2")
    determinant = int(round(np.linalg.det(matrix)))
    if abs(determinant) != 1:
        raise ValueError("exact finite permutation requires a unimodular matrix")
    if power < 0:
        matrix = np.rint(np.linalg.inv(matrix)).astype(np.int64)
        power = -power
    return np.linalg.matrix_power(matrix, int(power)).astype(np.int64)


def permutation_indices(size: int, matrix: np.ndarray = CAT_MAP, *, power: int = 1) -> tuple[np.ndarray, np.ndarray]:
    if size < 3:
        raise ValueError("torus size must be >=3")
    transform = integer_matrix_power(matrix, power)
    y, x = np.meshgrid(np.arange(size, dtype=np.int64), np.arange(size, dtype=np.int64), indexing="ij")
    source_x = (transform[0, 0] * x + transform[0, 1] * y) % size
    source_y = (transform[1, 0] * x + transform[1, 1] * y) % size
    return source_y, source_x


def exact_permutation(field: np.ndarray, matrix: np.ndarray = CAT_MAP, *, power: int = 1) -> np.ndarray:
    """Return f(T^power z) as an exact finite-grid permutation.

    This is a finite permutation diagnostic and is not called an ergodic map.
    """

    values = np.asarray(field)
    if values.ndim < 2 or values.shape[-2] != values.shape[-1]:
        raise ValueError("field must end in equal square dimensions")
    source_y, source_x = permutation_indices(values.shape[-1], matrix, power=power)
    return values[..., source_y, source_x]
