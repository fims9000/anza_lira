"""Exact toral maps and their explicitly approximate grid readout."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


CAT_MAP = np.asarray([[2, 1], [1, 1]], dtype=np.int64)
CAT_INVERSE = np.asarray([[1, -1], [-1, 2]], dtype=np.int64)
SHEAR_MAP = np.asarray([[1, 1], [0, 1]], dtype=np.int64)
SHEAR_INVERSE = np.asarray([[1, -1], [0, 1]], dtype=np.int64)


def torus_grid(size: int, *, centered: bool = True) -> np.ndarray:
    if size < 3:
        raise ValueError("torus grid requires size >= 3")
    axis = np.arange(size, dtype=np.float64) / float(size)
    if centered:
        axis -= 0.5
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((xx, yy), axis=-1)


def wrap_centered(points: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) + 0.5) % 1.0 - 0.5


def integer_matrix_power(matrix: np.ndarray, power: int) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.int64)
    if value.shape != (2, 2):
        raise ValueError("toral matrix must be 2x2")
    if power < 0:
        determinant = int(round(np.linalg.det(value)))
        if abs(determinant) != 1:
            raise ValueError("negative powers require an integer unimodular matrix")
        value = np.rint(np.linalg.inv(value)).astype(np.int64)
        power = -power
    return np.linalg.matrix_power(value, int(power)).astype(np.int64)


def torus_map(points: np.ndarray, matrix: np.ndarray = CAT_MAP, *, power: int = 1) -> np.ndarray:
    transform = integer_matrix_power(matrix, power).astype(np.float64)
    return wrap_centered(np.asarray(points, dtype=np.float64) @ transform.T)


def bilinear_torus_sample(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("observable must be a square scalar grid")
    size = values.shape[0]
    wrapped = wrap_centered(points)
    x_index = (wrapped[..., 0] + 0.5) * size
    y_index = (wrapped[..., 1] + 0.5) * size
    return map_coordinates(values, np.stack((y_index, x_index)), order=1, mode="grid-wrap", prefilter=False)


def koopman_transport(field: np.ndarray, matrix: np.ndarray = CAT_MAP, *, power: int = 1) -> np.ndarray:
    """Approximate U^power f(z)=f(T^power z) using periodic bilinear sampling."""
    grid = torus_grid(np.asarray(field).shape[0])
    return bilinear_torus_sample(field, torus_map(grid, matrix, power=power))


def exact_discrete_permutation(field: np.ndarray, matrix: np.ndarray = CAT_MAP, *, power: int = 1) -> np.ndarray:
    """Exact finite-grid permutation diagnostic; no finite-grid ergodicity claim."""
    values = np.asarray(field)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("field must be square")
    size = values.shape[0]
    transform = integer_matrix_power(matrix, power)
    jj, ii = np.meshgrid(np.arange(size, dtype=np.int64), np.arange(size, dtype=np.int64), indexing="xy")
    source_x = (transform[0, 0] * jj + transform[0, 1] * ii) % size
    source_y = (transform[1, 0] * jj + transform[1, 1] * ii) % size
    return values[source_y, source_x]


def cat_eigensystem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eig(CAT_MAP.astype(np.float64))
    unstable_index = int(np.argmax(np.abs(eigenvalues)))
    stable_index = 1 - unstable_index
    unstable = eigenvectors[:, unstable_index].real
    stable = eigenvectors[:, stable_index].real
    unstable /= np.linalg.norm(unstable)
    stable /= np.linalg.norm(stable)
    return eigenvalues.real, unstable, stable, eigenvectors.real
