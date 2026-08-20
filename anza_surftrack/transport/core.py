"""Frames, transitions, covariance propagation, and common mean motion."""

from __future__ import annotations

import numpy as np


def rotation(theta: np.ndarray | float) -> np.ndarray:
    value = np.asarray(theta, dtype=np.float64); c = np.cos(value); s = np.sin(value)
    return np.stack([np.stack([c, -s], axis=-1), np.stack([s, c], axis=-1)], axis=-2)


def transition_matrix(method: str, theta0: np.ndarray, theta1: np.ndarray, params: dict[str, float]) -> np.ndarray:
    r1 = rotation(theta1); r0t = np.swapaxes(rotation(theta0), -1, -2)
    shape = np.broadcast_shapes(np.shape(theta0), np.shape(theta1))
    local = np.zeros(shape + (2, 2), dtype=np.float64)
    if method == "G2_shear_compose":
        local[..., 0, 0] = 1.0; local[..., 1, 1] = 1.0; local[..., 0, 1] = params["alpha"]
    elif method == "G3_free_compose":
        local[..., 0, 0] = np.exp(params["a"]); local[..., 1, 1] = np.exp(params["b"])
    elif method == "G4_anza_cocycle":
        local[..., 0, 0] = np.exp(params["lambda"]); local[..., 1, 1] = np.exp(-params["lambda"])
    else:
        raise ValueError(f"no composed transition for {method}")
    return r1 @ local @ r0t


def initial_covariance(method: str, theta: np.ndarray, params: dict[str, float]) -> np.ndarray:
    if method == "G1_local_reset":
        return reset_covariance(theta, params)
    sigma = float(params.get("sigma0", 1.0))
    identity = np.eye(2, dtype=np.float64)
    return np.broadcast_to(identity * sigma * sigma, np.shape(theta) + (2, 2)).copy()


def reset_covariance(theta: np.ndarray, params: dict[str, float]) -> np.ndarray:
    r = rotation(theta); local = np.zeros(np.shape(theta) + (2, 2), dtype=np.float64)
    local[..., 0, 0] = params["sigma_u"] ** 2; local[..., 1, 1] = params["sigma_s"] ** 2
    return r @ local @ np.swapaxes(r, -1, -2)


def propagate_covariance(
    method: str, covariance: np.ndarray, theta0: np.ndarray, theta1: np.ndarray, params: dict[str, float],
) -> np.ndarray:
    if method == "G1_local_reset":
        return reset_covariance(theta1, params)
    j = transition_matrix(method, theta0, theta1, params)
    result = j @ covariance @ np.swapaxes(j, -1, -2)
    result = result + float(params["q"]) * np.eye(2)
    return 0.5 * (result + np.swapaxes(result, -1, -2))


def common_mean(last: np.ndarray, previous: np.ndarray | None, elapsed: np.ndarray | float = 1.0) -> np.ndarray:
    if previous is None:
        return np.asarray(last, dtype=np.float64).copy()
    return np.asarray(last) + np.asarray(elapsed)[..., None] * (np.asarray(last) - np.asarray(previous))


def gaussian_log_score(delta: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    sign, logdet = np.linalg.slogdet(covariance)
    if np.any(sign <= 0):
        raise ValueError("non-positive covariance in likelihood")
    solved = np.linalg.solve(covariance, delta[..., None])[..., 0]
    return -0.5 * np.sum(delta * solved, axis=-1) - 0.5 * logdet
