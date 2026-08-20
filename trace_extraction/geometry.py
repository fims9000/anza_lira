"""Axial model-native orientation and anisotropy mathematics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class AxialGeometry:
    orientation: np.ndarray
    coherence: np.ndarray
    anisotropy: np.ndarray

    @property
    def rho(self) -> np.ndarray:
        """Conventional axial resultant-length alias for coherence."""
        return self.coherence


def axial_distance(theta_a: np.ndarray | float, theta_b: np.ndarray | float) -> np.ndarray:
    delta = np.asarray(theta_a, dtype=np.float64) - np.asarray(theta_b, dtype=np.float64)
    return 0.5 * np.arccos(np.clip(np.cos(2.0 * delta), -1.0, 1.0))


def combine_axial_geometry(
    memberships: np.ndarray,
    theta: np.ndarray,
    sigma_u: np.ndarray,
    sigma_s: np.ndarray,
    *,
    rule_axis: int = 0,
    eps: float = 1e-8,
) -> AxialGeometry:
    mu = np.asarray(memberships, dtype=np.float64)
    theta_array = np.asarray(theta, dtype=np.float64)
    sigma_u_array = np.asarray(sigma_u, dtype=np.float64)
    sigma_s_array = np.asarray(sigma_s, dtype=np.float64)
    if not (mu.shape == theta_array.shape == sigma_u_array.shape == sigma_s_array.shape):
        raise ValueError("mu, theta, sigma_u, and sigma_s must have the same shape")
    if not all(np.isfinite(array).all() for array in (mu, theta_array, sigma_u_array, sigma_s_array)):
        raise ValueError("Axial geometry inputs must be finite")
    if np.any(mu < 0) or np.any(sigma_u_array <= 0) or np.any(sigma_s_array <= 0):
        raise ValueError("Memberships must be nonnegative and sigma values positive")
    weight = np.sum(mu, axis=rule_axis)
    c_value = np.sum(mu * np.cos(2.0 * theta_array), axis=rule_axis)
    s_value = np.sum(mu * np.sin(2.0 * theta_array), axis=rule_axis)
    orientation = 0.5 * np.arctan2(s_value, c_value)
    coherence = np.sqrt(np.square(c_value) + np.square(s_value)) / (weight + eps)
    per_rule_anisotropy = np.tanh(np.abs(np.log(sigma_u_array / sigma_s_array)))
    anisotropy = coherence * np.sum(mu * per_rule_anisotropy, axis=rule_axis) / (weight + eps)
    return AxialGeometry(
        orientation=orientation,
        coherence=np.clip(coherence, 0.0, 1.0),
        anisotropy=np.clip(anisotropy, 0.0, 1.0),
    )


def geometry_from_interpretation(snapshot: dict, *, rule_axis: int = 0) -> AxialGeometry:
    required = ("mu_map", "theta_map", "sigma_u_map", "sigma_s_map")
    missing = [key for key in required if key not in snapshot]
    if missing:
        raise ValueError(f"AZ interpretation snapshot lacks {missing}")

    def as_numpy(value: object) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    return combine_axial_geometry(*(as_numpy(snapshot[key]) for key in required), rule_axis=rule_axis)


def edge_geometry_confidence(
    p_yx: tuple[int, int],
    q_yx: tuple[int, int],
    probability: np.ndarray,
    orientation: np.ndarray,
    coherence: np.ndarray,
    anisotropy: np.ndarray,
) -> float:
    arrays = tuple(np.asarray(value) for value in (probability, orientation, coherence, anisotropy))
    if any(array.shape != arrays[0].shape for array in arrays) or arrays[0].ndim != 2:
        raise ValueError("Edge geometry maps must have the same 2-D shape")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Edge geometry maps must be finite")
    py, px = p_yx
    qy, qx = q_yx
    phi = math.atan2(qy - py, qx - px)
    g_p = 0.5 * (1.0 + math.cos(2.0 * (float(orientation[py, px]) - phi)))
    g_q = 0.5 * (1.0 + math.cos(2.0 * (float(orientation[qy, qx]) - phi)))
    score = (
        math.sqrt(max(float(probability[py, px] * probability[qy, qx]), 0.0))
        * math.sqrt(max(float(coherence[py, px] * coherence[qy, qx]), 0.0))
        * math.sqrt(max(float(anisotropy[py, px] * anisotropy[qy, qx]), 0.0))
        * g_p
        * g_q
    )
    return float(np.clip(score, 0.0, 1.0))


def local_pca_orientation(skeleton: np.ndarray, *, radius: int = 5) -> np.ndarray:
    skeleton = np.asarray(skeleton, dtype=bool)
    if skeleton.ndim != 2:
        raise ValueError(f"Expected 2-D skeleton, got {skeleton.shape}")
    output = np.zeros(skeleton.shape, dtype=np.float64)
    height, width = skeleton.shape
    for y, x in np.argwhere(skeleton):
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        points = np.argwhere(skeleton[y0:y1, x0:x1])
        if len(points) < 2:
            continue
        xy = np.column_stack((points[:, 1] + x0, points[:, 0] + y0)).astype(np.float64)
        centered = xy - xy.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(len(xy) - 1, 1)
        values, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, int(np.argmax(values))]
        output[y, x] = math.atan2(float(direction[1]), float(direction[0]))
    return output
