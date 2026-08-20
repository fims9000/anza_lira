"""Local area-preserving hyperbolic frame transport."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _vector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if value.shape != (2,) or not np.isfinite(value).all() or norm <= 1e-12:
        raise ValueError("a finite nonzero 2-vector is required")
    return value


def _unit(vector: np.ndarray) -> np.ndarray:
    value = _vector(vector)
    norm = float(np.linalg.norm(value))
    return value / norm


def rotation(theta: float) -> np.ndarray:
    cosine, sine = math.cos(theta), math.sin(theta)
    return np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)


@dataclass(frozen=True)
class HyperbolicFrame:
    """One local SL(2,R)-like cocycle element with curved outgoing frame."""

    theta: float
    curvature: float
    hyperbolicity: float = 0.35
    step: float = 1.0

    def __post_init__(self) -> None:
        values = (self.theta, self.curvature, self.hyperbolicity, self.step)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("frame parameters must be finite")
        if self.hyperbolicity < 0 or self.step <= 0:
            raise ValueError("hyperbolicity must be nonnegative and step positive")

    @property
    def tangent(self) -> np.ndarray:
        return np.asarray((math.cos(self.theta), math.sin(self.theta)), dtype=np.float64)

    @property
    def normal(self) -> np.ndarray:
        return np.asarray((-math.sin(self.theta), math.cos(self.theta)), dtype=np.float64)

    @property
    def outgoing_theta(self) -> float:
        return self.theta + self.curvature * self.step

    @property
    def outgoing_tangent(self) -> np.ndarray:
        return np.asarray((math.cos(self.outgoing_theta), math.sin(self.outgoing_theta)), dtype=np.float64)

    @property
    def outgoing_normal(self) -> np.ndarray:
        return np.asarray((-math.sin(self.outgoing_theta), math.cos(self.outgoing_theta)), dtype=np.float64)

    def matrix(self) -> np.ndarray:
        diagonal = np.diag((math.exp(self.hyperbolicity), math.exp(-self.hyperbolicity)))
        return rotation(self.outgoing_theta) @ diagonal @ rotation(-self.theta)

    def inverse_matrix(self) -> np.ndarray:
        return rotation(self.theta) @ np.diag(
            (math.exp(-self.hyperbolicity), math.exp(self.hyperbolicity))
        ) @ rotation(-self.outgoing_theta)

    def transport(self, vector: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        result = self.matrix() @ _vector(vector)
        return _unit(result) if normalize else result

    def inverse_transport(self, vector: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        result = self.inverse_matrix() @ _vector(vector)
        return _unit(result) if normalize else result


def axial_compatibility(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.dot(_unit(first), _unit(second)) ** 2)


def match_transported_frame(
    transported: np.ndarray,
    local_axes: np.ndarray,
    memberships: np.ndarray,
    *,
    temperature: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Permutation-invariant soft match of an axial frame to local modes."""

    incoming = _unit(transported)
    axes = np.asarray(local_axes, dtype=np.float64)
    mu = np.asarray(memberships, dtype=np.float64)
    if axes.ndim != 2 or axes.shape[1] != 2 or mu.shape != (axes.shape[0],):
        raise ValueError("local axes must be Rx2 and memberships R")
    if temperature <= 0 or not np.isfinite(axes).all() or not np.isfinite(mu).all():
        raise ValueError("finite inputs and positive temperature are required")
    normalized = np.stack([_unit(axis) for axis in axes])
    logits = np.log(np.clip(mu, 1e-8, 1.0)) + (normalized @ incoming) ** 2 / temperature
    logits -= logits.max()
    weights = np.exp(logits); weights /= weights.sum()
    signs = np.where(normalized @ incoming >= 0, 1.0, -1.0)
    matched = _unit(np.sum(weights[:, None] * normalized * signs[:, None], axis=0))
    return matched, weights
