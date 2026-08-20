"""Generic two-sided Gaussian fusion used only for mathematical S0 fixtures."""

from __future__ import annotations

import numpy as np


def precision_fusion(
    mean_forward: np.ndarray, covariance_forward: np.ndarray,
    mean_backward: np.ndarray, covariance_backward: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    precision_forward = np.linalg.inv(covariance_forward); precision_backward = np.linalg.inv(covariance_backward)
    covariance = np.linalg.inv(precision_forward + precision_backward)
    mean = covariance @ (
        precision_forward @ np.asarray(mean_forward)[..., None]
        + precision_backward @ np.asarray(mean_backward)[..., None]
    )
    return mean[..., 0], covariance


def anchor_disagreement(
    mean_forward: np.ndarray, covariance_forward: np.ndarray,
    mean_backward: np.ndarray, covariance_backward: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(mean_forward) - np.asarray(mean_backward)
    covariance = covariance_forward + covariance_backward
    return np.sum(delta * np.linalg.solve(covariance, delta[..., None])[..., 0], axis=-1)
