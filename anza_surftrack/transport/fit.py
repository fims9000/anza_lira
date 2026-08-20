"""Train-only bounded maximum-likelihood fitting for G1--G4."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from ..protocol import PROTOCOL
from ..synthetic3d.families import CaseBatch, generate_batch
from .core import initial_covariance, propagate_covariance, common_mean, gaussian_log_score


PARAMETERS = {
    "G1_local_reset": ("sigma_u", "sigma_s"),
    "G2_shear_compose": ("sigma0", "q", "alpha"),
    "G3_free_compose": ("sigma0", "q", "a", "b"),
    "G4_anza_cocycle": ("sigma0", "q", "lambda"),
}


def transition_residuals(batch: CaseBatch) -> tuple[np.ndarray, np.ndarray]:
    n, steps, _ = batch.true_points.shape
    residual = np.zeros_like(batch.true_points); elapsed = np.ones((n, steps), dtype=np.int16)
    last = batch.true_points[:, 0].copy(); previous = None
    last_k = np.zeros(n, dtype=np.int16); previous_points = np.zeros_like(last); has_previous = np.zeros(n, dtype=bool)
    for k in range(1, steps):
        gap = k - last_k
        predicted_hold = common_mean(last, None)
        predicted_velocity = common_mean(last, previous_points, gap)
        predicted = np.where(has_previous[:, None], predicted_velocity, predicted_hold)
        residual[:, k] = batch.true_points[:, k] - predicted; elapsed[:, k] = gap
        visible = batch.observed[:, k]
        previous_points[visible] = last[visible]; last[visible] = batch.true_points[visible, k]
        has_previous[visible] = True; last_k[visible] = k
    return residual, elapsed


def _fit_data(batch_size: int = 5000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residuals = []; angles = []; observed = []
    for start in range(0, 50_000, batch_size):
        batch = generate_batch("geom_train", start, min(batch_size, 50_000 - start)); residual, _ = transition_residuals(batch)
        residuals.append(residual.astype(np.float32)); angles.append(batch.true_theta.astype(np.float32))
        observed.append((batch.observed & (batch.truth_index == 0)))
    return np.concatenate(residuals), np.concatenate(angles), np.concatenate(observed)


def _objective(
    method: str, names: tuple[str, ...], values: np.ndarray,
    residual: np.ndarray, angles: np.ndarray, observed: np.ndarray,
) -> float:
    params = dict(zip(names, values, strict=True)); total = 0.0; count = 0
    covariance = initial_covariance(method, angles[:, 0], params); theta = angles[:, 0]
    for k in range(1, angles.shape[1]):
        covariance = propagate_covariance(method, covariance, theta, angles[:, k], params); theta = angles[:, k]
        selected = observed[:, k]
        if selected.any():
            total -= float(np.sum(gaussian_log_score(residual[selected, k], covariance[selected])))
            count += int(np.count_nonzero(selected))
    return total / max(1, count)


def fit_method(method: str, fit_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
    names = PARAMETERS[method]; bounds = [tuple(PROTOCOL["parameter_bounds"][name]) for name in names]
    residual, angles, observed = fit_data if fit_data is not None else _fit_data()
    starts = []
    for fraction in (0.25, 0.50, 0.75):
        starts.append(np.asarray([low + fraction * (high - low) for low, high in bounds], dtype=np.float64))
    results = []
    for start in starts:
        result = minimize(lambda value: _objective(method, names, value, residual, angles, observed), start, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 80, "ftol": 1e-9})
        results.append(result)
    best = min(results, key=lambda result: float(result.fun)); params = dict(zip(names, [float(value) for value in best.x], strict=True))
    hits = {name: bool(np.isclose(params[name], low, atol=1e-5) or np.isclose(params[name], high, atol=1e-5))
            for name, (low, high) in zip(names, bounds, strict=True)}
    return {"method": method, "params": params, "train_nll": float(best.fun), "success": bool(best.success),
            "message": str(best.message), "iterations": int(best.nit), "bound_hits": hits,
            "fit_split": "geom_train", "fit_cases": 50_000, "dev_accessed": False}


def fit_all() -> dict[str, Any]:
    data = _fit_data()
    output = {"G0_euclidean": {"method": "G0_euclidean", "params": {}, "fit_split": None, "dev_accessed": False}}
    for method in PARAMETERS:
        output[method] = fit_method(method, data)
        print(f"phase=SURFTRACK-S0-FIT method={method} nll={output[method]['train_nll']:.6f} params={output[method]['params']}", flush=True)
    return output
