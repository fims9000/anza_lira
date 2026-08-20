"""Deterministic paired bootstrap over independent synthetic pairs."""

from __future__ import annotations

import numpy as np


def bootstrap_macro_ranking_delta(
    candidate_differences: dict[str, np.ndarray],
    control_differences: dict[str, np.ndarray],
    *,
    resamples: int = 10_000,
    seed: int = 941_019,
) -> dict[str, float | int]:
    tasks = tuple(candidate_differences)
    if tasks != tuple(control_differences):
        raise ValueError("candidate/control tasks do not align")
    observed_by_task = []
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros(resamples, dtype=np.float64)
    for task in tasks:
        candidate = np.asarray(candidate_differences[task], dtype=np.float64)
        control = np.asarray(control_differences[task], dtype=np.float64)
        if candidate.shape != control.shape or candidate.ndim != 1:
            raise ValueError("paired difference arrays must align")
        indicator_candidate = (candidate > 0) + 0.5 * (candidate == 0)
        indicator_control = (control > 0) + 0.5 * (control == 0)
        delta = indicator_candidate - indicator_control
        observed_by_task.append(float(delta.mean()))
        indices = rng.integers(0, len(delta), size=(resamples, len(delta)))
        bootstrap += delta[indices].mean(axis=1) / len(tasks)
    observed = float(np.mean(observed_by_task))
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "observed_macro_ranking_delta": observed,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "resamples": int(resamples),
        "unit": "paired synthetic example within each predeclared task, task-macro averaged",
    }
