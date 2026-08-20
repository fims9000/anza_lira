"""Margin calibration, risk coverage, and paired scene bootstrap."""

from __future__ import annotations

import numpy as np


def fit_margin_calibration(margin: np.ndarray, error: np.ndarray, bins: int = 20) -> dict:
    finite = np.isfinite(margin); values = margin[finite]; failures = error[finite]
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    rows = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        selected = (values >= low) & (values <= high if high == edges[-1] else values < high)
        rows.append({"low": float(low), "high": float(high), "count": int(np.count_nonzero(selected)),
                     "empirical_error": float(np.mean(failures[selected])) if selected.any() else 0.0})
    return {"bins": rows, "source": "geom_calibration", "dev_accessed": False}


def calibrated_confidence(margin: np.ndarray, calibration: dict) -> np.ndarray:
    confidence = np.zeros_like(margin, dtype=float)
    for row in calibration["bins"]:
        selected = (margin >= row["low"]) & (margin <= row["high"])
        confidence[selected] = 1.0 - row["empirical_error"]
    confidence[np.isposinf(margin)] = 1.0
    return confidence


def risk_coverage(confidence: np.ndarray, error: np.ndarray) -> list[dict]:
    order = np.argsort(-confidence, kind="stable"); rows = []
    for coverage in np.linspace(0.05, 1.0, 20):
        count = max(1, int(round(coverage * len(order)))); selected = order[:count]
        rows.append({"coverage": float(coverage), "risk": float(np.mean(error[selected])), "count": count})
    return rows


def paired_bootstrap(a: np.ndarray, b: np.ndarray, *, resamples: int = 10_000, seed: int = 7301) -> dict:
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float); rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64); batch = 100
    for start in range(0, resamples, batch):
        size = min(batch, resamples - start); indices = rng.integers(0, len(delta), size=(size, len(delta)))
        samples[start:start + size] = np.mean(delta[indices], axis=1)
    return {"mean_delta": float(np.mean(delta)), "ci95_low": float(np.quantile(samples, .025)),
            "ci95_high": float(np.quantile(samples, .975)), "resamples": resamples, "unit": "synthetic_scene"}
