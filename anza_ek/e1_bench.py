"""Frozen identifiable zero-training patch pairs for ANZA-EK E1."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


VERSION = "ANZA_EK_E1_PATCH_BENCH_V1"
TASKS = (
    "straight_ridge_vs_blob",
    "faint_visible_continuation",
    "crossing_correct_vs_wrong",
    "close_parallel_separation",
    "curved_local_ridge",
    "oriented_clutter",
)
PATCH_SIZE = 65
PAIRS_PER_TASK = 256
SEED_BASE = 710_000_000
ORIENTATION_COUNT = 8


def _coordinates(theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis = np.arange(PATCH_SIZE, dtype=np.float64) / PATCH_SIZE - 0.5
    y, x = np.meshgrid(axis, axis, indexing="ij")
    cosine, sine = math.cos(theta), math.sin(theta)
    u = cosine * x + sine * y
    v = -sine * x + cosine * y
    return x, y, u, v


def _ridge(u: np.ndarray, v: np.ndarray, *, curve: np.ndarray | float = 0.0, width: float = 0.035, length: float = 0.43, amplitude: float = 1.0) -> np.ndarray:
    transverse = np.exp(-0.5 * ((v - curve) / width) ** 2)
    longitudinal = np.exp(-0.5 * (np.abs(u) / length) ** 8)
    return amplitude * transverse * longitudinal


def _background(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    phase = float(rng.uniform(-math.pi, math.pi))
    slope = float(rng.uniform(-0.7, 0.7))
    layers = 0.09 * np.sin(2 * math.pi * (5.5 * y + slope * x) + phase)
    texture = gaussian_filter(rng.normal(0.0, 0.055, x.shape), sigma=0.65)
    return layers + texture


def _half_mask(u: np.ndarray, side: str, gap: float = 0.035) -> np.ndarray:
    sharpness = 90.0
    if side == "left":
        return 1.0 / (1.0 + np.exp(sharpness * (u + gap)))
    if side == "right":
        return 1.0 / (1.0 + np.exp(-sharpness * (u - gap)))
    raise ValueError(side)


def _clutter(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, count: int = 6) -> np.ndarray:
    value = np.zeros_like(x)
    for _ in range(count):
        angle = float(rng.uniform(0.0, math.pi))
        cosine, sine = math.cos(angle), math.sin(angle)
        x0, y0 = rng.uniform(-0.30, 0.30, size=2)
        u = cosine * (x - x0) + sine * (y - y0)
        v = -sine * (x - x0) + cosine * (y - y0)
        value += _ridge(u, v, width=float(rng.uniform(0.022, 0.045)), length=float(rng.uniform(0.08, 0.20)), amplitude=float(rng.uniform(0.20, 0.48)))
    return value


def _perturb(value: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shift_y, shift_x = rng.integers(-1, 2, size=2)
    result = np.roll(value, (int(shift_y), int(shift_x)), axis=(0, 1))
    result = gaussian_filter(result, sigma=0.55)
    result = 0.92 * result + rng.normal(0.0, 0.035, result.shape)
    return result.astype(np.float64)


def generate_pair(task: str, index: int) -> dict[str, Any]:
    if task not in TASKS:
        raise ValueError(f"unknown E1 task: {task}")
    if not 0 <= int(index) < PAIRS_PER_TASK:
        raise IndexError(index)
    task_index = TASKS.index(task)
    seed = SEED_BASE + task_index * 1_000_000 + int(index)
    rng = np.random.default_rng(seed)
    orientation_index = int(rng.integers(0, ORIENTATION_COUNT))
    theta = orientation_index * math.pi / ORIENTATION_COUNT
    x, y, u, v = _coordinates(theta)
    background = _background(x, y, rng)
    positive = background.copy()
    negative = background.copy()

    if task == "straight_ridge_vs_blob":
        positive += _ridge(u, v, width=0.032, amplitude=0.85)
        negative += 0.85 * np.exp(-0.5 * (u**2 + v**2) / 0.10**2)
    elif task == "faint_visible_continuation":
        attenuation = np.where(np.abs(u) < 0.105, 0.24, 1.0)
        positive += attenuation * _ridge(u, v, width=0.032, amplitude=0.90)
        negative += _half_mask(u, "left") * _ridge(u, v + 0.045, width=0.032, amplitude=0.90)
        negative += _half_mask(u, "right") * _ridge(u, v - 0.065 - 0.20 * u, width=0.032, amplitude=0.90)
    elif task == "crossing_correct_vs_wrong":
        cross_angle = theta + math.pi / 3.0
        cu = math.cos(cross_angle) * x + math.sin(cross_angle) * y
        cv = -math.sin(cross_angle) * x + math.cos(cross_angle) * y
        positive += _ridge(u, v, width=0.030, amplitude=0.82) + _ridge(cu, cv, width=0.030, amplitude=0.58)
        negative += _half_mask(u, "left", 0.0) * _ridge(u, v, width=0.030, amplitude=0.82)
        negative += _half_mask(cu, "right", 0.0) * _ridge(cu, cv, width=0.030, amplitude=0.82)
        negative += 0.22 * _half_mask(u, "right", 0.02) * _ridge(u, v + 0.055, width=0.030)
    elif task == "close_parallel_separation":
        positive += _ridge(u, v, width=0.030, amplitude=0.88)
        negative += _ridge(u, v - 0.073, width=0.027, amplitude=0.72)
        negative += _ridge(u, v + 0.073, width=0.027, amplitude=0.72)
    elif task == "curved_local_ridge":
        positive_curve = 0.72 * u**2 - 0.012
        positive += _ridge(u, v, curve=positive_curve, width=0.030, amplitude=0.88)
        left_curve = 0.65 * u**2 + 0.025
        right_curve = -0.85 * u**2 - 0.045 + 0.16 * u
        negative += _half_mask(u, "left", 0.01) * _ridge(u, v, curve=left_curve, width=0.030, amplitude=0.88)
        negative += _half_mask(u, "right", 0.01) * _ridge(u, v, curve=right_curve, width=0.030, amplitude=0.88)
    elif task == "oriented_clutter":
        shared_clutter = _clutter(x, y, rng)
        positive += shared_clutter + _ridge(u, v, width=0.028, amplitude=0.72)
        negative += shared_clutter + _clutter(x, y, rng, count=2)

    perturb_rng = np.random.default_rng(seed + 90_000_000)
    positive_perturbed = _perturb(positive, perturb_rng)
    negative_perturbed = _perturb(negative, perturb_rng)
    return {
        "task": task,
        "index": int(index),
        "seed": seed,
        "orientation_index": orientation_index,
        "orientation": theta,
        "positive": positive.astype(np.float64),
        "negative": negative.astype(np.float64),
        "positive_perturbed": positive_perturbed,
        "negative_perturbed": negative_perturbed,
        "pixel_equal": bool(np.array_equal(positive, negative)),
        "l2_difference": float(np.linalg.norm(positive - negative)),
    }


def benchmark_config() -> dict[str, Any]:
    payload = {
        "version": VERSION,
        "tasks": list(TASKS),
        "patch_size": PATCH_SIZE,
        "pairs_per_task": PAIRS_PER_TASK,
        "seed_base": SEED_BASE,
        "orientation_count": ORIENTATION_COUNT,
        "identifiability": "positive and negative observed geometry differ; exact pixel equality forbidden",
        "perturbation": {"translation_px": 1, "gaussian_blur_sigma": 0.55, "noise_sigma": 0.035, "amplitude_scale": 0.92},
        "training": False,
        "classifier": False,
        "confirm_created": False,
        "cracks_accessed": False,
        "expert_accessed": False,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**payload, "sha256": digest}


def freeze_benchmark(path: Path) -> dict[str, Any]:
    payload = benchmark_config()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != encoded:
        raise ValueError("ANZA-EK E1 benchmark drift")
    path.write_text(encoded)
    return payload
