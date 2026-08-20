"""Five deterministic higher-order tasks matched exactly on frozen static features."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from .static_signature import match_in_static_nullspace, static_signature


VERSION = "ANZA_KS_DYNAMICS_MATCHED_V1"
PATCH_SIZE = 17
TASKS = (
    "continuous_vs_rearranged_ridge",
    "branch_history_switch",
    "s_curve_vs_moment_matched_zigzag",
    "ridge_vs_structured_clutter",
    "gap_continuation_vs_compensated_distractor",
)
SPLIT_SIZES = {"train": 2048, "dev": 1024, "confirm": 2048}
SPLIT_SEEDS = {"train": 811_000_000, "dev": 823_000_000, "confirm": 839_000_000}


def _rng(task: str, split: str, index: int) -> np.random.Generator:
    if task not in TASKS or split not in SPLIT_SIZES or not 0 <= index < SPLIT_SIZES[split]:
        raise ValueError("unknown task/split/index")
    task_offset = TASKS.index(task) * 100_000
    return np.random.default_rng(SPLIT_SEEDS[split] + task_offset + int(index))


def _ridge(y_path: np.ndarray, *, width: float = 0.65, amplitude: np.ndarray | float = 1.0) -> np.ndarray:
    y, x = np.meshgrid(np.arange(PATCH_SIZE), np.arange(PATCH_SIZE), indexing="ij")
    path = np.asarray(y_path, dtype=np.float64)
    amp = np.broadcast_to(np.asarray(amplitude, dtype=np.float64), (PATCH_SIZE,))
    return amp[x] * np.exp(-0.5 * ((y - path[x]) / width) ** 2)


def _base_candidates(task: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, PATCH_SIZE)
    center = 8.0 + rng.uniform(-0.4, 0.4)
    width = rng.uniform(0.52, 0.78)
    phase = rng.uniform(-np.pi, np.pi)
    if task == TASKS[0]:
        path = center + 1.2 * np.sin(1.1 * np.pi * x + phase)
        positive = _ridge(path, width=width)
        permutation = np.asarray([0, 1, 5, 6, 2, 3, 10, 11, 7, 8, 15, 16, 12, 13, 4, 9, 14])
        negative = _ridge(path[permutation], width=width)
    elif task == TASKS[1]:
        upper = center - 2.1 + 0.35 * x
        lower = center + 2.1 - 0.35 * x
        positive = _ridge(upper, width=width) + 0.55 * _ridge(lower, width=width)
        switched = np.where(np.arange(PATCH_SIZE) < 8, upper, lower)
        negative = _ridge(switched, width=width) + 0.55 * _ridge(np.where(np.arange(PATCH_SIZE) < 8, lower, upper), width=width)
    elif task == TASKS[2]:
        smooth = center + 2.0 * np.sin(np.pi * x)
        zigzag = center + 2.0 * np.interp(x, [-1.0, -0.5, 0.0, 0.5, 1.0], [0.0, 1.0, -1.0, 1.0, 0.0])
        positive = _ridge(smooth, width=width)
        negative = _ridge(zigzag, width=width)
    elif task == TASKS[3]:
        path = center + 1.0 * np.sin(0.8 * np.pi * x + phase)
        positive = _ridge(path, width=width)
        negative = np.zeros_like(positive)
        for start, end, offset in ((0, 4, -2.0), (4, 8, 1.8), (9, 13, -1.4), (13, 17, 2.2)):
            fragment = _ridge(path + offset, width=width)
            negative[:, start:end] += fragment[:, start:end]
    elif task == TASKS[4]:
        path = center + 0.8 * np.sin(np.pi * x + phase)
        positive = _ridge(path, width=width)
        positive[:, 7:10] *= 0.32
        negative = _ridge(path, width=width)
        negative[:, 7:10] *= 0.05
        distractor = _ridge(path + 2.4, width=width)
        negative[:, 6:11] += 0.42 * distractor[:, 6:11]
    else:  # pragma: no cover
        raise AssertionError(task)
    common = 0.12 * np.sin((np.arange(PATCH_SIZE)[None, :] + rng.uniform(0, 6)) * rng.uniform(0.8, 1.4))
    common = np.broadcast_to(common, positive.shape)
    texture = gaussian_filter(rng.normal(size=positive.shape), sigma=(0.7, 1.6))
    positive = positive + common + 0.10 * texture
    negative = negative + common + 0.10 * texture
    return positive, negative


def _generate(task: str, split: str, index: int) -> dict[str, Any]:
    rng = _rng(task, split, index)
    positive, negative = _base_candidates(task, rng)
    positive, negative = match_in_static_nullspace(positive, negative)
    static_delta = float(np.linalg.norm(static_signature(positive) - static_signature(negative)))
    return {
        "positive": positive,
        "negative": negative,
        "task": task,
        "split": split,
        "index": int(index),
        "orientation": 0.0,
        "static_delta": static_delta,
        "pixel_equal": bool(np.array_equal(positive, negative)),
        "l2_difference": float(np.linalg.norm(positive - negative)),
    }


def generate_pair(task: str, split: str, index: int) -> dict[str, Any]:
    if split == "confirm":
        raise PermissionError("ANZA-KS confirm is hash-only and access-locked in K0/K1")
    return _generate(task, split, index)


def confirm_stream_hash() -> str:
    """Generate only raw confirm bytes for a pre-experiment lock; expose no samples."""

    digest = hashlib.sha256()
    for task in TASKS:
        for index in range(SPLIT_SIZES["confirm"]):
            pair = _generate(task, "confirm", index)
            digest.update(task.encode())
            digest.update(index.to_bytes(4, "little"))
            digest.update(pair["positive"].tobytes())
            digest.update(pair["negative"].tobytes())
    return digest.hexdigest()


def benchmark_manifest() -> dict[str, Any]:
    return {
        "version": VERSION,
        "patch_size": PATCH_SIZE,
        "tasks": list(TASKS),
        "split_sizes": SPLIT_SIZES,
        "split_seeds": SPLIT_SEEDS,
        "static_matching": "orthogonal projection into the nullspace of a frozen static measurement bank; no Cat or information feature is imported",
        "static_tolerance": 1e-7,
        "confirm_policy": "generated and content-hashed in K0; samples and scores inaccessible until a future separately frozen confirm phase",
        "anza_ks_used_for_generation": False,
    }
