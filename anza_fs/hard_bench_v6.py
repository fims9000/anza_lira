"""Frozen on-demand StressBench V6-HARD with explicit pair-level truth."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from anza_hs.orientation_bank import orientation_bank_targets
from synthetic.geometry_generator import Branch, GeometrySample
from synthetic.instance_targets import rasterize_targets
from synthetic.seismic_background import render_seismic


VERSION = "ANZA_FS_STRESSBENCH_V6_HARD_V1"
IMAGE_SIZE = 96
CASES = (
    "parallel_2px", "parallel_3px", "parallel_5px", "parallel_gap_confuser",
    "acute_cross_15", "acute_cross_30", "x_weak_branch", "long_gap_12_20",
    "long_gap_20_32", "s_curve_gap", "history_confuser", "double_fault_clutter",
    "low_contrast_gap", "blurred_gap", "orientation_noise", "partial_occlusion",
)
SPLIT_BASE = {
    "train": 610_000_000,
    "calibration": 620_000_000,
    "development": 630_000_000,
    "confirm": 640_000_000,
}
SPLIT_SIZE = {"train": 512, "calibration": 512, "development": 512, "confirm": 512}
NEGATIVE_EVENTS_CALIBRATION_PLUS_DEVELOPMENT = 1024
POSITIVE_EVENTS_CALIBRATION_PLUS_DEVELOPMENT = 1024


def _curve(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack((x.astype(np.float32), y.astype(np.float32)), axis=1)


def _disk(point_xy: np.ndarray, radius: int = 2) -> np.ndarray:
    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    x, y = float(point_xy[0]), float(point_xy[1])
    return ((xx - x) ** 2 + (yy - y) ** 2 <= radius**2)


def _geometry(case: str, rng: np.random.Generator) -> tuple[GeometrySample, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    x = np.linspace(6.0, 89.0, 168, dtype=np.float32)
    phase = float(rng.uniform(-0.35, 0.35))
    if case == "s_curve_gap":
        positive_y = 25.0 + 10.0 * np.sin(2.0 * np.pi * (x / 96.0 + phase))
    elif case in {"acute_cross_15", "acute_cross_30"}:
        angle = 15.0 if case.endswith("15") else 30.0
        positive_y = 25.0 + np.tan(np.deg2rad(angle)) * (x - 48.0) * 0.18
    elif case == "orientation_noise":
        positive_y = 25.0 + 4.0 * np.sin(x / 9.0 + phase) + rng.normal(0.0, 0.45, x.shape)
    else:
        positive_y = 25.0 + 4.5 * np.sin(x / 16.0 + phase)
    positive = _curve(x, positive_y)

    free_gap = {"parallel_2px": 2.0, "parallel_3px": 3.0, "parallel_5px": 5.0}.get(case, 3.0)
    center_distance = 3.0 + free_gap
    lower_center = 67.0
    upper_y = lower_center - center_distance / 2.0 + 1.1 * np.sin(x / 21.0 + phase)
    lower_y = lower_center + center_distance / 2.0 + 1.1 * np.sin(x / 21.0 + phase)
    upper = _curve(x, upper_y)
    lower = _curve(x, lower_y)
    if case == "parallel_gap_confuser":
        left_x = np.linspace(8.0, 43.0, 72, dtype=np.float32)
        right_x = np.linspace(53.0, 88.0, 72, dtype=np.float32)
        upper = _curve(left_x, np.full_like(left_x, 66.0) + 0.7 * np.sin(left_x / 9.0))
        lower = _curve(right_x, np.full_like(right_x, 66.0) + 0.7 * np.sin(right_x / 9.0))
        negative_points = (upper[-1], lower[0])
    elif case == "history_confuser":
        t = np.linspace(0.0, 1.0, 96, dtype=np.float32)
        upper = _curve(8.0 + 38.0 * t, 64.0 - 8.0 * t + 3.0 * t**2)
        lower = _curve(88.0 - 38.0 * t, 76.0 - 8.0 * t - 3.0 * t**2)
        negative_points = (upper[-1], lower[-1])
    else:
        middle = len(x) // 2
        negative_points = (upper[middle], lower[middle])

    branches = [
        Branch(1, 1, positive, float(rng.uniform(3.0, 5.5))),
        Branch(2, 2, upper, float(rng.uniform(2.5, 5.0))),
        Branch(3, 3, lower, float(rng.uniform(2.5, 5.0))),
    ]
    if case == "x_weak_branch":
        y = np.linspace(7.0, 46.0, 80, dtype=np.float32)
        branches.append(Branch(4, 4, _curve(48.0 + 0.18 * (y - 25.0), y), 1.2))
    elif case in {"acute_cross_15", "acute_cross_30"}:
        angle = np.deg2rad(15.0 if case.endswith("15") else 30.0)
        t = np.linspace(-30.0, 30.0, 100, dtype=np.float32)
        branches.append(Branch(4, 4, _curve(48.0 + t * np.cos(angle), 25.0 + t * np.sin(angle)), 2.0))
    elif case == "double_fault_clutter":
        for branch_id, y0 in ((4, 43.0), (5, 50.0)):
            branches.append(Branch(branch_id, branch_id, _curve(x, y0 + 2.0 * np.sin(x / 11.0 + branch_id)), 1.8))

    geometry = GeometrySample(case, tuple(branches), strata=(case, "v6_hard"))
    positive_points = (positive[len(positive) // 6], positive[5 * len(positive) // 6])
    return geometry, np.stack(positive_points), np.stack(negative_points), (positive, np.concatenate((upper, lower)))


def hard_bench_config() -> dict[str, Any]:
    payload = {
        "version": VERSION,
        "image_size": IMAGE_SIZE,
        "cases": list(CASES),
        "split_seed_base": SPLIT_BASE,
        "split_sizes": SPLIT_SIZE,
        "positive_events_calibration_plus_development": POSITIVE_EVENTS_CALIBRATION_PLUS_DEVELOPMENT,
        "negative_events_calibration_plus_development": NEGATIVE_EVENTS_CALIBRATION_PLUS_DEVELOPMENT,
        "geometry": {
            "minimum_free_parallel_gap_px": 2.0,
            "line_radius_px": 1,
            "positive_event": "two anchors on one continuous visible instance",
            "negative_event": "two anchors on distinct latent instances",
            "long_gap_12_20_attenuation_width_px": 16,
            "long_gap_20_32_attenuation_width_px": 26,
        },
        "confirm_status": "LOCKED_UNOPENED",
        "test_status": "NOT_CREATED",
        "training_seed": 41,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**payload, "sha256": digest}


def generate_hard_sample(split: str, index: int, *, allow_confirm: bool = False) -> dict[str, Any]:
    if split not in SPLIT_BASE:
        raise ValueError(f"unknown V6-HARD split: {split}")
    if split == "confirm" and not allow_confirm:
        raise PermissionError("ANZA-FS V6-HARD confirm is LOCKED_UNOPENED")
    if not 0 <= int(index) < SPLIT_SIZE[split]:
        raise IndexError(index)
    seed = SPLIT_BASE[split] + int(index)
    rng = np.random.default_rng(seed)
    case = CASES[int(index) % len(CASES)]
    geometry, positive_points, negative_points, _paths = _geometry(case, rng)
    targets = rasterize_targets(geometry, IMAGE_SIZE, line_radius=1)
    image = render_seismic(geometry, IMAGE_SIZE, np.random.default_rng(seed + 17_000_000))

    corridor_x0, corridor_x1 = 40, 56
    if case == "long_gap_12_20":
        corridor_x0, corridor_x1 = 40, 56
    elif case == "long_gap_20_32":
        corridor_x0, corridor_x1 = 35, 61
    if case in {"long_gap_12_20", "long_gap_20_32", "low_contrast_gap", "partial_occlusion"}:
        y0 = max(0, int(np.floor(positive_points[:, 1].min())) - 8)
        y1 = min(IMAGE_SIZE, int(np.ceil(positive_points[:, 1].max())) + 9)
        local = image[:, y0:y1, corridor_x0:corridor_x1]
        if case == "partial_occlusion":
            image[:, y0:y1, corridor_x0:corridor_x1] = local.mean(axis=(1, 2), keepdims=True)
        else:
            image[:, y0:y1, corridor_x0:corridor_x1] = np.clip(0.5 + 0.22 * (local - 0.5), 0.0, 1.0)
    if case == "low_contrast_gap":
        image = np.clip(0.5 + 0.42 * (image - 0.5), 0.0, 1.0)
    elif case == "blurred_gap":
        image = gaussian_filter(image, sigma=(0.0, 1.35, 1.35)).astype(np.float32)
    elif case == "orientation_noise":
        image = np.clip(image + np.random.default_rng(seed + 29_000_000).normal(0.0, 0.16, image.shape), 0.0, 1.0).astype(np.float32)

    positive_anchors = np.stack([_disk(point) for point in positive_points])
    negative_anchors = np.stack([_disk(point) for point in negative_points])
    bank, valid = orientation_bank_targets(targets)
    return {
        "image": np.asarray(image, dtype=np.float32),
        **targets,
        "orientation_bank_target": bank,
        "orientation_valid": valid,
        "positive_anchor_masks": positive_anchors,
        "negative_anchor_masks": negative_anchors,
        "positive_event_count": 1,
        "negative_event_count": 1,
        "case": case,
        "split": split,
        "index": int(index),
        "seed": seed,
        "image_size": IMAGE_SIZE,
        "benchmark_version": VERSION,
    }


def freeze_hard_bench(path: Path) -> dict[str, Any]:
    payload = hard_bench_config()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != encoded:
        raise ValueError("StressBench V6-HARD protocol drift")
    path.write_text(encoded)
    return payload
