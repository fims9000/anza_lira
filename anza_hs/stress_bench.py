"""Frozen ANZA-HS StressBench V5, generated on demand before H1 training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from synthetic.crossing_trace_bench_v2 import tangent_set_targets
from synthetic.geometry_generator import Branch, Gap, GeometrySample, generate_geometry, scale_geometry
from synthetic.instance_targets import rasterize_targets
from synthetic.seismic_background import render_seismic


VERSION = "anza_hs_stressbench_v5"
CASES = (
    "long_curved_gap", "s_shaped_fault", "close_parallel", "x_crossing", "acute_crossing",
    "weak_branch", "low_contrast", "local_blur", "orientation_noise", "partial_occlusion",
    "history_confuser",
)
SPLIT_BASE = {"train": 510_000_000, "dev": 520_000_000, "confirm": 530_000_000}
SPLIT_SIZE = {"train": 352, "dev": 264, "confirm": 264}
IMAGE_SIZE = 64
DEV_CALIBRATION_COUNT = 44


def _quadratic(points: tuple[tuple[float, float], ...], samples: int = 96) -> np.ndarray:
    controls = np.asarray(points, dtype=np.float32)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)[:, None]
    return (1 - t) ** 2 * controls[0] + 2 * (1 - t) * t * controls[1] + t**2 * controls[2]


def _custom_geometry(case: str, rng: np.random.Generator) -> GeometrySample:
    throw = float(rng.uniform(3.0, 6.0))
    if case == "long_curved_gap":
        curve = _quadratic(((0.05, 0.82), (0.45, 0.08), (0.95, 0.36)), samples=120)
        left, right = curve[:49], curve[71:]
        gap = curve[48:72]
        return GeometrySample(case, (Branch(1, 1, left, throw), Branch(2, 1, right, throw)), gaps=(Gap("positive", gap, 1),), strata=(case,))
    if case == "s_shaped_fault":
        t = np.linspace(0.0, 1.0, 128, dtype=np.float32)
        x = 0.06 + 0.88 * t
        y = 0.50 + 0.28 * np.sin(2.0 * np.pi * (t + 0.10))
        return GeometrySample(case, (Branch(1, 1, np.stack((x, y), axis=1), throw),), strata=(case,))
    raise ValueError(case)


def _base_geometry(case: str, rng: np.random.Generator) -> GeometrySample:
    if case in {"long_curved_gap", "s_shaped_fault"}:
        return _custom_geometry(case, rng)
    source = {
        "close_parallel": "near_parallel", "x_crossing": "x_junction",
        "acute_crossing": "acute_angle_crossing", "weak_branch": "weak_branch_crossing",
        "low_contrast": "single_straight", "local_blur": "curved_fault",
        "orientation_noise": "curved_crossing", "partial_occlusion": "fault_with_gap",
        "history_confuser": "close_non_intersecting",
    }[case]
    geometry = generate_geometry(source, rng)
    return GeometrySample(case, geometry.branches, geometry.junctions, geometry.gaps, tuple((*geometry.strata, case)))


def stressbench_config() -> dict[str, Any]:
    payload = {
        "version": VERSION, "cases": list(CASES), "split_seed_base": SPLIT_BASE,
        "split_sizes": SPLIT_SIZE, "image_size": IMAGE_SIZE,
        "dev_calibration": f"dev[0:{DEV_CALIBRATION_COUNT}]",
        "dev_gate": f"dev[{DEV_CALIBRATION_COUNT}:{SPLIT_SIZE['dev']}]",
        "confirm_status": "LOCKED_UNOPENED", "test_status": "NOT_CREATED",
        "training_seed": 41,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**payload, "sha256": digest}


def generate_stress_sample(split: str, index: int, *, allow_confirm: bool = False) -> dict[str, Any]:
    if split not in SPLIT_BASE:
        raise ValueError(f"unknown StressBench split: {split}")
    if split == "confirm" and not allow_confirm:
        raise PermissionError("ANZA-HS StressBench confirm is LOCKED_UNOPENED")
    if not 0 <= int(index) < SPLIT_SIZE[split]:
        raise IndexError(index)
    seed = SPLIT_BASE[split] + int(index)
    case = CASES[int(index) % len(CASES)]
    rng = np.random.default_rng(seed)
    geometry = scale_geometry(_base_geometry(case, rng), IMAGE_SIZE)
    targets = rasterize_targets(geometry, IMAGE_SIZE)
    image = render_seismic(geometry, IMAGE_SIZE, np.random.default_rng(seed + 19_000_000))
    if case == "low_contrast":
        image = np.clip(0.5 + 0.30 * (image - 0.5), 0.0, 1.0)
    elif case == "local_blur":
        image = gaussian_filter(image, sigma=(0.0, 1.25, 1.25)).astype(np.float32)
    elif case == "orientation_noise":
        image = np.clip(image + np.random.default_rng(seed + 23_000_000).normal(0.0, 0.12, image.shape), 0.0, 1.0).astype(np.float32)
    elif case == "partial_occlusion":
        y0, y1 = 27, 37; x0, x1 = 25, 39
        image[:, y0:y1, x0:x1] = np.mean(image[:, y0:y1, x0:x1], axis=(1, 2), keepdims=True)
        targets["visible_fault_mask"][y0:y1, x0:x1] = False
        targets["visible_centerline_map"][y0:y1, x0:x1] = False
        targets["branch_masks"][:, y0:y1, x0:x1] = False
        targets["branch_centerlines"][:, y0:y1, x0:x1] = False
        targets["branch_tangent_cos2"][:, y0:y1, x0:x1] = 0.0
        targets["branch_tangent_sin2"][:, y0:y1, x0:x1] = 0.0
    return {
        "image": np.asarray(image, dtype=np.float32), **targets, **tangent_set_targets(geometry, targets, IMAGE_SIZE),
        "case": case, "split": split, "index": int(index), "seed": seed,
        "image_size": IMAGE_SIZE, "benchmark_version": VERSION,
    }


def freeze_stressbench(path: Path) -> dict[str, Any]:
    payload = stressbench_config(); encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != encoded:
        raise ValueError("StressBench V5 protocol drift")
    path.write_text(encoded)
    return payload
