"""Independent connectivity/diffusion development stream; test is fail-closed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .crossing_trace_bench_v2 import tangent_set_targets
from .crossing_trace_bench_v3 import CONTEXT_CASES, PAIRED_GAP_COUNT
from .geometry_generator import generate_geometry, scale_geometry
from .instance_targets import rasterize_targets
from .seismic_background import render_seismic


SPLIT_SEED_BASE_V5 = {
    "train": 410_000_000,
    "validation": 420_000_000,
    "confirm": 430_000_000,
    "test": 440_000_000,
}
SPLIT_SIZES_V5 = {name: 512 for name in SPLIT_SEED_BASE_V5}


def benchmark_v5_config() -> dict[str, Any]:
    payload = {
        "version": "crossing_trace_bench_v5_connectivity_diffusion",
        "split_seed_base": SPLIT_SEED_BASE_V5,
        "split_sizes": SPLIT_SIZES_V5,
        "paired_positive_gaps": PAIRED_GAP_COUNT,
        "paired_negative_gaps": PAIRED_GAP_COUNT,
        "context_cases": list(CONTEXT_CASES),
        "test_status": "LOCKED_UNOPENED",
        "legacy_v3_v4_selection_reused": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def _pair_id(index: int) -> int | None:
    if 0 <= index < PAIRED_GAP_COUNT:
        return index
    if PAIRED_GAP_COUNT <= index < 2 * PAIRED_GAP_COUNT:
        return index - PAIRED_GAP_COUNT
    return None


def _case(index: int) -> tuple[str, int | None]:
    pair = _pair_id(index)
    if index < PAIRED_GAP_COUNT:
        return "fault_with_gap", pair
    if index < 2 * PAIRED_GAP_COUNT:
        return "negative_gap", pair
    return CONTEXT_CASES[(index - 2 * PAIRED_GAP_COUNT) % len(CONTEXT_CASES)], None


def sample_seed_v5(split: str, index: int) -> int:
    if split not in SPLIT_SEED_BASE_V5:
        raise ValueError(f"unknown v5 split: {split}")
    if not 0 <= int(index) < SPLIT_SIZES_V5[split]:
        raise IndexError(index)
    pair = _pair_id(int(index))
    return SPLIT_SEED_BASE_V5[split] + (pair if pair is not None else int(index))


def _render_sample_v5(split: str, index: int, *, image_size: int = 128) -> dict[str, Any]:
    seed = sample_seed_v5(split, index)
    case, pair_id = _case(int(index))
    geometry = scale_geometry(generate_geometry(case, np.random.default_rng(seed)), image_size)
    targets = rasterize_targets(geometry, image_size)
    return {
        "image": render_seismic(geometry, image_size, np.random.default_rng(seed + 19_000_000)),
        **targets,
        **tangent_set_targets(geometry, targets, image_size),
        "case": case,
        "pair_id": pair_id,
        "split": split,
        "index": int(index),
        "seed": int(seed),
        "image_size": int(image_size),
        "benchmark_version": "crossing_trace_bench_v5_connectivity_diffusion",
    }


def generate_sample_v5(split: str, index: int, *, image_size: int = 128) -> dict[str, Any]:
    if split == "test":
        raise PermissionError("CrossingTraceBench-v5 test is LOCKED_UNOPENED")
    return _render_sample_v5(split, index, image_size=image_size)


def generate_authorized_test_sample_v5(
    index: int,
    *,
    calibration_freeze: dict[str, Any],
    image_size: int = 128,
) -> dict[str, Any]:
    """Open v5 test only with an immutable validation-calibration receipt."""

    payload = dict(calibration_freeze)
    digest = payload.pop("freeze_sha256", None)
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    required = {
        "status": "CALIBRATION_FROZEN",
        "v5_test_samples_opened": 0,
        "old_confirm_used_for_calibration": False,
        "expert_data_accessed": False,
    }
    if digest != expected or any(payload.get(key) != value for key, value in required.items()):
        raise PermissionError("v5 test authorization requires a valid frozen validation calibration")
    protocol = payload.get("protocol", {})
    if protocol.get("v5_test") != "LOCKED_UNOPENED" or "validation" not in protocol.get("calibration_stream", ""):
        raise PermissionError("v5 test authorization provenance is invalid")
    return _render_sample_v5("test", index, image_size=image_size)


def freeze_benchmark_v5_config(path: Path) -> dict[str, Any]:
    payload = benchmark_v5_config()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("CrossingTraceBench-v5 config drift")
    path.write_text(encoded)
    return payload
