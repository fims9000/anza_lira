"""Independent structural-affinity development stream (test fail-closed)."""

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


SPLIT_SEED_BASE_V4 = {
    "train": 310_000_000,
    "validation": 320_000_000,
    "confirm": 330_000_000,
    "test": 340_000_000,
}
SPLIT_SIZES_V4 = {name: 512 for name in SPLIT_SEED_BASE_V4}
LOCAL8_OFFSETS = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
RADIUS2_OFFSETS = ((-2, -2), (0, -2), (2, -2), (-2, 0), (2, 0), (-2, 2), (0, 2), (2, 2))


def benchmark_v4_config() -> dict[str, Any]:
    payload = {
        "version": "crossing_trace_bench_v4_affinity",
        "split_seed_base": SPLIT_SEED_BASE_V4,
        "split_sizes": SPLIT_SIZES_V4,
        "paired_positive_gaps": PAIRED_GAP_COUNT,
        "paired_negative_gaps": PAIRED_GAP_COUNT,
        "context_cases": list(CONTEXT_CASES),
        "local8_offsets": [list(item) for item in LOCAL8_OFFSETS],
        "radius2_offsets": [list(item) for item in RADIUS2_OFFSETS],
        "test_status": "LOCKED_UNOPENED",
        "legacy_v3_selection_reused": False,
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


def sample_seed_v4(split: str, index: int) -> int:
    if split not in SPLIT_SEED_BASE_V4:
        raise ValueError(f"unknown v4 split: {split}")
    if not 0 <= int(index) < SPLIT_SIZES_V4[split]:
        raise IndexError(index)
    pair = _pair_id(int(index))
    return SPLIT_SEED_BASE_V4[split] + (pair if pair is not None else int(index))


def generate_sample_v4(split: str, index: int, *, image_size: int = 128) -> dict[str, Any]:
    if split == "test":
        raise PermissionError("CrossingTraceBench-v4 test is LOCKED_UNOPENED")
    seed = sample_seed_v4(split, index)
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
        "benchmark_version": "crossing_trace_bench_v4_affinity",
    }


def freeze_benchmark_v4_config(path: Path) -> dict[str, Any]:
    payload = benchmark_v4_config()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("CrossingTraceBench-v4 config drift")
    path.write_text(encoded)
    return payload
