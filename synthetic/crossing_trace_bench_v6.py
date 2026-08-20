"""Independent predicted-endpoint development/test stream for path completion."""

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


SPLIT_SEED_BASE_V6 = {"development": 510_000_000, "test": 520_000_000}
SPLIT_SIZES_V6 = {"development": 512, "test": 512}


def benchmark_v6_config() -> dict[str, Any]:
    core = {
        "version": "crossing_trace_bench_v6_predicted_endpoints",
        "split_seed_base": SPLIT_SEED_BASE_V6,
        "split_sizes": SPLIT_SIZES_V6,
        "paired_positive_gaps": PAIRED_GAP_COUNT,
        "paired_negative_gaps": PAIRED_GAP_COUNT,
        "context_cases": list(CONTEXT_CASES),
        "test_status": "LOCKED_UNOPENED",
        "v5_results_used_for_selection": False,
    }
    return {**core, "sha256": hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def _case(index: int) -> tuple[str, int | None]:
    if index < PAIRED_GAP_COUNT:
        return "fault_with_gap", index
    if index < 2 * PAIRED_GAP_COUNT:
        return "negative_gap", index - PAIRED_GAP_COUNT
    return CONTEXT_CASES[(index - 2 * PAIRED_GAP_COUNT) % len(CONTEXT_CASES)], None


def _render(split: str, index: int, image_size: int) -> dict[str, Any]:
    if split not in SPLIT_SEED_BASE_V6 or not 0 <= int(index) < SPLIT_SIZES_V6[split]:
        raise ValueError(f"invalid v6 sample {split}:{index}")
    case, pair_id = _case(int(index))
    seed = SPLIT_SEED_BASE_V6[split] + (int(pair_id) if pair_id is not None else int(index))
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
        "benchmark_version": benchmark_v6_config()["version"],
    }


def generate_sample_v6(split: str, index: int, *, image_size: int = 128) -> dict[str, Any]:
    if split == "test":
        raise PermissionError("CrossingTraceBench-v6 test is LOCKED_UNOPENED")
    return _render(split, index, image_size)


def freeze_benchmark_v6_config(path: Path) -> dict[str, Any]:
    payload = benchmark_v6_config()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != encoded:
        raise ValueError("CrossingTraceBench-v6 config drift")
    path.write_text(encoded)
    return payload

