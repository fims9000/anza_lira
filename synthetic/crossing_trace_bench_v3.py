"""Frozen context-repair stream with paired gap controls and gate targets.

The v3 stream is independent from the opened legacy test and the completed v2
repair cycle.  Its test split is deliberately inaccessible from this module;
the bounded B cycle may use train, validation, and (after candidate selection)
confirm only.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import Dataset

from .crossing_trace_bench_v2 import tangent_set_targets
from .geometry_generator import GeometrySample, generate_geometry, scale_geometry
from .instance_targets import rasterize_targets
from .seismic_background import render_seismic


SPLIT_SEED_BASE_V3 = {
    "train": 210_000_000,
    "validation": 220_000_000,
    "confirm": 230_000_000,
    "test": 240_000_000,
}
SPLIT_SIZES_V3 = {name: 512 for name in SPLIT_SEED_BASE_V3}
PAIRED_GAP_COUNT = 128
GATE_SIGMA_PX = 2.5
TEST_LOCK_STATUS = "LOCKED_UNOPENED"
CONTEXT_CASES = (
    "x_junction",
    "t_junction",
    "y_junction",
    "acute_angle_crossing",
    "near_parallel",
    "curved_fault",
    "curved_crossing",
    "similar_tangent_crossing",
    "nontrivial_pairing",
    "close_non_intersecting",
    "crossing_near_junction",
    "asymmetric_crossing",
    "weak_branch_crossing",
)
HARD_GATE_NEGATIVE_CASES = {
    "near_parallel",
    "curved_fault",
    "close_non_intersecting",
}


def _protocol_payload() -> dict[str, Any]:
    return {
        "version": "crossing_trace_bench_v3",
        "split_seed_base": SPLIT_SEED_BASE_V3,
        "split_sizes": SPLIT_SIZES_V3,
        "paired_positive_gaps_per_split": PAIRED_GAP_COUNT,
        "paired_negative_gaps_per_split": PAIRED_GAP_COUNT,
        "context_cases": list(CONTEXT_CASES),
        "gate_sigma_px": GATE_SIGMA_PX,
        "test_status": TEST_LOCK_STATUS,
        "legacy_test_stream": "IMMUTABLE_NOT_REUSED",
        "v2_stream": "IMMUTABLE_NOT_REUSED",
        "matching_variables": [
            "gap_length_px",
            "endpoint_distance_px",
            "local_axial_orientation_rad",
            "geometry_seed",
            "render_difficulty_seed",
        ],
    }


def benchmark_v3_config() -> dict[str, Any]:
    payload = _protocol_payload()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def _pair_id(index: int) -> int | None:
    if 0 <= index < PAIRED_GAP_COUNT:
        return index
    if PAIRED_GAP_COUNT <= index < 2 * PAIRED_GAP_COUNT:
        return index - PAIRED_GAP_COUNT
    return None


def sample_seed_v3(split: str, index: int) -> int:
    if split not in SPLIT_SEED_BASE_V3:
        raise ValueError(f"Unknown CrossingTraceBench-v3 split: {split}")
    if not 0 <= int(index) < SPLIT_SIZES_V3[split]:
        raise IndexError(index)
    pair = _pair_id(int(index))
    # A positive/negative pair intentionally shares its geometry difficulty
    # seed.  Lineage semantics, not a resampled geometry, define the label.
    offset = pair if pair is not None else int(index)
    return SPLIT_SEED_BASE_V3[split] + offset


def _scheduled_case(index: int) -> tuple[str, int | None]:
    pair = _pair_id(index)
    if index < PAIRED_GAP_COUNT:
        return "fault_with_gap", pair
    if index < 2 * PAIRED_GAP_COUNT:
        return "negative_gap", pair
    return CONTEXT_CASES[(index - 2 * PAIRED_GAP_COUNT) % len(CONTEXT_CASES)], None


def _gap_match_metadata(geometry: GeometrySample, pair_id: int | None, seed: int) -> dict[str, Any] | None:
    if pair_id is None or not geometry.gaps:
        return None
    gap = geometry.gaps[0]
    delta = np.asarray(gap.points_xy[-1] - gap.points_xy[0], dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(gap.points_xy.astype(np.float64), axis=0), axis=1)
    return {
        "pair_id": int(pair_id),
        "gap_length_px": float(segment_lengths.sum()),
        "endpoint_distance_px": float(np.linalg.norm(delta)),
        "local_axial_orientation_rad": float(math.atan2(float(delta[1]), float(delta[0])) % math.pi),
        "geometry_seed": int(seed),
        "render_difficulty_seed": int(seed + 17_000_000),
    }


def _gate_targets(targets: dict[str, Any], case: str) -> dict[str, np.ndarray]:
    junction = np.asarray(targets["junction_map"], dtype=np.float32)
    if junction.any():
        soft = gaussian_filter(junction, sigma=GATE_SIGMA_PX, mode="constant")
        soft /= max(float(soft.max()), 1e-8)
    else:
        soft = np.zeros_like(junction)
    hard_negative = np.zeros_like(junction, dtype=bool)
    if case in HARD_GATE_NEGATIVE_CASES:
        hard_negative = np.asarray(targets["visible_fault_mask"], dtype=bool)
    valid = np.asarray(targets["visible_fault_mask"], dtype=bool) | (soft > 1e-3) | hard_negative
    # Background texture is explicitly sampled as negative as well; otherwise
    # the direct gate could remain high outside annotated structure.
    valid |= np.indices(valid.shape).sum(axis=0) % 8 == 0
    return {
        "gate_target": soft.astype(np.float32),
        "gate_valid_mask": valid,
        "gate_hard_negative_mask": hard_negative,
    }


def generate_sample_v3(
    split: str,
    index: int,
    *,
    image_size: int = 128,
) -> dict[str, Any]:
    if split == "test":
        raise PermissionError("CrossingTraceBench-v3 test is LOCKED_UNOPENED")
    seed = sample_seed_v3(split, index)
    case, pair_id = _scheduled_case(int(index))
    geometry_rng = np.random.default_rng(seed)
    geometry = scale_geometry(generate_geometry(case, geometry_rng), image_size)
    targets = rasterize_targets(geometry, image_size)
    tangent_targets = tangent_set_targets(geometry, targets, image_size)
    render_rng = np.random.default_rng(seed + 17_000_000)
    return {
        "image": render_seismic(geometry, image_size, render_rng),
        **targets,
        **tangent_targets,
        **_gate_targets(targets, case),
        "case": case,
        "split": split,
        "index": int(index),
        "seed": int(seed),
        "pair_id": pair_id,
        "gap_match": _gap_match_metadata(geometry, pair_id, seed),
        "image_size": int(image_size),
        "benchmark_version": "crossing_trace_bench_v3",
        "scientific_scope": "frozen context-repair development stream; test locked",
    }


class CrossingTraceBenchV3(Dataset[dict[str, Any]]):
    def __init__(self, split: str, *, image_size: int = 128) -> None:
        if split == "test":
            raise PermissionError("CrossingTraceBench-v3 test is LOCKED_UNOPENED")
        if split not in SPLIT_SIZES_V3:
            raise ValueError(f"Unknown CrossingTraceBench-v3 split: {split}")
        self.split = split
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return SPLIT_SIZES_V3[self.split]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = generate_sample_v3(self.split, int(index), image_size=self.image_size)
        for key in (
            "image",
            "visible_fault_mask",
            "latent_fault_mask",
            "positive_gap_mask",
            "negative_gap_mask",
            "branch_masks",
            "branch_centerlines",
            "junction_map",
            "continuation_relation_matrix",
            "continuation_eligible_matrix",
            "gt_theta_set",
            "gt_theta_valid",
            "gt_branch_theta",
            "gt_branch_theta_valid",
            "gt_mode_count",
            "gate_target",
            "gate_valid_mask",
            "gate_hard_negative_mask",
        ):
            value = torch.from_numpy(np.asarray(sample[key]))
            if key == "gt_mode_count":
                dtype = torch.int64
            elif key in {
                "gt_theta_valid",
                "gt_branch_theta_valid",
                "continuation_relation_matrix",
                "continuation_eligible_matrix",
                "gate_valid_mask",
                "gate_hard_negative_mask",
            }:
                dtype = torch.bool
            else:
                dtype = torch.float32
            if value.ndim == 2 and key not in {
                "gt_mode_count",
                "continuation_relation_matrix",
                "continuation_eligible_matrix",
            }:
                value = value.unsqueeze(0)
            sample[key] = value.to(dtype=dtype)
        return sample


def freeze_benchmark_v3_config(path: Path) -> dict[str, Any]:
    payload = benchmark_v3_config()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("CrossingTraceBench-v3 config drift after freeze")
    path.write_text(encoded)
    return payload
