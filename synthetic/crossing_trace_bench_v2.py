"""Independent synthetic stream with exact local axial tangent-set targets."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch.utils.data import Dataset

from .geometry_generator import GEOMETRY_TYPES, GeometrySample, generate_geometry, scale_geometry
from .instance_targets import rasterize_targets
from .seismic_background import render_seismic


SPLIT_SEED_BASE_V2 = {"train": 110_000_000, "validation": 120_000_000, "test": 130_000_000}
SPLIT_SIZES_V2 = {"train": 10_000, "validation": 2_000, "test": 2_000}
MAX_GT_MODES = 4
TANGENT_DEDUP_TOLERANCE_RAD = math.radians(7.5)


def benchmark_v2_config() -> dict[str, Any]:
    payload = {
        "version": "crossing_trace_bench_v2",
        "split_seed_base": SPLIT_SEED_BASE_V2,
        "split_sizes": SPLIT_SIZES_V2,
        "geometry_types": list(GEOMETRY_TYPES),
        "max_gt_modes": MAX_GT_MODES,
        "tangent_dedup_tolerance_rad": TANGENT_DEDUP_TOLERANCE_RAD,
        "old_test_stream": "IMMUTABLE_AND_NOT_REUSED",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def sample_seed_v2(split: str, index: int) -> int:
    if split not in SPLIT_SEED_BASE_V2:
        raise ValueError(f"Unknown CrossingTraceBench-v2 split: {split}")
    if not 0 <= int(index) < SPLIT_SIZES_V2[split]:
        raise IndexError(index)
    return SPLIT_SEED_BASE_V2[split] + int(index)


def _axial_distance_scalar(first: float, second: float) -> float:
    delta = first - second
    return abs(math.atan2(math.sin(delta), abs(math.cos(delta))))


def _axial_mean(angles: list[float]) -> float:
    if not angles:
        raise ValueError("axial mean requires at least one angle")
    cosine = sum(math.cos(2.0 * value) for value in angles)
    sine = sum(math.sin(2.0 * value) for value in angles)
    return float((0.5 * math.atan2(sine, cosine)) % math.pi)


def _unique_axial(angles: list[float]) -> list[float]:
    result: list[float] = []
    for angle in angles:
        normalized = float(angle % math.pi)
        if all(_axial_distance_scalar(normalized, existing) > TANGENT_DEDUP_TOLERANCE_RAD for existing in result):
            result.append(normalized)
    return result


def _nearest_branch_theta(targets: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    centerlines = np.asarray(targets["branch_centerlines"], dtype=bool)
    branch_masks = np.asarray(targets["branch_masks"], dtype=bool)
    cos2 = np.asarray(targets["branch_tangent_cos2"], dtype=np.float32)
    sin2 = np.asarray(targets["branch_tangent_sin2"], dtype=np.float32)
    fields = np.zeros_like(cos2)
    valid = branch_masks.copy()
    for index, centerline in enumerate(centerlines):
        if not centerline.any():
            continue
        nearest = distance_transform_edt(~centerline, return_distances=False, return_indices=True)
        nearest_cos2 = cos2[index][tuple(nearest)]
        nearest_sin2 = sin2[index][tuple(nearest)]
        fields[index] = np.mod(0.5 * np.arctan2(nearest_sin2, nearest_cos2), math.pi)
    return fields, valid


def tangent_set_targets(
    geometry: GeometrySample,
    targets: dict[str, Any],
    image_size: int,
) -> dict[str, np.ndarray]:
    branch_theta, branch_valid = _nearest_branch_theta(targets)
    branch_masks = np.asarray(targets["branch_masks"], dtype=bool)
    height = width = int(image_size)
    theta_set = np.zeros((MAX_GT_MODES, height, width), dtype=np.float32)
    theta_valid = np.zeros_like(theta_set, dtype=bool)
    mode_count = np.zeros((height, width), dtype=np.uint8)

    structural_y, structural_x = np.nonzero(branch_masks.any(axis=0))
    for y, x in zip(structural_y.tolist(), structural_x.tolist()):
        active = np.flatnonzero(branch_masks[:, y, x])
        angles = _unique_axial([float(branch_theta[index, y, x]) for index in active])
        if len(angles) > MAX_GT_MODES:
            raise ValueError("generated tangent set exceeds MAX_GT_MODES")
        mode_count[y, x] = len(angles)
        theta_set[: len(angles), y, x] = angles
        theta_valid[: len(angles), y, x] = True

    branch_index = {int(value): index for index, value in enumerate(targets["branch_ids"])}
    for junction in geometry.junctions:
        center_x = int(round(float(junction.point_xy[0])))
        center_y = int(round(float(junction.point_xy[1])))
        incident = [branch_index[int(value)] for value in junction.incident_branch_ids]
        incident_angles = {
            int(branch_id): float(branch_theta[index, center_y, center_x])
            for branch_id, index in zip(junction.incident_branch_ids, incident)
        }
        if junction.junction_type == "y_branch":
            axes = _unique_axial([incident_angles[int(value)] for value in junction.incident_branch_ids])
        else:
            used: set[int] = set()
            axes = []
            for first, second in junction.continuation_relation:
                axes.append(_axial_mean([incident_angles[int(first)], incident_angles[int(second)]]))
                used.update((int(first), int(second)))
            axes.extend(
                incident_angles[int(value)]
                for value in junction.incident_branch_ids
                if int(value) not in used
            )
            axes = _unique_axial(axes)
        if len(axes) > MAX_GT_MODES:
            raise ValueError("junction tangent set exceeds MAX_GT_MODES")
        for y in range(max(0, center_y - 2), min(height, center_y + 3)):
            for x in range(max(0, center_x - 2), min(width, center_x + 3)):
                if (x - center_x) ** 2 + (y - center_y) ** 2 > 4:
                    continue
                mode_count[y, x] = len(axes)
                theta_set[:, y, x] = 0.0
                theta_valid[:, y, x] = False
                theta_set[: len(axes), y, x] = axes
                theta_valid[: len(axes), y, x] = True

    for gap in geometry.gaps:
        if gap.gap_type != "positive":
            continue
        delta = gap.points_xy[-1] - gap.points_xy[0]
        angle = float(math.atan2(float(delta[1]), float(delta[0])) % math.pi)
        selected = np.asarray(targets["positive_gap_mask"], dtype=bool) & (mode_count == 0)
        mode_count[selected] = 1
        theta_set[0, selected] = angle
        theta_valid[0, selected] = True

    if not np.array_equal(theta_valid.sum(axis=0).astype(np.uint8), mode_count):
        raise AssertionError("gt_mode_count must equal gt_theta_valid cardinality")
    return {
        "gt_mode_count": mode_count,
        "gt_theta_set": theta_set,
        "gt_theta_valid": theta_valid,
        "gt_branch_theta": branch_theta.astype(np.float32),
        "gt_branch_theta_valid": branch_valid,
    }


def generate_sample_v2(
    split: str,
    index: int,
    *,
    image_size: int = 128,
    case: str | None = None,
) -> dict[str, Any]:
    seed = sample_seed_v2(split, index)
    rng = np.random.default_rng(seed)
    selected_case = case or GEOMETRY_TYPES[int(rng.integers(0, len(GEOMETRY_TYPES)))]
    geometry = scale_geometry(generate_geometry(selected_case, rng), image_size)
    targets = rasterize_targets(geometry, image_size)
    tangent_targets = tangent_set_targets(geometry, targets, image_size)
    return {
        "image": render_seismic(geometry, image_size, rng),
        **targets,
        **tangent_targets,
        "case": selected_case,
        "split": split,
        "index": int(index),
        "seed": seed,
        "image_size": int(image_size),
        "benchmark_version": "crossing_trace_bench_v2",
        "scientific_scope": "controlled mechanism development; independent from frozen opened test",
    }


class CrossingTraceBenchV2(Dataset[dict[str, Any]]):
    def __init__(self, split: str, *, image_size: int = 128, length: int | None = None) -> None:
        if split not in SPLIT_SIZES_V2:
            raise ValueError(f"Unknown CrossingTraceBench-v2 split: {split}")
        self.split = split
        self.image_size = int(image_size)
        self.length = SPLIT_SIZES_V2[split] if length is None else int(length)
        if not 0 < self.length <= SPLIT_SIZES_V2[split]:
            raise ValueError("Requested length must be within the frozen v2 split size")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = generate_sample_v2(self.split, int(index), image_size=self.image_size)
        for key in (
            "image",
            "visible_fault_mask",
            "latent_fault_mask",
            "gap_mask",
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
        ):
            value = torch.from_numpy(np.asarray(sample[key]))
            if key in {"gt_mode_count"}:
                dtype = torch.int64
            elif key in {"gt_theta_valid", "gt_branch_theta_valid", "continuation_relation_matrix", "continuation_eligible_matrix"}:
                dtype = torch.bool
            else:
                dtype = torch.float32
            if value.ndim == 2 and key not in {"gt_mode_count", "continuation_relation_matrix", "continuation_eligible_matrix"}:
                value = value.unsqueeze(0)
            sample[key] = value.to(dtype=dtype)
        return sample


def freeze_benchmark_v2_config(path: Path) -> dict[str, Any]:
    payload = benchmark_v2_config()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("CrossingTraceBench-v2 config drift after freeze")
    path.write_text(encoded)
    return payload
