"""On-the-fly controlled benchmark for fault branch identity and continuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry_generator import GEOMETRY_TYPES, generate_geometry, scale_geometry
from .instance_targets import rasterize_targets
from .seismic_background import render_seismic


SPLIT_SEED_BASE = {"train": 10_000_000, "validation": 20_000_000, "test": 30_000_000}
SPLIT_SIZES = {"train": 10_000, "validation": 2_000, "test": 2_000}


def sample_seed(split: str, index: int) -> int:
    if split not in SPLIT_SEED_BASE:
        raise ValueError(f"Unknown CrossingTraceBench split: {split}")
    if not 0 <= int(index) < SPLIT_SIZES[split]:
        raise IndexError(index)
    return SPLIT_SEED_BASE[split] + int(index)


def generate_sample(
    split: str,
    index: int,
    *,
    image_size: int = 128,
    case: str | None = None,
) -> dict[str, Any]:
    seed = sample_seed(split, index)
    rng = np.random.default_rng(seed)
    selected_case = case or GEOMETRY_TYPES[int(rng.integers(0, len(GEOMETRY_TYPES)))]
    normalized = generate_geometry(selected_case, rng)
    geometry = scale_geometry(normalized, image_size)
    targets = rasterize_targets(geometry, image_size)
    image = render_seismic(geometry, image_size, rng)
    return {
        "image": image,
        **targets,
        "case": selected_case,
        "split": split,
        "index": int(index),
        "seed": seed,
        "image_size": int(image_size),
        "scientific_scope": "controlled structural benchmark; not an F3 physical simulator",
    }


class CrossingTraceBench(Dataset[dict[str, Any]]):
    def __init__(self, split: str, *, image_size: int = 128, length: int | None = None) -> None:
        if split not in SPLIT_SIZES:
            raise ValueError(f"Unknown CrossingTraceBench split: {split}")
        self.split = split
        self.image_size = int(image_size)
        self.length = SPLIT_SIZES[split] if length is None else int(length)
        if not 0 < self.length <= SPLIT_SIZES[split]:
            raise ValueError("Requested length must be within the frozen split size")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = generate_sample(self.split, int(index), image_size=self.image_size)
        tensor_keys = {
            "image": torch.float32,
            "visible_fault_mask": torch.float32,
            "latent_fault_mask": torch.float32,
            "gap_mask": torch.float32,
            "positive_gap_mask": torch.float32,
            "negative_gap_mask": torch.float32,
            "positive_gap_masks": torch.float32,
            "negative_gap_masks": torch.float32,
            "positive_gap_owner": torch.int64,
            "visible_centerline_map": torch.float32,
            "latent_centerline_map": torch.float32,
            "instance_visualization_map": torch.int64,
            "instance_masks": torch.float32,
            "instance_overlap_mask": torch.bool,
            "branch_visualization_map": torch.int64,
            "branch_masks": torch.float32,
            "branch_centerlines": torch.float32,
            "branch_tangent_cos2": torch.float32,
            "branch_tangent_sin2": torch.float32,
            "junction_map": torch.float32,
            "endpoint_map": torch.float32,
            "continuation_relation_matrix": torch.bool,
            "continuation_eligible_matrix": torch.bool,
        }
        scalar_maps = {
            "positive_gap_owner",
            "instance_visualization_map",
            "branch_visualization_map",
            "continuation_relation_matrix",
            "continuation_eligible_matrix",
        }
        for key, dtype in tensor_keys.items():
            value = torch.from_numpy(np.asarray(sample[key]))
            if value.ndim == 2 and key not in scalar_maps:
                value = value.unsqueeze(0)
            sample[key] = value.to(dtype=dtype)
        return sample
