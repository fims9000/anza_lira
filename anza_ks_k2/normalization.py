"""Train-only frozen feature normalization for K2 controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .benchmark import generate_sample
from .dense_features import METHODS, dense_orientation_features


NORMALIZATION_INDICES = tuple(range(256))


def compute_feature_norm(*, device: str = "cuda") -> dict[str, object]:
    maps = []
    for index in NORMALIZATION_INDICES:
        image = torch.from_numpy(generate_sample("train", index)["image"][:1])
        maps.append(F.interpolate(image[None], size=(24, 24), mode="bilinear", align_corners=False)[0])
    structural_maps = torch.stack(maps)
    result: dict[str, object] = {"split": "train", "indices": [0, len(NORMALIZATION_INDICES) - 1], "sample_count": len(NORMALIZATION_INDICES), "methods": {}}
    for method in METHODS:
        total = torch.zeros(104, dtype=torch.float64, device=device)
        total_square = torch.zeros_like(total)
        count = 0
        for start in range(0, len(structural_maps), 4):
            batch = structural_maps[start : start + 4].to(device)
            with torch.inference_mode():
                values = dense_orientation_features(batch, method).reshape(-1, 104).to(torch.float64)
            total += values.sum(dim=0); total_square += (values * values).sum(dim=0); count += len(values)
        mean = total / count
        variance = (total_square / count - mean.square()).clamp_min(0.0)
        std = torch.sqrt(variance)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        result["methods"][method] = {"mean": mean.cpu().tolist(), "std": std.cpu().tolist(), "observation_count": count}
    return result


def save_feature_norm(path: Path, *, device: str = "cuda") -> dict[str, object]:
    result = compute_feature_norm(device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
