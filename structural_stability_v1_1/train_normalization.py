"""Leakage-safe RGB normalization from SS_TRAIN image pixels only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from datasets.cracks import load_section_image
from structural_stability_v1_1.amendment import sha256_file
from structural_stability_v1_1.protocol import ROOT, canonical_hash, protocol_hash


def compute_train_only_normalization(section_ids: Sequence[int], output_path: Path) -> dict[str, Any]:
    ordered = [int(value) for value in section_ids]
    if len(ordered) != 220 or len(set(ordered)) != 220:
        raise ValueError("V1.1 normalization requires exactly 220 unique SS_TRAIN sections")
    source = Path(__file__)
    total = np.zeros(3, dtype=np.float64)
    squared = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    image_hashes: dict[str, str] = {}
    for section_id in ordered:
        path = ROOT / "data/cracks/images" / f"section_{section_id:03d}.png"
        image = np.asarray(load_section_image(path), dtype=np.float64)
        values = image.reshape(-1, 3)
        total += values.sum(axis=0)
        squared += np.square(values).sum(axis=0)
        pixel_count += values.shape[0]
        image_hashes[str(section_id)] = sha256_file(path)
    mean = total / pixel_count
    variance = np.maximum(squared / pixel_count - np.square(mean), 0.0)
    payload = {
        "status": "TRAIN_ONLY_NORMALIZATION_FROZEN",
        "protocol_sha256": protocol_hash(),
        "train_section_ids": ordered,
        "train_section_count": len(ordered),
        "section_list_sha256": canonical_hash(ordered),
        "pixel_count": int(pixel_count),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "code_path": source.relative_to(ROOT).as_posix(),
        "code_sha256": sha256_file(source),
        "image_sha256": image_hashes,
        "non_train_image_pixels_read": False,
        "expert_data_accessed": False,
    }
    payload["sha256"] = canonical_hash(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
