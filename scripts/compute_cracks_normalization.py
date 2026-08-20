#!/usr/bin/env python3
"""Compute CRACKS RGB normalization from Setting A training images only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.cracks import load_section_image
from scripts.audit_cracks_archives import DATA_ROOT, write_json
from scripts.prepare_cracks_protocol import V2_RESULT_ROOT, canonical_hash


DEFAULT_PROTOCOL = V2_RESULT_ROOT / "protocol.json"
DEFAULT_OUTPUT = V2_RESULT_ROOT / "cracks" / "crowd_target" / "normalization.json"


def compute_rgb_stats(image_root: Path, section_ids: Sequence[int]) -> dict[str, object]:
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for section_id in section_ids:
        image = load_section_image(image_root / f"section_{int(section_id):03d}.png").astype(np.float64)
        flat = image.reshape(-1, 3)
        total += flat.sum(axis=0)
        total_sq += np.square(flat).sum(axis=0)
        pixel_count += flat.shape[0]
    if pixel_count == 0:
        raise ValueError("Cannot compute normalization from zero training pixels")
    mean = total / pixel_count
    variance = np.maximum(total_sq / pixel_count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid CRACKS normalization statistics")
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "pixel_count": int(pixel_count),
        "section_count": len(section_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--image-root", type=Path, default=DATA_ROOT / "images")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    section_ids = protocol["setting_a"]["training_section_ids"]
    stats = compute_rgb_stats(args.image_root, section_ids)
    stats.update(
        {
            "status": "TRAIN_ONLY",
            "protocol_sha256": protocol["sha256"],
            "section_ids_sha256": canonical_hash(section_ids),
        }
    )
    write_json(args.output, stats)
    print("CRACKS NORMALIZATION: COMPLETE")
    print(f"SECTIONS: {stats['section_count']}")
    print(f"MEAN: {stats['mean']}")
    print(f"STD: {stats['std']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
