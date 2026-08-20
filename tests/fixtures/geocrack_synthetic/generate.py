"""Generate tiny controlled crack-like images strictly for pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


CASE_NAMES = (
    "straight",
    "diagonal",
    "two_independent",
    "t_junction",
    "x_junction",
    "gap",
    "low_contrast",
    "border",
    "empty",
)


def _segments(case_name: str, *, offset: int = 0) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    center = 112 + offset
    cases = {
        "straight": [((28, center), (196, center))],
        "diagonal": [((34, 34 + offset), (190, 190 + offset))],
        "two_independent": [((30, 72 + offset), (194, 72 + offset)), ((30, 152 + offset), (194, 152 + offset))],
        "t_junction": [((112, 36), (112, 188)), ((48, 72), (176, 72))],
        "x_junction": [((40, 40), (184, 184)), ((184, 40), (40, 184))],
        "gap": [((30, center), (106, center)), ((112, center), (194, center))],
        "low_contrast": [((28, center), (196, center))],
        "border": [((0, center), (170, center))],
        "empty": [],
    }
    return cases[case_name]


def generate_synthetic_geocrack(
    root: str | Path,
    *,
    variants_per_case: int = 1,
    size: int = 224,
) -> dict[str, Any]:
    """Write deterministic images/masks and exact case metadata under ``root``."""
    if size != 224:
        raise ValueError("GeoCrack integration fixture must use 224x224 patches")
    if variants_per_case < 1:
        raise ValueError("variants_per_case must be positive")
    root = Path(root)
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    for case_index, case_name in enumerate(CASE_NAMES):
        for variant in range(variants_per_case):
            source_id = f"synthetic_{case_name}_{variant:02d}"
            image = Image.fromarray(np.full((size, size, 3), 70, dtype=np.uint8), mode="RGB")
            mask = Image.new("L", (size, size), color=0)
            image_draw = ImageDraw.Draw(image)
            mask_draw = ImageDraw.Draw(mask)
            offset = variant - variants_per_case // 2
            intensity = 82 if case_name == "low_contrast" else 205
            for start, end in _segments(case_name, offset=offset):
                image_draw.line((start, end), fill=(intensity,) * 3, width=3)
                mask_draw.line((start, end), fill=255, width=1)
            image_name = f"{source_id}_original_patch0.png"
            mask_name = f"{source_id}_binarymask_patch0.png"
            image.save(images_dir / image_name)
            mask.save(masks_dir / mask_name)
            samples.append(
                {
                    "case": case_name,
                    "source_image_id": source_id,
                    "image_path": f"images/{image_name}",
                    "mask_path": f"masks/{mask_name}",
                    "segments": _segments(case_name, offset=offset),
                    "synthetic_only": True,
                }
            )
    manifest = {
        "name": "geocrack_synthetic_pipeline_fixture",
        "scientific_result": False,
        "warning": "TEST FIXTURE ONLY; NEVER MIX WITH GEOCRACK STUDY RESULTS",
        "case_names": list(CASE_NAMES),
        "sample_count": len(samples),
        "samples": samples,
    }
    (root / "fixture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
