#!/usr/bin/env python3
"""Post-hoc 3D CCTA NIfTI evaluation. Ground truth NEVER enters inference.

Compute before/after Dice/clDice, false-added voxels and component counts for
train or validation only. Held-out CT28 test requires a separate frozen,
pre-registered evaluation tool and is intentionally blocked here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from ct28_mask_metrics import evaluate


HELD_OUT_CT28 = {"954", "958", "972", "973", "980", "984"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_files(
    *,
    patient_id: str,
    before: Path,
    after: Path,
    reference: Path,
    output: Path,
) -> dict:
    if str(patient_id) in HELD_OUT_CT28:
        raise ValueError("Held-out Graph-LIRA test is blocked before freezing protocol")
    source = {path.resolve() for path in (before, after, reference)}
    if output.resolve() in source or output.exists() or output.suffix != ".json":
        raise ValueError("Do not overwrite inputs, results, or use non-JSON output")
    images = [nib.load(str(path)) for path in (before, after, reference)]
    if len({tuple(img.shape) for img in images}) != 1 or images[0].ndim != 3:
        raise ValueError("Before/after/reference must share one 3D grid")
    if any(not np.allclose(img.affine, images[0].affine, atol=1e-4, rtol=0)
           for img in images[1:]):
        raise ValueError("Mask affine mismatch: register before evaluation")
    masks = []
    for image in images:
        mask = np.asanyarray(image.dataobj)
        if not np.isfinite(mask).all():
            raise ValueError("Mask contains nonfinite values")
        masks.append(np.asarray(mask > 0.5, dtype=bool))
    stats = evaluate(*masks, images[0].affine)
    stats.update({
        "scope": "posthoc_train_validation_only",
        "patient_id": str(patient_id),
        "truth_used_for_inference": False,
        "before_sha256": file_hash(before),
        "after_sha256": file_hash(after),
        "reference_sha256": file_hash(reference),
        "dice_delta": stats["dice_after"] - stats["dice_before"],
        "cldice_delta": stats["cldice_after"] - stats["cldice_before"],
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_files(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
