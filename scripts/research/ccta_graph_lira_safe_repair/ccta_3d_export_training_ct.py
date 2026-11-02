#!/usr/bin/env python3
"""Export one SHA-verified *train-only* CT28 CCTA NIfTI from the local split ZIP.

No full-mask ground truth and no held-out patients are opened. Do not use this
helper to open test patients before the frozen experiment is authorized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ccta_3d_train_gap_simulator import TRAIN_CT28_IDS
from extract_ct28_scene_candidate_features import (
    EXPECTED_ALIGNMENT, EXPECTED_GEOMETRY,
    SplitZipReader, nifti_from_gz_bytes, expected_affine,
)


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "varvara/anza_lira_local_data/imagecas_801_1000"


def export_training_ct(
    scan_id: int,
    archive_dir: Path,
    out: Path,
    report_path: Path,
) -> dict:
    if scan_id not in TRAIN_CT28_IDS:
        raise ValueError("Only CT28 training patients may be exported here")
    if out.exists() or report_path.exists():
        raise FileExistsError("Output already exists, refusing to overwrite")
    if out.name.split(".")[-2:] != ["nii", "gz"] or report_path.suffix != ".json":
        raise ValueError("Expected .nii.gz and .json output names")
    reader = SplitZipReader(
        archive_dir / "801-1000.z04",
        archive_dir / "801-1000.zip",
    )
    raw, _ = reader.extract_entry_bytes(f"801-1000/{scan_id}.img.nii.gz")
    geometry = pd.read_csv(EXPECTED_GEOMETRY).set_index("scan_id").loc[scan_id]
    alignment = pd.read_csv(EXPECTED_ALIGNMENT).set_index("scan_id").loc[scan_id]
    digest = hashlib.sha256(raw).hexdigest()
    if digest != str(alignment["ct_sha256"]):
        raise ValueError("CCTA archive raw-byte SHA mismatch")
    meta, _, _ = nifti_from_gz_bytes(raw)
    expected_shape = tuple(int(geometry[f"shape_{axis}"]) for axis in "xyz")
    expected_spacing = tuple(float(geometry[f"spacing_{axis}"]) for axis in "xyz")
    if tuple(meta["shape"]) != expected_shape or not np.allclose(
        meta["spacing"], expected_spacing, atol=1e-7, rtol=0
    ) or not np.allclose(meta["affine"], expected_affine(geometry), atol=1e-6, rtol=0):
        raise ValueError("CCTA training geometry/affine alignment differs from frozen CT28")
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    report = {
        "status": "ct28_train_only_export_verified",
        "scan_id": scan_id, "raw_ct_sha256": digest,
        "shape": list(expected_shape), "spacing_mm": list(expected_spacing),
        "output": str(out.resolve()),
        "ct_source": "imagecas_801_1000_frozen_archive",
        "ground_truth_mask_exported": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-id", type=int, required=True)
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_training_ct(args.scan_id, args.archive_dir,
                                        args.output, args.report), indent=2))


if __name__ == "__main__":
    main()
