#!/usr/bin/env python3
"""Export SHA-verified CT28 VALIDATION CCTA volumes, with held-out firewall.

Train export remains separate and unchanged. This module will only extract
five previously declared development-validation ImageCAS scans; it never opens
six held-out Graph-LIRA test scans.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ccta_3d_train_gap_simulator import VAL_CT28_IDS
from ccta_3d_export_training_ct import ARCHIVE
from extract_ct28_scene_candidate_features import (
    EXPECTED_ALIGNMENT,EXPECTED_GEOMETRY,SplitZipReader,
    nifti_from_gz_bytes,expected_affine,
)


def export_validation_ct(
    scan_id:int,archive_dir:Path,out:Path,report_path:Path,
)->dict:
    if scan_id not in VAL_CT28_IDS:
        raise ValueError("Only five CT28 development-validation CCTA scans allowed")
    if out.exists() or report_path.exists():
        raise FileExistsError("Validation CT output is immutable")
    if not out.name.endswith(".nii.gz") or report_path.suffix!=".json":
        raise ValueError("Expected .nii.gz and JSON output extensions")
    if out.resolve()==report_path.resolve():
        raise ValueError("Image and report output must be distinct")
    reader=SplitZipReader(
        archive_dir/"801-1000.z04",archive_dir/"801-1000.zip"
    )
    raw,_=reader.extract_entry_bytes(f"801-1000/{scan_id}.img.nii.gz")
    geometry=pd.read_csv(EXPECTED_GEOMETRY).set_index("scan_id").loc[scan_id]
    alignment=pd.read_csv(EXPECTED_ALIGNMENT).set_index("scan_id").loc[scan_id]
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=str(alignment["ct_sha256"]):
        raise ValueError("CT28 validation CCTA archive SHA mismatch")
    meta,_,_=nifti_from_gz_bytes(raw)
    expected_shape=tuple(int(geometry[f"shape_{axis}"]) for axis in "xyz")
    expected_spacing=tuple(float(geometry[f"spacing_{axis}"]) for axis in "xyz")
    if (
        tuple(meta["shape"])!=expected_shape or
        not np.allclose(meta["spacing"],expected_spacing,atol=1e-7,rtol=0) or
        not np.allclose(meta["affine"],expected_affine(geometry),atol=1e-6,rtol=0)
    ):
        raise ValueError("Frozen CT28 validation geometry/affine mismatch")
    out.parent.mkdir(parents=True,exist_ok=True)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    out.write_bytes(raw)
    report={
        "status":"ct28_validation_only_export_verified",
        "scan_id":scan_id,
        "source":"verified_ImageCAS_CCTA_train_val_archive",
        "raw_ct_sha256":digest,
        "shape":list(expected_shape),
        "spacing_mm":list(expected_spacing),
        "output":str(out.resolve()),
        "ground_truth_mask_exported":False,
        "held_out_test_accessed":False,
    }
    report_path.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-id",type=int,required=True)
    parser.add_argument("--archive-dir",type=Path,default=ARCHIVE)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export_validation_ct(
        args.scan_id,args.archive_dir,args.output,args.report,
    ),indent=2))


if __name__=="__main__":
    main()
