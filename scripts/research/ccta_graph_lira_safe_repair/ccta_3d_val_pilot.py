#!/usr/bin/env python3
"""Five-patient frozen CT28 CCTA development-validation full-mask pilot.

Validation is NOT the six-patient held-out test, and it is not independent of
the already selected scene-level K10/15 model. This code opens only five
predeclared CT28 val scans after a separately persisted immutable protocol.

Reference mask is used only to make a controlled damage and to score final
masks post hoc; relation, CT paths and mask inference see ONLY damaged input.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import traceback

import nibabel as nib
import numpy as np

from ccta_3d_cli import run_cli
from ccta_3d_evaluate import evaluate_files
from ccta_3d_train_gap_simulator import (
    VAL_CT28_IDS,synthesize_validation_gaps,
)
from ccta_3d_train_pilot import (
    REFERENCE_DIR,SELECTED_PACK,_reference_mask,digest,
)
from ccta_3d_val_export import export_validation_ct
from ccta_3d_val_freeze import verify_validation_freeze
from ccta_3d_export_training_ct import ARCHIVE
from ccta_3d_cli import DEFAULT_PATH


ROOT=Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT=ROOT/"varvara/anza_lira_local_results/ccta_3d_validation"


def run_validation_pilot(
    *,
    patient:int,
    freeze_path:Path,
    output_dir:Path,
    model_pack:Path=SELECTED_PACK,
    path_protocol:Path=DEFAULT_PATH,
    reference_mask:Path|None=None,
    archive_dir:Path=ARCHIVE,
)->dict:
    if patient not in VAL_CT28_IDS:
        raise ValueError("Only five predeclared development-validation CT28 patients allowed")
    if output_dir.exists():
        raise FileExistsError("Immutable validation pilot output already exists")
    frozen=verify_validation_freeze(
        freeze_path,model=model_pack,path_protocol=path_protocol
    )
    if patient not in frozen["patients"]:
        raise ValueError("Patient not present in frozen validation protocol")
    reference_mask=reference_mask or REFERENCE_DIR/f"{patient}.coronary.nii.gz"
    if not reference_mask.is_file():
        raise FileNotFoundError(f"Validation GT coronary mask missing: {reference_mask}")
    settings=frozen["settings"]
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(
        prefix=f".val_{patient}_staging_",dir=output_dir.parent
    ))
    try:
        ct_path=staging/"ct.nii.gz"
        ct_report=export_validation_ct(
            patient,archive_dir,ct_path,staging/"ct_export.json"
        )
        if ct_report["raw_ct_sha256"]!=frozen["expected_validation_CT_SHA256"][str(patient)]:
            raise ValueError("CT archive sha differs from predeclared validation inventory")
        refmask,affine,ref_img=_reference_mask(reference_mask,ct_path)
        damaged,gaps=synthesize_validation_gaps(
            refmask,affine,patient_id=patient,
            gaps=settings["controlled_gaps"],
            radius_mm=settings["gap_radius_mm"],
            seed=settings["gap_seed"],
        )
        broken=staging/"broken_mask.nii.gz"
        image=nib.Nifti1Image(
            damaged.astype(np.uint8),affine,header=ref_img.header.copy()
        )
        image.header.set_data_dtype(np.uint8)
        nib.save(image,str(broken))
        (staging/"gaps.json").write_text(json.dumps(gaps,indent=2)+"\n")
        candidate_preview=run_cli(
            ct_path=ct_path,mask_path=broken,patient_id=str(patient),
            mode="candidates",model_pack=None,
            output_mask=staging/"inspection_unchanged.nii.gz",
            report_path=staging/"candidates.json",
            radius_mm=settings["repair_radius_mm"],
            max_new_voxels=settings["max_new_voxels_per_decision"],
            min_component_voxels=settings["min_component_voxels"],
            trace_mm=settings["trace_mm"],
            max_endpoints=settings["max_endpoints"],
            max_scenes=settings["max_scenes"],
            threshold=.5,path_protocol=path_protocol,
            strict_count_support=settings["strict_count_support"],
        )
        predictions=run_cli(
            ct_path=ct_path,mask_path=broken,patient_id=str(patient),
            mode="k10",model_pack=model_pack,
            output_mask=staging/"repaired.nii.gz",
            output_additions=staging/"added_voxels.nii.gz",
            report_path=staging/"decisions.json",
            radius_mm=settings["repair_radius_mm"],
            max_new_voxels=settings["max_new_voxels_per_decision"],
            min_component_voxels=settings["min_component_voxels"],
            trace_mm=settings["trace_mm"],
            max_endpoints=settings["max_endpoints"],
            max_scenes=settings["max_scenes"],
            threshold=.5,path_protocol=path_protocol,
            strict_count_support=settings["strict_count_support"],
        )
        results=evaluate_files(
            patient_id=str(patient),before=broken,
            after=staging/"repaired.nii.gz",
            reference=reference_mask,
            output=staging/"posthoc_metrics.json",
        )
        report={
            "status":"complete_frozen_ct28_development_validation_pilot",
            "protocol_version":frozen["version"],
            "protocol_sha256":digest(freeze_path),
            "patient_id":patient,
            "split":"val",
            "ct_sha256":ct_report["raw_ct_sha256"],
            "original_reference_sha256":digest(reference_mask),
            "frozen_selected_model_sha256":frozen["selected_K10_SHA256"],
            "frozen_path_protocol_sha256":frozen["selected_path_protocol_sha256"],
            "ground_truth_used_for_inference":False,
            "mask_damage_controlled":True,
            "inference_candidate_scene_count":candidate_preview["scene_count"],
            "inference_accepted_repairs":predictions["accepted_decisions"],
            "mask_added_voxels":predictions["added_voxels"],
            "dice_delta":results["dice_delta"],
            "cldice_delta":results["cldice_delta"],
            "false_GT_component_bridge_pairs":results.get("gt_inconsistent_new_component_pairs"),
            "held_out_test_accessed":False,
            "methodology_warning":"Same five validation subjects were used previously for K10/15 selection; descriptive only.",
        }
        (staging/"pilot_manifest.json").write_text(
            json.dumps(report,ensure_ascii=False,indent=2)+"\n",
            encoding="utf-8",
        )
    except Exception as exc:
        (staging/"pilot_failure.json").write_text(
            json.dumps({
                "status":"failed_frozen_ct28_development_validation_pilot",
                "patient_id":patient,
                "protocol_sha256":digest(freeze_path),
                "error_type":type(exc).__name__,
                "error":str(exc),
                "traceback":traceback.format_exc(limit=8),
                "retained_files":sorted(p.name for p in staging.iterdir()),
            },ensure_ascii=False,indent=2)+"\n",
            encoding="utf-8",
        )
        os.rename(staging,output_dir)
        raise
    os.rename(staging,output_dir)
    for file in output_dir.glob("*.json"):
        original=file.read_text(encoding="utf-8")
        rewritten=original.replace(str(staging),str(output_dir))
        if rewritten!=original:
            file.write_text(rewritten,encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-id",type=int,required=True)
    parser.add_argument("--freeze",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--model-pack",type=Path,default=SELECTED_PACK)
    parser.add_argument("--path-protocol",type=Path,default=DEFAULT_PATH)
    parser.add_argument("--archive-dir",type=Path,default=ARCHIVE)
    parser.add_argument("--reference-mask",type=Path)
    args=parser.parse_args()
    out=args.output_dir or DEFAULT_OUTPUT/f"patient_{args.patient_id}"
    result=run_validation_pilot(
        patient=args.patient_id,freeze_path=args.freeze,output_dir=out,
        model_pack=args.model_pack,path_protocol=args.path_protocol,
        archive_dir=args.archive_dir,reference_mask=args.reference_mask,
    )
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
