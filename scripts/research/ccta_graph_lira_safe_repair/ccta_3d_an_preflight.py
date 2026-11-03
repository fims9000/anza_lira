#!/usr/bin/env python3
"""Read-only local preflight for REAL CT28 train/dev-val 3D ANZA experiments.

No CCTA medical volume or mask bytes are opened. No files are downloaded,
no model/containers modified and six held-out CT28 test IDs are never probed.
For missing inputs print an actionable local path inventory instead of
starting an experiment that cannot finish.

The presence/size check is necessary, not sufficient for clinical provenance;
full export/volume stages subsequently perform SHA and affine verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from ccta_3d_an_patch_cache import SCENES
from ccta_3d_train_pilot import REFERENCE_DIR,SELECTED_PACK
from ccta_3d_export_training_ct import ARCHIVE
from ccta_3d_cli import DEFAULT_PATH
from extract_ct28_scene_candidate_features import EXPECTED_PATIENT_IDS
from ccta_3d_k10 import K10_15_SHA256


REQUIRED_CT28_FILES=(
    "scene_plan_train_val.csv",
    "scene_endpoints_train_val.csv",
    "scene_candidates_train_val.csv",
    "scene_ct_alignment_train_val.csv",
)


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def assess_inputs(
    *,
    scene_dir:Path=SCENES,
    archive_dir:Path=ARCHIVE,
    reference_dir:Path=REFERENCE_DIR,
    k10_pack:Path=SELECTED_PACK,
    path_protocol:Path=DEFAULT_PATH,
    require_development_val:bool=False,
    min_free_gb:float=5.,
    expected_k10_sha256:str=K10_15_SHA256,
)->dict:
    if not 0<=min_free_gb<=1000:
        raise ValueError("Unexpected disk free space threshold")
    train=sorted(EXPECTED_PATIENT_IDS["train"])
    val=sorted(EXPECTED_PATIENT_IDS["val"])
    problems=[]
    checks=[]
    def check(path:Path,what:str):
        available=path.is_file() and path.stat().st_size>0
        checks.append({
            "input":what,
            "path":str(path),
            "available_nonempty":available,
            "bytes":int(path.stat().st_size) if available else None,
        })
        if not available:
            problems.append(what)
        return available

    for name in REQUIRED_CT28_FILES:
        check(scene_dir/name,f"frozen_scene:{name}")
    for part in ("801-1000.z04","801-1000.zip"):
        check(archive_dir/part,f"ImageCAS_CT_archive:{part}")
    for pid in train+(val if require_development_val else []):
        check(
            reference_dir/f"{pid}.coronary.nii.gz",
            f"reference_mask_{'val' if pid in val else 'train'}:{pid}",
        )
    pack_valid=check(k10_pack,"selected_K10_15_local_model_pack")
    if pack_valid:
        digest=sha(k10_pack)
        if digest!=expected_k10_sha256:
            problems.append("selected_K10_model_SHA_mismatch")
    else:
        digest=None
    check(path_protocol,"selected_CT_path_v2_protocol")
    disk_root=archive_dir if archive_dir.exists() else scene_dir if scene_dir.exists() else Path.cwd()
    disk_check=shutil.disk_usage(str(disk_root))
    free_gb=float(disk_check.free)/(1024**3)
    if free_gb<min_free_gb:
        problems.append("not_enough_free_disk_space")
    return {
        "status":"all_required_local_sources_present" if not problems else "missing_or_invalid_local_inputs",
        "stage":"train_plus_development_val" if require_development_val else "train_only",
        "patient_count_train":len(train),
        "patient_count_val_required":len(val) if require_development_val else 0,
        "selected_model_expected_sha256":expected_k10_sha256,
        "selected_model_found_sha256":digest,
        "available_free_gb":round(free_gb,3),
        "disk_minimum_gb":min_free_gb,
        "checked_inputs":checks,
        "issues":problems,
        "held_out_test_accessed":False,
        "warnings":[
            "Preflight is read-only metadata; CT/mask affine+SHA verified during later export/processing.",
            "PyTorch/CUDA runtime availability and RAM/VRAM are checked by the training stage.",
            "Do not commit original medical CCTA volumes or local K10 weights to Git.",
        ],
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir",type=Path,default=SCENES)
    parser.add_argument("--archive-dir",type=Path,default=ARCHIVE)
    parser.add_argument("--reference-dir",type=Path,default=REFERENCE_DIR)
    parser.add_argument("--model-pack",type=Path,default=SELECTED_PACK)
    parser.add_argument("--path-protocol",type=Path,default=DEFAULT_PATH)
    parser.add_argument("--require-validation",action="store_true")
    parser.add_argument("--min-free-gb",type=float,default=5.)
    parser.add_argument("--receipt",type=Path)
    args=parser.parse_args()
    result=assess_inputs(
        scene_dir=args.scene_dir,archive_dir=args.archive_dir,
        reference_dir=args.reference_dir,k10_pack=args.model_pack,
        path_protocol=args.path_protocol,
        require_development_val=args.require_validation,
        min_free_gb=args.min_free_gb,
    )
    if args.receipt is not None:
        if args.receipt.exists():
            raise FileExistsError("Immutable preflight receipt already exists")
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(
            json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"
        )
    print(json.dumps({
        "status":result["status"],
        "stage":result["stage"],
        "issues":result["issues"],
        "free_gb":result["available_free_gb"],
        "held_out_test_accessed":False,
    },ensure_ascii=False,indent=2))
    if result["issues"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
