#!/usr/bin/env python3
"""Generate paired medical-CT attribution controls from *existing* 3D patches.

A scientifically necessary negative control: CT-only, endpoint-only, or
spatially scrambled CT. No relation labels enter the transformed input.
Original CT28 source cache and medical files remain unchanged; all derived
cache files are create-only with SHA provenance.

These are separate candidate-level ablation datasets; never silently load
them as the full-image model for 3D clinical repair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np

from ccta_3d_an_patch_cache import DEFAULT_CACHE
from extract_ct28_scene_candidate_features import EXPECTED_PATIENT_IDS


MODES=("endpoint_only","ct_only","scrambled_ct")
ALL_IDS=EXPECTED_PATIENT_IDS["train"]|EXPECTED_PATIENT_IDS["val"]


def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda:source.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def _input_cache(path:Path,patient:int)->tuple[dict,dict]:
    receipt=json.loads((path/f"patches_{patient}.json").read_text(encoding="utf-8"))
    source=path/f"patches_{patient}.npz"
    if (receipt.get("patient_id")!=patient or
        receipt.get("split")!=("train" if patient in EXPECTED_PATIENT_IDS["train"] else "val") or
        receipt.get("cache_sha256")!=digest(source) or
        receipt.get("model_input_contains_ground_truth") is not False or
        receipt.get("input_control_mode","full_ct")!="full_ct"):
        raise ValueError("Expected immutable full-CCTA original patch cache")
    with np.load(source,allow_pickle=False) as data:
        payload={key:data[key] for key in data.files}
    if not {"patches","y","candidate_type","scene_id","candidate_id"}<=set(payload):
        raise ValueError("Missing frozen 3D patch/cache label schema")
    if payload["patches"].ndim!=5 or payload["patches"].shape[1]!=2:
        raise ValueError("Expected 2-channel NCDHW candidate patches")
    return receipt,payload


def make_control_cache(
    *,
    source_dir:Path,
    output_dir:Path,
    mode:str,
    patients:list[int]|None=None,
    seed:int=20261102,
)->dict:
    if mode not in MODES:
        raise ValueError("Unsupported image reliance negative-control variant")
    selected=sorted(ALL_IDS) if patients is None else patients
    if not selected or len(set(selected))!=len(selected) or not set(selected)<=ALL_IDS:
        raise ValueError("Only unique 17 train / 5 validation CT28 patients allowed")
    if output_dir.exists():
        raise FileExistsError("Negative-control dataset already exists")
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=".ct28_control_",dir=output_dir.parent))
    manifests=[]
    try:
        for patient in selected:
            source,payload=_input_cache(source_dir,patient)
            patches=payload["patches"].copy()
            if not np.isfinite(patches).all():
                raise ValueError("Nonfinite source CT candidate patch")
            if mode=="endpoint_only":
                patches[:,0]=0
            elif mode=="ct_only":
                patches[:,1]=0
            else:
                # Shuffle CT spatial arrangement inside *each candidate*.
                # Preserves its intensity histogram and endpoint channel;
                # destroys voxel-level correspondence between CT and anatomy.
                rng=np.random.default_rng(seed+int(patient))
                for index in range(len(patches)):
                    rng.shuffle(patches[index,0].flat)
            payload["patches"]=patches
            destination=staging/f"patches_{patient}.npz"
            np.savez_compressed(destination,**payload)
            receipt={
                **source,
                "cache_sha256":digest(destination),
                "input_control_mode":mode,
                "source_original_cache_sha256":source["cache_sha256"],
                "negative_control_seed":seed,
                "model_input_contains_ground_truth":False,
                "derived_from":"frozen_CT28_original_local_CCTA_candidate_patch_cache",
                "warnings":[
                    "Control is for candidate-level attribution only.",
                    "Cannot use derived patches as normal full CT model in Graph-LIRA inference.",
                ],
            }
            (staging/f"patches_{patient}.json").write_text(
                json.dumps(receipt,indent=2)+"\n",encoding="utf-8"
            )
            manifests.append({
                "patient_id":patient,
                "split":receipt["split"],
                "source_sha256":source["cache_sha256"],
                "control_sha256":receipt["cache_sha256"],
            })
        summary={
            "mode":mode,
            "seed":seed,
            "patient_count":len(selected),
            "train_subjects":sum(p in EXPECTED_PATIENT_IDS["train"] for p in selected),
            "validation_subjects":sum(p in EXPECTED_PATIENT_IDS["val"] for p in selected),
            "patients":manifests,
            "held_out_test_accessed":False,
            "purpose":"candidate-level negative control of actual CCTA image evidence",
        }
        (staging/"control_manifest.json").write_text(
            json.dumps(summary,indent=2)+"\n",encoding="utf-8"
        )
        os.rename(staging,output_dir)
        return summary
    except Exception:
        shutil.rmtree(staging)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--mode",choices=MODES,required=True)
    parser.add_argument("--patients",type=int,nargs="+")
    parser.add_argument("--seed",type=int,default=20261102)
    args=parser.parse_args()
    result=make_control_cache(
        source_dir=args.source_cache,output_dir=args.output_dir,
        mode=args.mode,patients=args.patients,seed=args.seed,
    )
    print(json.dumps({key:result[key] for key in (
        "mode","patient_count","train_subjects","validation_subjects",
        "held_out_test_accessed",
    )},indent=2))


if __name__=="__main__":
    main()
