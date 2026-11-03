#!/usr/bin/env python3
"""Check true CT28 3D candidate endpoint containment in fixed ANZA/CNN patches.

Read only committed train/val SCENE endpoints and frozen candidate identities.
No CT, mask, diagnosis, GROUND-TRUTH candidate targets or held-out patients.

For all 1,404 scenes, measure whether every candidate endpoint is actually
within the oriented physical 24mm patch, before and after the exact 15x
(0.8 mm position, 20 degree tangent) K10 perturbations. The learned candidate
model currently consumes Gaussian endpoint heatmaps; out-of-frame endpoints
are silent distribution shifts and must be reported rather than ignored.

This is geometric FOV coverage, NOT evidence that a sampled CT patch is inside
the medical image field of view, nor that vessel structure is correctly covered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from ccta_3d_an_patches import PatchConfig,local_frame
from ct28_perturbations import perturb_scene_endpoints
from ct28_cardinality_benchmark import load_frozen,TRAIN_IDS,VAL_IDS

SCENES=ROOT/"artifacts/varvara/ct28_graph_lira"
CANDIDATES=SCENES/"scene_candidates_train_val.csv"
ENDPOINTS=SCENES/"scene_endpoints_train_val.csv"
PLAN=SCENES/"scene_plan_train_val.csv"
SEED=20260920
REPS=15


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda:source.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def endpoint_margin_mm(points:np.ndarray,config:PatchConfig)->float:
    """Signed distance (mm) to nearest patch cube face, in candidate frame."""
    center,frame=local_frame(points)
    local=(points-center)@frame
    return float((config.side-1)*config.voxel_mm/2 - np.abs(local).max())


def run(
    *,
    plan_path:Path=PLAN,
    endpoint_path:Path=ENDPOINTS,
    candidate_path:Path=CANDIDATES,
    config:PatchConfig=PatchConfig(),
    perturbations:int=REPS,
    seed:int=SEED,
)->tuple[dict,pd.DataFrame]:
    if perturbations<1 or perturbations>50:
        raise ValueError("Require a prespecified 1..50 perturbation count")
    train,val=load_frozen(plan_path)
    plans=pd.concat([train,val],ignore_index=True)
    if len(plans)!=1404:
        raise ValueError("Frozen CT28 scene count changed")
    endpoints=pd.read_csv(endpoint_path)
    candidates=pd.read_csv(candidate_path,usecols=[
        "scene_id","candidate_id","candidate_type","endpoint_ids"
    ])
    if len(endpoints)!=7020 or len(candidates)!=15092:
        raise ValueError("Frozen endpoint or candidate count changed")
    lookup=plans.set_index("scene_id")[["scan_id","split"]]
    if not lookup.index.is_unique:
        raise ValueError("Duplicate scene plans")
    if not set(endpoints.scene_id)==set(plans.scene_id):
        raise ValueError("Endpoint scene coverage mismatch")
    if not set(candidates.scene_id)==set(plans.scene_id):
        raise ValueError("Candidate scene coverage mismatch")
    if endpoints[["scene_id","endpoint_id"]].duplicated().any():
        raise ValueError("Ambiguous physical endpoint")
    if candidates[["scene_id","candidate_id"]].duplicated().any():
        raise ValueError("Ambiguous candidate identity")
    if not set(candidates.candidate_type)=={"PAIR","JUNCTION"}:
        raise ValueError("Unknown candidate type")

    results=[]
    scene_candidates={sid:rows for sid,rows in candidates.groupby("scene_id",sort=False)}
    for scene_id,scene_endpoints in endpoints.groupby("scene_id",sort=False):
        study=lookup.loc[scene_id]
        split=str(study["split"])
        patient=int(study["scan_id"])
        if patient not in (TRAIN_IDS if split=="train" else VAL_IDS if split=="val" else set()):
            raise ValueError("Test/unknown patient in patch-geometry audit")
        local_candidates=scene_candidates[scene_id]
        for rep in range(-1,perturbations):
            current=scene_endpoints if rep<0 else perturb_scene_endpoints(
                scene_endpoints,seed=seed,repetition=rep,
                position_noise_mm=.8,tangent_noise_deg=20.
            )
            physical=current.set_index("endpoint_id")
            if not physical.index.is_unique:
                raise ValueError("Perturbed endpoint ID duplication")
            for row in local_candidates.itertuples(index=False):
                ids=str(row.endpoint_ids).split("|")
                if len(ids)!=(2 if row.candidate_type=="PAIR" else 3):
                    raise ValueError("Candidate degree inconsistent with frozen type")
                if any(s not in physical.index for s in ids):
                    raise ValueError("Candidate references a missing endpoint")
                pts=physical.loc[ids,["x_lps","y_lps","z_lps"]].to_numpy(dtype=float)
                margin=endpoint_margin_mm(pts,config)
                results.append({
                    "scene_id":scene_id,
                    "scan_id":patient,
                    "split":split,
                    "candidate_type":row.candidate_type,
                    "repeat":rep,
                    "patch_margin_mm":margin,
                    "all_endpoints_inside_patch":int(margin>=0),
                    "near_patch_edge_within_1mm":int(0<=margin<1.),
                })
    result=pd.DataFrame(results)
    if len(result)!=15092*(perturbations+1):
        raise ValueError("Candidate containment records incomplete")
    grouped={}
    with_modes=result.assign(
        mode=np.where(result["repeat"].eq(-1),"clean","perturbed")
    )
    for (split,kind,mode),subset in with_modes.groupby(
        ["split","candidate_type","mode"],sort=True
    ):
        key=f"{split}/{kind}/{mode}"
        grouped[key]={
            "samples":len(subset),
            "outside_patch":int((subset.all_endpoints_inside_patch==0).sum()),
            "outside_fraction":float((subset.all_endpoints_inside_patch==0).mean()),
            "near_edge_within_1mm":int(subset.near_patch_edge_within_1mm.sum()),
            "min_margin_mm":float(subset.patch_margin_mm.min()),
            "p01_margin_mm":float(subset.patch_margin_mm.quantile(.01)),
            "median_margin_mm":float(subset.patch_margin_mm.median()),
        }
    report={
        "status":"measured_frozen_CT28_3D_patch_physical_endpoint_coverage",
        "candidate_scenes":1404,
        "source_candidate_rows":15092,
        "endpoints":7020,
        "perturbation_repetitions":perturbations,
        "position_noise_mm":.8,
        "tangent_noise_deg":20.,
        "seed":seed,
        "patch_side":config.side,
        "patch_voxel_mm":config.voxel_mm,
        "physical_half_extent_mm":(config.side-1)*config.voxel_mm/2.,
        "groups":grouped,
        "sha256":{
            "plan":sha(plan_path),"endpoints":sha(endpoint_path),
            "candidates":sha(candidate_path)
        },
        "held_out_test_accessed":False,
        "ground_truth_used_for_model_or_FOV_audit":False,
        "warnings":[
            "Physical endpoint containment does not prove CT patch is inside scanned volume.",
            "Out-of-patch endpoints create silent loss of endpoint heatmap evidence; fail closed in downstream 3D neural use.",
            "Patch geometry fixed 24mm is frozen experimental setup, not automatically appropriate for natural candidate distances.",
        ],
    }
    return report,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--side",type=int,default=24)
    parser.add_argument("--voxel-mm",type=float,default=1.)
    parser.add_argument("--perturbations",type=int,default=15)
    parser.add_argument("--seed",type=int,default=20260920)
    args=parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Patch endpoint audit evidence is create-only")
    report,records=run(
        config=PatchConfig(side=args.side,voxel_mm=args.voxel_mm),
        perturbations=args.perturbations,seed=args.seed,
    )
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/"physical_patch_coverage.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n"
    )
    # Save only physical margin summaries, not raw real endpoint coordinates.
    records.groupby(
        ["scan_id","split","candidate_type","repeat"],as_index=False
    ).agg(
        candidate_count=("patch_margin_mm","size"),
        min_margin_mm=("patch_margin_mm","min"),
        outside_patch=("all_endpoints_inside_patch",lambda x:int((x==0).sum())),
        near_edge=("near_patch_edge_within_1mm","sum")
    ).to_csv(args.output_dir/"per_patient_perturbation_patch_coverage.csv",index=False)
    print(json.dumps({
        "status":report["status"],
        "groups":report["groups"],
        "held_out_test_accessed":False,
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
