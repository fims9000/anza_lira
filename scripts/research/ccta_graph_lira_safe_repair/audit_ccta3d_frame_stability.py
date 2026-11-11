#!/usr/bin/env python3
"""Read-only CT28 endpoint-frame stability audit; requires no medical CT.

For each frozen scene, recreate the same 15 endpoint perturbations used by
CT28 consistency. Compare the canonical moving frame against a fixed physical
LPS frame, without reading labels during patch creation or changing K10/15.
The deterministic output measures frame rotation, NOT network accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ccta_3d_an_patch_cache import SCENES
from ccta_3d_an_patches import local_frame
from ct28_perturbations import perturb_scene_endpoints


def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def _rotation(a: np.ndarray,b: np.ndarray) -> float:
    return float(np.rad2deg(np.arccos(np.clip((np.trace(a.T@b)-1.)/2.,-1.,1.))))


def audit(*, scenes_dir:Path, output:Path, repetitions:int=15,
          seed:int=20260920, position_noise_mm:float=.8,
          tangent_noise_deg:float=20., split:str="val")->dict:
    if repetitions<1 or repetitions>100 or position_noise_mm<0 or tangent_noise_deg<0:
        raise ValueError("Invalid perturbation settings")
    if split not in {"train","val"}:
        raise ValueError("Only frozen development splits permitted")
    if output.exists():
        raise FileExistsError("Create a fresh output file; never overwrite previous evidence")
    endpoint_file=scenes_dir/"scene_endpoints_train_val.csv"
    candidate_file=scenes_dir/"scene_candidates_train_val.csv"
    plan_file=scenes_dir/"scene_plan_train_val.csv"
    plans=pd.read_csv(plan_file)
    patient_by_scene=plans.loc[plans.split.eq(split),["scene_id","scan_id"]]
    endpoints=pd.read_csv(endpoint_file).merge(patient_by_scene,on="scene_id",validate="many_to_one")
    candidates=pd.read_csv(candidate_file).merge(patient_by_scene,on="scene_id",validate="many_to_one")
    if endpoints.empty or candidates.empty:
        raise ValueError("No frozen CT28 candidates to audit")
    if candidates[["scene_id","candidate_id"]].duplicated().any():
        raise ValueError("Duplicate candidate identity")
    ep_groups={key:g for key,g in endpoints.groupby("scene_id",sort=False)}
    candidate_groups={key:g for key,g in candidates.groupby("scene_id",sort=False)}
    rows=[]
    for scene_id,group in candidate_groups.items():
        ep=ep_groups[scene_id]
        if ep.endpoint_id.duplicated().any():
            raise ValueError("Duplicate endpoint in scene")
        lookup=ep.set_index("endpoint_id")
        perturbed=[]
        for repetition in range(repetitions):
            item=perturb_scene_endpoints(
                ep,seed=seed,repetition=repetition,
                position_noise_mm=position_noise_mm,
                tangent_noise_deg=tangent_noise_deg,
            )
            perturbed.append(item.set_index("endpoint_id"))
        for row in group.itertuples(index=False):
            ids=str(row.endpoint_ids).split("|")
            if len(ids)!=(2 if row.candidate_type=="PAIR" else 3):
                raise ValueError("Unexpected candidate degree")
            original=lookup.loc[ids,["x_lps","y_lps","z_lps"]].to_numpy(dtype=float)
            _,base=local_frame(original,"canonical")
            _,fixed=local_frame(original,"fixed_lps")
            largest=0.
            flips=0
            for item in perturbed:
                pts=item.loc[ids,["x_lps","y_lps","z_lps"]].to_numpy(dtype=float)
                _,rotated=local_frame(pts,"canonical")
                _,fixed_perturbed=local_frame(pts,"fixed_lps")
                if not np.array_equal(fixed,fixed_perturbed):
                    raise AssertionError("Fixed LPS orientation changed under perturbation")
                angle=_rotation(base,rotated)
                largest=max(largest,angle)
                flips+=angle>60.
            rows.append({
                "scene_id":scene_id,
                "scan_id":int(row.scan_id),
                "candidate_id":row.candidate_id,
                "candidate_type":row.candidate_type,
                "positive":int(row.y),
                "canonical_max_rotation_deg":largest,
                "canonical_rotations_gt_60":int(flips),
                "canonical_any_rotation_gt_60":bool(largest>60.),
                "canonical_any_rotation_gt_120":bool(largest>120.),
                "fixed_lps_orientation_changed":False,
            })
    frame=pd.DataFrame(rows)
    groups={}
    for (kind,positive),subset in frame.groupby(["candidate_type","positive"],sort=True):
        groups[f"{kind}/y={positive}"]={
            "n_candidates":len(subset),
            "fraction_any_rotation_gt60":float(subset.canonical_any_rotation_gt_60.mean()),
            "fraction_any_rotation_gt120":float(subset.canonical_any_rotation_gt_120.mean()),
            "mean_max_rotation_deg":float(subset.canonical_max_rotation_deg.mean()),
        }
    report={
        "status":"geometric_audit_only_no_neural_predictions",
        "split":split,
        "seed":seed,"repetitions":repetitions,
        "position_noise_mm":position_noise_mm,
        "tangent_noise_deg":tangent_noise_deg,
        "total_candidates":len(frame),
        "total_scenes":len(candidate_groups),
        "groups":groups,
        "baseline_frame_mode":"canonical",
        "experimental_frame_mode":"fixed_lps",
        "medical_images_read":False,
        "held_out_test_accessed":False,
        "source_sha256":{p.name:_sha256(p) for p in (plan_file,endpoint_file,candidate_file)},
        "interpretation":"Frame rotation alone does not establish classifier sensitivity or ANZA superiority.",
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    frame.to_csv(output.with_suffix(".csv"),index=False)
    return report


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes-dir",type=Path,default=SCENES)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--split",choices=("train","val"),default="val")
    p.add_argument("--repetitions",type=int,default=15)
    p.add_argument("--seed",type=int,default=20260920)
    args=p.parse_args()
    print(json.dumps(audit(scenes_dir=args.scenes_dir,output=args.output,
                           split=args.split,repetitions=args.repetitions,seed=args.seed),
                     indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
