#!/usr/bin/env python3
"""Deterministic *real CT28 coordinates* 3D endpoint readout audit, no CT required.

Quantify exact failures in the old summed-Gaussian 3-voxel NMS and the new
per-tip subvoxel moment encoder for TRUE PAIR/JUNCTION candidates.
Validation label y is used only to choose known positive candidates, never
to modify neural model inputs or fit any model. Six held-out CT28 test
subjects are excluded. Report immutable SHA-bound per-candidate metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from models.structural_bridge3d import _peak_centers,_peak_centers_from_channels
from ccta_3d_an_patches import PatchConfig,local_frame

SCENES=ROOT/"artifacts/varvara/ct28_graph_lira"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda:handle.read(1<<20),b""):
            h.update(data)
    return h.hexdigest()


def decode_one(points_lps:np.ndarray,config:PatchConfig)->dict:
    """Both renderings use the SAME physical center and LPS coordinate basis."""
    points=np.asarray(points_lps,dtype=np.float64)
    if points.shape not in ((2,3),(3,3)):
        raise ValueError("Expected two or three endpoint LPS coordinates")
    points=points[np.lexsort((points[:,2],points[:,1],points[:,0]))]
    center,frame=local_frame(points,config.frame_mode)
    local=(frame.T@(points-center).T).T/config.voxel_mm
    expected=local+(config.side-1)/2.
    spatial=np.arange(config.side,dtype=np.float32)-(config.side-1)/2.
    grid=np.stack(np.meshgrid(spatial,spatial,spatial,indexing="ij"))
    fields=[]
    for pos in local:
        fields.append(np.exp(-(
            (grid[0]-pos[0])**2+(grid[1]-pos[1])**2+(grid[2]-pos[2])**2
        )/(2*(config.endpoint_sigma_mm/config.voxel_mm)**2)).astype(np.float32))
    while len(fields)<3:
        fields.append(np.zeros_like(fields[0]))
    separate=torch.from_numpy(np.stack(fields)[None])
    summed=torch.from_numpy(
        np.minimum(1.,np.sum(np.stack(fields),axis=0,dtype=np.float32))[None]
    )
    old,old_valid=_peak_centers(summed,len(points))
    new,new_valid=_peak_centers_from_channels(separate,len(points))
    older=old[0].numpy()
    latest=new[0].numpy()
    old_assignments=[
        max(float(np.linalg.norm(older[j]-expected[k])) for j,k in enumerate(perm))
        for perm in itertools.permutations(range(len(points)))
    ]
    old_best=min(old_assignments)
    new_max=float(np.linalg.norm(latest-expected,axis=1).max())
    pairwise=[
        float(np.linalg.norm(points[i]-points[j]))
        for i in range(len(points)) for j in range(i+1,len(points))
    ]
    return {
        "degree":len(points),
        "min_pair_distance_mm":min(pairwise),
        "old_peaks_valid":bool(old_valid.item()),
        "new_peaks_valid":bool(new_valid.item()),
        "legacy_best_assignment_max_error_mm":old_best*config.voxel_mm,
        "new_subvoxel_max_error_mm":new_max*config.voxel_mm,
        "legacy_all_within_1mm":bool(old_valid.item() and old_best*config.voxel_mm<=1.),
        "new_all_within_1mm":bool(new_valid.item() and new_max*config.voxel_mm<=1.),
    }


def audit(scenes:Path,output:Path,config:PatchConfig)->dict:
    if config.endpoint_encoding!="separate":
        raise ValueError("New readout audit requires separate-tip patch protocol")
    if output.exists() or output.with_suffix(".csv").exists():
        raise FileExistsError("Immutable CT28 endpoint readout audit already exists")
    plan_file=scenes/"scene_plan_train_val.csv"
    ep_file=scenes/"scene_endpoints_train_val.csv"
    cand_file=scenes/"scene_candidates_train_val.csv"
    plans=pd.read_csv(plan_file)
    endpoints=pd.read_csv(ep_file)
    candidates=pd.read_csv(cand_file)
    allowed=set(plans.loc[plans.split.eq("val"),"scene_id"])
    positives=candidates.loc[candidates.scene_id.isin(allowed)&candidates.y.astype(int).eq(1)]
    keys=endpoints.loc[endpoints.scene_id.isin(allowed)].set_index(["scene_id","endpoint_id"])
    if keys.index.has_duplicates:
        raise ValueError("Duplicate endpoint identity")
    rows=[]
    for candidate in positives.itertuples(index=False):
        names=str(candidate.endpoint_ids).split("|")
        pts=np.stack([
            keys.loc[(candidate.scene_id,name),["x_lps","y_lps","z_lps"]].to_numpy(dtype=float)
            for name in names
        ])
        case=decode_one(pts,config)
        rows.append({
            "scene_id":str(candidate.scene_id),
            "candidate_id":str(candidate.candidate_id),
            "candidate_type":str(candidate.candidate_type),
            **case,
        })
    df=pd.DataFrame(rows)
    summary={}
    for kind,group in df.groupby("candidate_type",sort=True):
        summary[kind]={
            "n_true_candidates":int(len(group)),
            "min_endpoint_dist_lt_3mm":int((group.min_pair_distance_mm<3.).sum()),
            "old_valid_fraction":float(group.old_peaks_valid.mean()),
            "new_valid_fraction":float(group.new_peaks_valid.mean()),
            "legacy_all_within_1mm_fraction":float(group.legacy_all_within_1mm.mean()),
            "new_all_within_1mm_fraction":float(group.new_all_within_1mm.mean()),
            "legacy_mean_max_error_mm":float(group.legacy_best_assignment_max_error_mm.mean()),
            "new_mean_max_error_mm":float(group.new_subvoxel_max_error_mm.mean()),
            "legacy_worst_error_mm":float(group.legacy_best_assignment_max_error_mm.max()),
            "new_worst_error_mm":float(group.new_subvoxel_max_error_mm.max()),
        }
    if set(summary)!={"PAIR","JUNCTION"}:
        raise ValueError("Frozen validation true PAIR/JUNCTION candidate types changed")
    report={
        "status":"frozen_CT28_validation_true_candidate_geometry_only",
        "n_candidates":len(df),
        "groups":summary,
        "patch":vars(config),
        "held_out_CT28_test_accessed":False,
        "medical_CT_read":False,
        "labels_used_only_for_audit_subset":True,
        "source_sha256":{
            p.name:sha(p) for p in (plan_file,ep_file,cand_file)
        },
        "interpretation":"Endpoint geometric recovery only; cannot establish neural or clinical accuracy",
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    df.to_csv(output.with_suffix(".csv"),index=False)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes",type=Path,default=SCENES)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--side",type=int,default=24)
    p.add_argument("--voxel-mm",type=float,default=1.)
    a=p.parse_args()
    audit(a.scenes,a.out,PatchConfig(side=a.side,voxel_mm=a.voxel_mm,
                                     frame_mode="fixed_lps",endpoint_encoding="separate"))
