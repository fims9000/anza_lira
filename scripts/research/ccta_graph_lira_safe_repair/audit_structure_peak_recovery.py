#!/usr/bin/env python3
"""Audit structural-bridge endpoint peak recovery on CT28 frozen candidates.

No CT or vessel labels are used as network input. The y field is used ONLY for
stratified diagnostic reporting. All 6 held-out CT28 test IDs are excluded.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from models.structural_bridge3d import _peak_centers
from ccta_3d_an_patches import PatchConfig,local_frame


def run(scenes:Path,output:Path,frame_mode:str="fixed_lps",side:int=24,negative_limit:int=242):
    if output.exists():
        raise FileExistsError("Immutable audit already exists")
    cfg=PatchConfig(side=side,frame_mode=frame_mode)
    plan=pd.read_csv(scenes/"scene_plan_train_val.csv")
    endpoints=pd.read_csv(scenes/"scene_endpoints_train_val.csv")
    candidates=pd.read_csv(scenes/"scene_candidates_train_val.csv")
    allowed=set(plan.loc[plan.split.eq("val"),"scene_id"])
    endpoints=endpoints[endpoints.scene_id.isin(allowed)]
    candidates=candidates[candidates.scene_id.isin(allowed)]
    positives=candidates.loc[candidates.y.astype(int).eq(1)]
    negatives=candidates.loc[candidates.y.astype(int).eq(0)]
    negatives=negatives.sample(n=min(negative_limit,len(negatives)),random_state=20261111)
    sampled=pd.concat((positives,negatives),ignore_index=True)
    by_scene={k:v.set_index("endpoint_id") for k,v in endpoints.groupby("scene_id")}
    coords=np.stack(np.meshgrid(*(np.arange(side,dtype=float) for _ in range(3)),
                                indexing="ij"),axis=0)
    rows=[]
    for item in sampled.itertuples(index=False):
        refs=by_scene[item.scene_id]
        points=refs.loc[str(item.endpoint_ids).split("|"),
                        ["x_lps","y_lps","z_lps"]].to_numpy(dtype=float)
        center,frame=local_frame(points,cfg.frame_mode)
        local=(frame.T@(points.T-center[:,None])).T/cfg.voxel_mm+(side-1)/2
        heat=np.zeros((side,side,side),dtype=np.float32)
        for point in local:
            heat+=np.exp(-((coords-point[:,None,None,None])**2).sum(0)/
                         (2*(cfg.endpoint_sigma_mm/cfg.voxel_mm)**2)).astype(np.float32)
        heat=heat.clip(0,1)
        peaks,valid=_peak_centers(torch.from_numpy(heat[None]),len(points))
        predicted=peaks[0].numpy()
        dist=min(max(np.linalg.norm(predicted[idx]-local[k]) for k,idx in enumerate(order))
                 for order in itertools.permutations(range(len(points))))
        sep=min(np.linalg.norm(points[i]-points[j]) for i in range(len(points))
                for j in range(i+1,len(points)))
        rows.append({
            "kind":item.candidate_type,"label":int(item.y),"valid":bool(valid.item()),
            "worst_endpoint_recovery_error_voxel":float(dist),
            "all_peaks_within_2mm":bool(valid.item() and dist<=2),
            "minimum_endpoint_separation_mm":float(sep),
        })
    report={"status":"frozen_CCTA_geometry_only","frame_mode":frame_mode,
            "side":side,"sample_n":len(rows),"held_out_CT28_test_accessed":False,"groups":{}}
    frame=pd.DataFrame(rows)
    for (kind,label),g in frame.groupby(["kind","label"]):
        report["groups"][f"{kind}/y={label}"]={
            "n":len(g),"peaks_valid_fraction":float(g.valid.mean()),
            "fraction_all_within_2mm":float(g.all_peaks_within_2mm.mean()),
            "median_worst_recovery_error_voxel":float(g.worst_endpoint_recovery_error_voxel.median()),
            "minimum_endpoint_separation_mm":float(g.minimum_endpoint_separation_mm.min()),
        }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n")
    frame.to_csv(output.with_suffix(".csv"),index=False)
    print(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--scenes",type=Path,default=ROOT/"artifacts/varvara/ct28_graph_lira")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--frame-mode",choices=("canonical","fixed_lps"),default="fixed_lps")
    parser.add_argument("--side",type=int,default=24)
    parser.add_argument("--negative-limit",type=int,default=242)
    a=parser.parse_args()
    run(a.scenes,a.out,frame_mode=a.frame_mode,side=a.side,negative_limit=a.negative_limit)
