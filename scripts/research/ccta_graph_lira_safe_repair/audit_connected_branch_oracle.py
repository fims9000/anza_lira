#!/usr/bin/env python3
"""Oracle stress: two paths in the SAME connected vessel tree need not be same branch.

One synthetic Y-shaped 3D vessel has a straight trunk and a branch. The PAIR
along the trunk is the annotated continuation; the PAIR from trunk to branch
is a HARD negative despite both lying in the same 26-connected component.
This is a directional branch-identity task and cannot be measured by clDice
or undirected 26-connectedness alone. No CT28 or patient data is opened.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label as cc_label
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from models.structural_bridge3d import VesselBridgeReadout,FuzzyConnectedness3d


BRANCH_ANGLES_DEGREES=(15,20,25,35,50,65,90)


def generate(n:int=14,seed:int=20261111,side:int=24):
    if n<4 or side<20:
        raise ValueError("Expected >=4 synthetic Y-shaped samples and side>=20")
    rng=np.random.default_rng(seed)
    xyz=np.stack(np.meshgrid(*(np.arange(side,dtype=np.float32) for _ in range(3)),
                             indexing="ij"),axis=0)
    mid=(side-1)/2.
    ys=[]
    probs=[]
    merged=[]
    components_same=[]
    for i in range(n):
        angle=BRANCH_ANGLES_DEGREES[i%len(BRANCH_ANGLES_DEGREES)]
        theta=np.deg2rad(angle)
        # Every label pair shares EXACT SAME vessel image and only candidate
        # endpoints differ. Permutation/contrast can never encode class.
        turn=float(rng.uniform(-.4,.4))
        trunk_a=np.array([4.,mid+turn,mid],dtype=float)
        trunk_b=np.array([side-5.,mid+turn,mid],dtype=float)
        junction=np.array([mid,mid+turn,mid],dtype=float)
        # Realistically ambiguous hard branch: at shallow angles it nearly
        # follows the trunk and may look like a valid continuation.
        spur_b=junction+np.array([7.*np.cos(theta),7.*np.sin(theta),0.])
        density=np.zeros((side,side,side),dtype=np.float32)
        for p,q in ((trunk_a,trunk_b),(junction,spur_b)):
            for t in np.linspace(0,1,115):
                center=(1-t)*p+t*q
                dist=((xyz-center[:,None,None,None])**2).sum(axis=0)
                density=np.maximum(density,np.exp(-dist/(2*.75**2)).astype(np.float32))
        mask=(density>.25).astype(np.float32)
        cc,_=cc_label(mask,np.ones((3,3,3),dtype=np.uint8))
        for label,end in ((1,trunk_b),(0,spur_b)):
            p=np.stack((trunk_a,end))
            tips=[]
            for point in p:
                dist=((xyz-point[:,None,None,None])**2).sum(0)
                tips.append(np.exp(-dist/(2*1.5**2)).astype(np.float32))
            ends=tuple(tuple(np.rint(x).astype(int)) for x in p)
            same=(cc[ends[0]]>0 and cc[ends[0]]==cc[ends[1]])
            if not same:
                raise AssertionError("Both alternatives must be 26-connected ground truth")
            components_same.append(int(same))
            ys.append(label)
            probs.append(mask)
            merged.append(np.stack(tips))
    return np.stack(probs),np.stack(merged),np.asarray(ys),np.asarray(components_same)


def run(output:Path,n:int=14,seed:int=20261111):
    if output.exists():
        raise FileExistsError("Immutable oracle report already exists")
    vessel,tips,y,connected=generate(n=n,seed=seed)
    with torch.no_grad():
        v=torch.from_numpy(vessel)
        h=torch.from_numpy(tips)
        new=VesselBridgeReadout(2)(v,h).numpy()
        old=VesselBridgeReadout(2)(v,h.sum(dim=1).clamp(0,1)).numpy()
        binary=FuzzyConnectedness3d(2,steps=32)(
            v,h.sum(dim=1).clamp(0,1)
        ).numpy()
    metrics={}
    angle_groups={}
    for i in range(n):
        angle=BRANCH_ANGLES_DEGREES[i%len(BRANCH_ANGLES_DEGREES)]
        angle_groups.setdefault(str(angle),[]).append(2*i+1)
    for name,p in (("separate_tip_path_oracle",new),
                   ("summed_tip_path_oracle",old),
                   ("component_connectedness_oracle",binary)):
        metrics[name]={
            "AUROC":float(roc_auc_score(y,p)),
            "AUPRC":float(average_precision_score(y,p)),
            "mean_true":float(p[y==1].mean()),
            "mean_false":float(p[y==0].mean()),
            "false_accepted_at_085":int(((p>=.85)&(y==0)).sum()),
            "true_accepted_at_085":int(((p>=.85)&(y==1)).sum()),
            "false_branch_by_angle_degrees":{
                angle:{
                    "n":len(index),
                    "mean_false_score":float(p[index].mean()),
                    "false_accepted_at_085":int((p[index]>=.85).sum()),
                } for angle,index in angle_groups.items()
            },
        }
    result={
        "status":"same_connected_tree_branch_identity_ORACLE_only",
        "n_paired_trees":n,
        "n_candidates":len(y),
        "all_candidate_endpoints_in_one_gt_component":bool(connected.all()),
        "fixed_train_seed":seed,
        "branch_angle_grid_degrees":list(BRANCH_ANGLES_DEGREES),
        "held_out_ct28_test_accessed":False,
        "clinical_data_used":False,
        "notes":"This is a hand-constructed directional-continuation label, not universal coronary branch truth.",
        "metrics":metrics,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--n",type=int,default=14)
    args=parser.parse_args()
    run(args.out,args.n)
