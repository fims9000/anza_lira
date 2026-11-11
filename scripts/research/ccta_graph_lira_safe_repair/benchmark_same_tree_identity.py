#!/usr/bin/env python3
"""True hard-negative PAIR benchmark: same 26-connected vascular tree, different branch.

Each generated 3D Y tree contributes exactly two candidates with IDENTICAL CT
voxels and dense vessel targets. They share the first endpoint and have equal
endpoint distance. Only the second endpoint differs: trunk continuation=1,
branch turn=0. Train and validation are completely different randomized trees.

This is an artificial *straight-through-continuation* convention; at an actual
coronary bifurcation both branches can be anatomically valid. Results MUST NOT
be represented as vessel-clinical labels or CT28 test scores.

Run from repository root. No medical data accessed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, label as cc_label
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from benchmark_structural_tubes import fit, normalize_synthetic_hu_histograms


def _unit(x:np.ndarray)->np.ndarray:
    return x/max(1e-12,float(np.linalg.norm(x)))


def _field(grid:np.ndarray,point:np.ndarray,sigma:float)->np.ndarray:
    return np.exp(-np.square(grid-point[:,None,None,None]).sum(axis=0)
                  /(2*sigma*sigma)).astype(np.float32)


def make_paired_trees(
    n_trees:int,seed:int,*,side:int=24,
    angle_low_deg:float=12.,angle_high_deg:float=75.,
    same_histogram:bool=False,
) -> tuple[np.ndarray,np.ndarray,np.ndarray,dict]:
    if n_trees<4 or side<22 or not (3<=angle_low_deg<angle_high_deg<=100):
        raise ValueError("Invalid paired Y tree experiment")
    rng=np.random.default_rng(seed)
    grid=np.stack(np.meshgrid(*(
        np.arange(side,dtype=np.float32) for _ in range(3)
    ),indexing="ij"),axis=0)
    x=np.zeros((2*n_trees,4,side,side,side),dtype=np.float32)
    y=np.tile(np.asarray([1.,0.],dtype=np.float32),n_trees)
    targets=np.zeros((2*n_trees,1,side,side,side),dtype=np.float32)
    angles=[]
    target_diffs=[]
    min_fovs=[]
    for i in range(n_trees):
        center=np.full(3,(side-1)/2,dtype=np.float64)+rng.uniform(-.8,.8,size=3)
        axis=_unit(rng.normal(size=3))
        axis_b=_unit(np.cross(axis,np.eye(3)[np.argmin(np.abs(axis))]))
        halfspan=float(rng.uniform(3.9,5.1))
        a=center-halfspan*axis
        b=center+halfspan*axis
        hub=center+axis*rng.uniform(-.4,.4)
        angle=float(rng.uniform(angle_low_deg,angle_high_deg))
        bend_axis=_unit(np.cos(np.deg2rad(angle))*axis+
                        np.sin(np.deg2rad(angle))*axis_b)
        # Solve |branch_end-a| == |b-a|, preventing endpoint-distance shortcuts.
        offset=hub-a
        projection=float(np.dot(offset,bend_axis))
        length_ab=float(np.linalg.norm(b-a))
        span=np.sqrt(max(0.,projection**2+length_ab**2-
                         float(np.dot(offset,offset))))-projection
        branch=hub+span*bend_axis
        if np.min([a,b,branch])<1 or np.max([a,b,branch])>side-2:
            raise ValueError("Generated endpoint outside safe 3D patch")
        target_diffs.append(abs(np.linalg.norm(branch-a)-length_ab))
        angle_actual=math.degrees(math.acos(float(np.clip(
            np.dot(axis,bend_axis),-1,1
        ))))
        angles.append(angle_actual)
        radius=float(rng.uniform(.68,.95))
        vessel=np.zeros((side,side,side),dtype=np.float32)
        for start,end in ((a,b),(hub,branch)):
            for t in np.linspace(0,1,100):
                vessel=np.maximum(vessel,_field(
                    grid,start*(1-t)+end*t,radius
                ))
        binary=(vessel>.27).astype(np.float32)
        labels,_=cc_label(binary>0,np.ones((3,3,3),dtype=np.uint8))
        loc=[np.rint(point).astype(int) for point in (a,b,branch)]
        ids=[int(labels[tuple(pos)]) for pos in loc]
        if not (ids[0]>0 and ids[0]==ids[1]==ids[2]):
            raise AssertionError("Positive and negative must be in SAME GT vessel component")
        # Identical noisy HU for both branches within every Y tree.
        raw=np.clip(-.74+float(rng.uniform(.83,1.08))*vessel+
                    rng.normal(0,float(rng.uniform(.04,.09)),size=vessel.shape),
                    -1,1).astype(np.float32)
        ct=gaussian_filter(raw,sigma=.35).astype(np.float32)
        for j,endpoint in enumerate((b,branch)):
            row=2*i+j
            x[row,0]=ct
            x[row,1]=_field(grid,a,1.5)
            x[row,2]=_field(grid,endpoint,1.5)
            targets[row,0]=binary
        min_fovs.append(float(np.min(np.concatenate((a,b,branch)))))
    if same_histogram:
        x=normalize_synthetic_hu_histograms(x)
    # Verify no pair-dependent CT signal/label leakage.
    if not np.array_equal(x[::2,0],x[1::2,0]):
        raise AssertionError("Hard negatives must have byte-identical CT within pair")
    if not np.array_equal(targets[::2],targets[1::2]):
        raise AssertionError("Hard negative vessel targets must match within Y tree")
    return x,y,targets,{
        "n_independent_trees":n_trees,
        "label_convention":"straight_trunk_vs_turning_branch_only",
        "branch_angle_deg_min":float(min(angles)),
        "branch_angle_deg_max":float(max(angles)),
        "maximum_positive_negative_endpoint_distance_difference_vox":
            float(max(target_diffs)),
        "same_ct_per_pair":True,
        "same_dense_gt_per_pair":True,
        "held_out_CT28_test_accessed":False,
    }


def _standalone_oracle(x:np.ndarray,truth:np.ndarray)->dict:
    from models.structural_bridge3d import VesselBridgeReadout
    readout=VesselBridgeReadout(degree=2)
    with torch.no_grad():
        scores=[]
        for k in range(0,len(truth),8):
            scores.extend(readout(
                torch.from_numpy(truth[k:k+8,0]),
                torch.from_numpy(x[k:k+8,1:]),
            ).cpu().numpy().tolist())
    p=np.asarray(scores)
    return {
        "AUROC":float(roc_auc_score(np.tile([1,0],len(p)//2),p)),
        "pairwise_trunk_wins":int(np.sum(p[::2]>p[1::2])),
        "pairwise_branch_wins":int(np.sum(p[1::2]>p[::2])),
        "false_branch_accepted_tau085":int(np.sum(p[1::2]>=.85)),
        "true_trunk_accepted_tau085":int(np.sum(p[::2]>=.85)),
    }


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--train-trees",type=int,default=48)
    p.add_argument("--val-trees",type=int,default=24)
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--seeds",nargs="+",type=int,default=[17,41])
    p.add_argument("--device",default="cpu")
    p.add_argument("--histogram-control",action="store_true")
    p.add_argument("--endpoints-only",action="store_true",
                   help="Train identical models with CT channel masked, diagnostic only")
    args=p.parse_args()
    if args.out.exists() or args.epochs<1 or args.train_trees<4 or args.val_trees<4:
        raise ValueError("Bad or existing experiment output")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Requested CUDA unavailable")
    torch.set_num_threads(2)
    train=make_paired_trees(args.train_trees,20261111,
                            same_histogram=args.histogram_control)
    val=make_paired_trees(args.val_trees,20261911,
                          same_histogram=args.histogram_control)
    if args.endpoints_only:
        def without_ct(batch):
            features=batch[0].copy()
            features[:,0]=0
            return (features,batch[1],batch[2],batch[3])
        train=without_ct(train)
        val=without_ct(val)
    models=(
        "conv3d_sep_structure",
        "anza3d_sep_structure",
        "anza3d_noaxial_sep_structure",
        "conv3d_deepsets_sep_structure",
        "anza3d_deepsets_sep_structure",
    )
    measurements=[]
    for seed in args.seeds:
        for variant in models:
            values=fit(
                variant,"PAIR",train[:3],val[:3],epochs=args.epochs,
                batch_size=8,seed=seed,device=args.device,
            )
            # fit returns scores only as aggregate diagnostic; for paired
            # tree-specific outcomes one must add a dedicated score export.
            measurements.append(values)
            print(json.dumps(values),flush=True)
    report={
        "status":"paired_same_tree_branch_identity_synthetic_only",
        "warning":"No clinical branch labels; all negative and positive endpoints are actually 26-connected.",
        "training_tree_seed":20261111,
        "independent_validation_tree_seed":20261911,
        "train_meta":train[3],"validation_meta":val[3],
        "equal_histogram":bool(args.histogram_control),
        "endpoints_only":bool(args.endpoints_only),
        "models":list(models),"seeds":args.seeds,
        "oracle":_standalone_oracle(val[0],val[2]),
        "measurements":measurements,
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":report["status"],"n_runs":len(measurements),
                      "oracle":report["oracle"]},flush=True))


if __name__=="__main__":
    main()
