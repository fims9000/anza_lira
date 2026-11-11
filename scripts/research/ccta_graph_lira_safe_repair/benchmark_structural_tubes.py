#!/usr/bin/env python3
"""Synthetic 3D tube connectivity challenge for ANZA/CNN structure variants.

Not ImageCAS, not a medical performance claim. It tests whether dense vessel
supervision + candidate bottleneck improve PAIR/JUNCTION continuity decisions
on held-out random 3D curve geometries under exactly matched model training.
The validation sample is generated from a different immutable RNG stream and
never used to choose epochs, thresholds or hyperparameters in this script.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter,label as cc_label
from sklearn.metrics import average_precision_score,roc_auc_score
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from models.ccta3d_candidate_encoder import Candidate3dEncoder,parameter_budget
from models.structural_bridge3d import vessel_structure_loss,VesselBridgeReadout,_peak_centers,_peak_centers_from_channels


def _unit(v):
    return v/max(1e-8,np.linalg.norm(v))


def _add_curve(centers,axis_points):
    a,b=axis_points
    for t in np.linspace(0,1,41):
        centers.append(a*(1-t)+b*t)


def make_phantoms(n:int,seed:int,kind:str,side:int=12,*,separate_endpoints:bool=False,close_junction:bool=False):
    if kind not in {"PAIR","JUNCTION"} or side<10:
        raise ValueError("Unsupported structural challenge")
    rng=np.random.default_rng(seed)
    grid=np.stack(np.meshgrid(*(np.arange(side,dtype=np.float32) for _ in range(3)),
                              indexing="ij"),axis=0)
    patch=np.empty((n,4 if separate_endpoints else 2,side,side,side),dtype=np.float32)
    vessel_masks=np.empty((n,1,side,side,side),dtype=np.float32)
    y=np.arange(n,dtype=np.float32)%2
    rng.shuffle(y)
    for i,positive in enumerate(y):
        for attempt in range(64 if close_junction and kind=="JUNCTION" else 32):
            center=np.full(3,(side-1)/2)+rng.normal(0,.24,3)
            u=_unit(rng.normal(size=3))
            ref=np.eye(3)[np.argmin(abs(u))]
            v=_unit(np.cross(u,ref))
            w=_unit(np.cross(u,v))
            extent=rng.uniform(3.1,3.7)
            curves=[]
            if kind=="PAIR":
                ends=np.stack((center-u*extent,center+u*extent))
                if positive:
                    curve=[center-u*extent]
                    bend=rng.uniform(-1.1,1.1)*v+rng.uniform(-.8,.8)*w
                    for t in np.linspace(0,1,41):
                        curve.append(ends[0]*(1-t)+ends[1]*t+np.sin(np.pi*t)*bend)
                    curves.append(curve)
                else:
                    # Similar endpoints and gross vessel volume, but the two
                    # visible arms divert to separate off-axis branches.
                    a=center-u*1.4+v*2.3
                    b=center+u*1.4-v*2.3
                    curve1=[];curve2=[]
                    _add_curve(curve1,(ends[0],a))
                    _add_curve(curve2,(ends[1],b))
                    curves.extend((curve1,curve2))
            else:
                # For close-tip stress, the first two true junction ends are
                # only ~2.2 voxels apart, BELOW the legacy 3-voxel NMS.
                dirs=(np.stack((u,.8*u+.6*v,-u)) if close_junction else
                      np.stack((u,-.5*u+.866*v,-.5*u-.866*v)))
                ends=np.stack([center+direction*extent for direction in dirs])
                for k,point in enumerate(ends):
                    if positive or k!=2:
                        terminal=center
                    else:
                        # A hard negative must genuinely fail to join the
                        # other two arms. Previously this arm ended at
                        # center+2.6*w+1.3*v; its thick vessel mask remained
                        # 26-connected, so labels claimed FALSE while the
                        # ground-truth geometry showed TRUE connectedness.
                        # Redirect the third arm tangentially *away* from
                        # the hub, maintaining similar line length without
                        # creating a coincident central branch.
                        terminal=point+w*extent
                    curve=[]
                    _add_curve(curve,(point,terminal))
                    curves.append(curve)
            vessel=np.zeros((side,side,side),dtype=np.float32)
            for curve in curves:
                for point in curve:
                    r2=((grid-point[:,None,None,None])**2).sum(axis=0)
                    vessel=np.maximum(vessel,np.exp(-r2/(2*.8**2)).astype(np.float32))
            # Distractors are present with the SAME distribution in both classes.
            if rng.random()<.5:
                start=center+3.5*w+rng.uniform(-1,1)*v
                direction=_unit(rng.normal(size=3))
                for t in np.linspace(-2,2,22):
                    point=start+direction*t
                    r2=((grid-point[:,None,None,None])**2).sum(axis=0)
                    vessel=np.maximum(vessel,np.exp(-r2/(2*.65**2)).astype(np.float32))
            target=(vessel>.42).astype(np.float32)
            background=rng.normal(-.75,.10,size=target.shape).astype(np.float32)
            intensity=rng.uniform(.8,1.)*vessel+background
            intensity=gaussian_filter(intensity,sigma=.4)
            intensity=np.clip(intensity,-1.,1.)
            heat=np.zeros((side,side,side),dtype=np.float32)
            separate=[]
            for point in ends:
                r2=((grid-point[:,None,None,None])**2).sum(axis=0)
                field=np.exp(-r2/(2*1.5**2)).astype(np.float32)
                heat+=field
                separate.append(field)
            patch[i,0]=intensity
            if separate_endpoints:
                while len(separate)<3:
                    separate.append(np.zeros_like(heat))
                patch[i,1:]=np.stack(separate,axis=0)
            else:
                patch[i,1]=heat.clip(0,1)
            vessel_masks[i,0]=target
            # Validate synthetic relation identity from the ACTUAL voxelized
            # reference, not from generation intent. Previously ~20% of
            # negative JUNCTION tubes were accidentally 26-connected.
            components,_=cc_label(target>0.5,np.ones((3,3,3),dtype=np.uint8))
            locations=np.rint(ends).astype(int)
            if (locations<0).any() or (locations>=side).any():
                continue
            ids=[int(components[tuple(v)]) for v in locations]
            observed=all(k>0 and k==ids[0] for k in ids)
            if observed==bool(positive):
                break
        else:
            raise RuntimeError(
                f"Cannot generate topologically valid {kind} y={int(positive)} "
                f"sample after 64 attempts" if close_junction else f"sample after 32 attempts"
            )
    return patch,y,vessel_masks


def heatmap_for_audit(x:np.ndarray)->np.ndarray:
    if x.ndim!=5 or x.shape[1] not in (2,4):
        raise ValueError("Expected two or four-channel patch input")
    return x[:,1] if x.shape[1]==2 else np.clip(x[:,1:].sum(axis=1),0,1)


def model_input_from_phantom(x:np.ndarray,variant:str)->np.ndarray:
    if "_sep_" in variant:
        if x.shape[1]!=4:
            raise ValueError("Separate-tip model requires four-channel synthetic patches")
        return x
    if x.shape[1]==2:
        return x
    return np.concatenate((x[:,:1],heatmap_for_audit(x)[:,None]),axis=1)


def simple_shortcut_audit(train,valid):
    """Train-oriented one-number prediction controls, never val-tuned signs."""
    train_x,train_y,_=train
    valid_x,valid_y,_=valid
    def features(x):
        ct=x[:,0]
        h=heatmap_for_audit(x)
        middle=ct.shape[-1]//2
        return {
            "hu_mean":ct.mean(axis=(1,2,3)),
            "hu_std":ct.std(axis=(1,2,3)),
            "hu_bright_fraction":(ct>-.35).mean(axis=(1,2,3)),
            "hu_midplane_mean":ct[:,:,middle,:].mean(axis=(1,2)),
            "heatmap_mean":h.mean(axis=(1,2,3)),
            "heatmap_max":h.max(axis=(1,2,3)),
        }
    a,b=features(train_x),features(valid_x)
    result={}
    for name in sorted(a):
        train_auc=roc_auc_score(train_y,a[name])
        orientation=1. if train_auc>=.5 else -1.
        hold_auc=roc_auc_score(valid_y,orientation*b[name])
        result[name]={
            "train_oriented_AUROC":float(max(train_auc,1.-train_auc)),
            "holdout_AUROC":float(hold_auc),
            "holdout_AUPRC":float(average_precision_score(valid_y,orientation*b[name])),
            "orientation_frozen_from_train":int(orientation),
        }
    return result


def normalize_synthetic_hu_histograms(input_patches:np.ndarray)->np.ndarray:
    """Same exact scalar HU histogram for every candidate, no label leakage.

    Per-patch rank remapping preserves spatial intensity order, while
    completely removing class-dependent global mean, variance and other
    1D HU histogram shortcuts. This is a *synthetic stress test only*, not
    a proposed medical HU preprocessing or production normalization.
    """
    if input_patches.ndim!=5 or input_patches.shape[1]!=2:
        raise ValueError("Expected candidate patch tensor B,2,D,H,W")
    output=input_patches.copy()
    n=output[0,0].size
    # Centered in [-1,1], every patch receives identical sorted values.
    target=np.linspace(-.99,.99,n,dtype=np.float32)
    for i in range(len(output)):
        flat=output[i,0].reshape(-1)
        index=np.argsort(flat,kind="stable")
        ranks=np.empty(n,dtype=np.int64)
        ranks[index]=np.arange(n)
        output[i,0]=target[ranks].reshape(output[i,0].shape)
    return output


def permute_hu_voxels(input_patches:np.ndarray,seed:int=20261111)->np.ndarray:
    """Destroy 3D spatial continuity but preserve HU histogram and heatmap."""
    if input_patches.ndim!=5 or input_patches.shape[1]!=2:
        raise ValueError("Expected B,2,D,H,W patches")
    rng=np.random.default_rng(seed)
    copied=input_patches.copy()
    for row in range(len(copied)):
        flat=copied[row,0].reshape(-1).copy()
        copied[row,0]=rng.permutation(flat).reshape(copied[row,0].shape)
    return copied


def fit(variant:str,kind:str,train,valid,*,epochs:int,batch_size:int,seed:int,device:str):
    train_x,train_y,train_masks=train
    valid_x,valid_y,valid_masks=valid
    train_x=model_input_from_phantom(train_x,variant)
    valid_x=model_input_from_phantom(valid_x,variant)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model=Candidate3dEncoder(variant,channels=6,candidate_type=kind).to(device)
    structured=variant.endswith("_structure")
    tensors=[torch.from_numpy(train_x),torch.from_numpy(train_y),torch.from_numpy(train_masks)]
    loader=DataLoader(TensorDataset(*tensors),batch_size=batch_size,shuffle=True,
                      generator=torch.Generator().manual_seed(seed))
    criterion=nn.BCEWithLogitsLoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4)
    start=time.perf_counter()
    for _ in range(epochs):
        model.train()
        for x,y,target in loader:
            x=x.to(device);y=y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if structured:
                logits,dense,bridge=model(x,return_structure=True)
                loss=(criterion(logits,y)+.35*vessel_structure_loss(dense,target.to(device),
                                                                   skeleton_iterations=2)
                      +.15*criterion(torch.logit(bridge),y))
            else:
                loss=criterion(model(x),y)
            loss.backward()
            optimizer.step()
    model.eval()
    scores=[];bridges=[]
    with torch.no_grad():
        for start_index in range(0,len(valid_y),batch_size):
            tensor=torch.from_numpy(valid_x[start_index:start_index+batch_size]).to(device)
            if structured:
                output,_,bridge=model(tensor,return_structure=True)
                bridges.extend(bridge.detach().cpu().tolist())
            else:
                output=model(tensor)
            scores.extend(torch.sigmoid(output).cpu().tolist())
    p=np.asarray(scores)
    # Deliberately preserve each HU histogram, randomize only its 3D location.
    # A model still scoring highly has learned a shortcut rather than tracing
    # anatomy; this control has the SAME frozen positive/negative labels.
    permuted=permute_hu_voxels(valid_x,seed=20261911)
    scrambled_scores=[]
    with torch.no_grad():
        for start_index in range(0,len(valid_y),batch_size):
            batch=torch.from_numpy(
                permuted[start_index:start_index+batch_size]
            ).to(device)
            scrambled_scores.extend(torch.sigmoid(model(batch)).cpu().tolist())
    p_scrambled=np.asarray(scrambled_scores)
    predicted=p>=.5
    strict=p>=.85
    true=valid_y.astype(bool)
    return {
        "variant":variant,"candidate_kind":kind,"seed":seed,
        "parameter_count":model.parameter_count(),
        "train_seconds":time.perf_counter()-start,
        "n_val":len(true),
        "val_AUROC":float(roc_auc_score(true,p)),
        "val_AUPRC":float(average_precision_score(true,p)),
        "HU_voxel_permuted_AUROC":float(roc_auc_score(true,p_scrambled)),
        "HU_voxel_permuted_AUPRC":float(average_precision_score(true,p_scrambled)),
        "val_precision_at_05":float(np.sum(predicted&true)/max(1,np.sum(predicted))),
        "val_recall_at_05":float(np.sum(predicted&true)/max(1,np.sum(true))),
        "val_false_positive_count":int(np.sum(predicted&~true)),
        "val_true_positive_count":int(np.sum(predicted&true)),
        "fixed_tau_085_true_accepted":int(np.sum(strict&true)),
        "fixed_tau_085_false_accepted":int(np.sum(strict&~true)),
        "fixed_tau_085_abstained":int(np.sum(~strict)),
        "fixed_tau_085_precision":(
            float(np.sum(strict&true)/np.sum(strict)) if strict.any() else None
        ),
        "fixed_tau_085_recall":float(np.sum(strict&true)/max(1,np.sum(true))),
        "val_brier":float(np.mean(np.square(p-valid_y))),

        "mean_bridge_positive":float(np.mean(np.asarray(bridges)[true])) if bridges else None,
        "mean_bridge_negative":float(np.mean(np.asarray(bridges)[~true])) if bridges else None,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--train-samples",type=int,default=128)
    p.add_argument("--val-samples",type=int,default=64)
    p.add_argument("--seeds",nargs="+",type=int,default=[17,41])
    p.add_argument("--kinds",nargs="+",choices=("PAIR","JUNCTION"),default=["PAIR","JUNCTION"])
    p.add_argument("--device",default="cpu")
    p.add_argument("--intensity-control",choices=("raw","rank"),default="raw")
    p.add_argument("--separate-endpoints",action="store_true")
    p.add_argument("--close-junction",action="store_true")
    a=p.parse_args()
    if a.out.exists() or a.epochs<1 or a.train_samples<16 or a.val_samples<16:
        raise ValueError("Invalid benchmark request or existing output")
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    torch.set_num_threads(2)
    models=(("conv3d","anza3d","conv3d_structure","anza3d_structure",
             "conv3d_sep_structure","anza3d_sep_structure","anza3d_noaxial_sep_structure")
            if a.separate_endpoints else
            ("conv3d","anza3d","conv3d_structure","anza3d_structure","anza3d_noaxial_structure"))
    measurements=[]
    oracle={}
    shortcut={}
    for kind in a.kinds:
        train=make_phantoms(a.train_samples,20261111,kind,
                            separate_endpoints=a.separate_endpoints,close_junction=a.close_junction)
        validation=make_phantoms(a.val_samples,20261911,kind,
                                 separate_endpoints=a.separate_endpoints,close_junction=a.close_junction)
        if a.intensity_control=="rank":
            train=(normalize_synthetic_hu_histograms(train[0]),train[1],train[2])
            validation=(normalize_synthetic_hu_histograms(validation[0]),validation[1],validation[2])
        shortcut[kind]=simple_shortcut_audit(train,validation)
        print(json.dumps({"simple_shortcuts":kind,"features":shortcut[kind]}),flush=True)
        # Upper bound: run the geometric bridge readout on EXACT VESSEL TRUTH.
        # If even oracle vessel masks do not separate classes, no neural
        # encoder can repair the structural geometry by learning better HU.
        degree=2 if kind=="PAIR" else 3
        with torch.no_grad():
            oracle_p=VesselBridgeReadout(
                degree,junction_strategy="adaptive_star"
            )(
                torch.from_numpy(validation[2][:,0]),
                torch.from_numpy(validation[0][:,1:] if a.separate_endpoints else validation[0][:,1]),
            ).numpy()
        oracle[kind]={
            "AUROC":float(roc_auc_score(validation[1],oracle_p)),
            "AUPRC":float(average_precision_score(validation[1],oracle_p)),
            "positive_mean":float(oracle_p[validation[1]==1].mean()),
            "negative_mean":float(oracle_p[validation[1]==0].mean()),
        }
        # Independent SciPy 26-connected-components diagnostic on the perfect
        # vessel reference. Distinguishes a bridge-readout failure from a
        # negative sample whose separate branch IDs are physically connected.
        peaks,valid=((_peak_centers_from_channels(torch.from_numpy(validation[0][:,1:]),degree))
                     if a.separate_endpoints else
                     _peak_centers(torch.from_numpy(validation[0][:,1]),degree))
        source_points=peaks.numpy().round().astype(int)
        connected=[]
        for idx in range(len(validation[1])):
            component,_=cc_label(validation[2][idx,0]>0,np.ones((3,3,3),dtype=np.uint8))
            sampled=[int(component[tuple(v)]) for v in source_points[idx]]
            connected.append(bool(valid[idx]) and all(x>0 and x==sampled[0] for x in sampled))
        connected=np.asarray(connected,dtype=bool)
        oracle[kind]["GT_component_connected_fraction_positive"]=float(connected[validation[1]==1].mean())
        oracle[kind]["GT_component_connected_fraction_negative"]=float(connected[validation[1]==0].mean())
        # Reject a benchmark with contradictory labels instead of reporting
        # favorable AUROC from images whose true topology disagrees with y.
        if (oracle[kind]["GT_component_connected_fraction_positive"]!=1.0 or
            oracle[kind]["GT_component_connected_fraction_negative"]!=0.0):
            raise AssertionError(
                f"Invalid {kind} benchmark: binary GT components disagree with "
                f"labels; positive={oracle[kind]['GT_component_connected_fraction_positive']}, "
                f"negative={oracle[kind]['GT_component_connected_fraction_negative']}"
            )
        print(json.dumps({"oracle":kind,**oracle[kind]}),flush=True)
        for seed in a.seeds:
            for variant in models:
                result=fit(variant,kind,train,validation,epochs=a.epochs,
                           batch_size=8,seed=seed,device=a.device)
                measurements.append(result)
                print(json.dumps(result,ensure_ascii=False),flush=True)
    report={
        "status":"synthetic_topology_diagnostic_only",
        "warning":"No real ImageCAS CCTA or clinical labels; no validation tuning or ANZA superiority claim.",
        "generated_train_seed":20261111,"generated_val_seed":20261911,
        "train_size":a.train_samples,"val_size":a.val_samples,
        "epochs":a.epochs,"kinds":a.kinds,"seeds":a.seeds,"models":list(models),
        "synthetic_intensity_control":a.intensity_control,
        "synthetic_endpoint_encoding":"separate" if a.separate_endpoints else "summed",
        "close_junction_stress":bool(a.close_junction),
        "heldout_CT28_test_accessed":False,
        "measurements":measurements,
        "oracle_geometry":oracle,
        "simple_shortcuts":shortcut,
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":report["status"],"n_runs":len(measurements),
                      "output":str(a.out)},ensure_ascii=False))


if __name__=="__main__":
    main()
