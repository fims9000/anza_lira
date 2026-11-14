"""V33 Thebe expert TRAIN/segmentation-CAL matched AZConv-v1 vs U-Net.

Different real field than CRACKS. No historical weights; no Thebe TEST.
TRAIN1: 100 sections; VAL1: first 30 section only for topology threshold;
VAL1 30-39 buffer; VAL1 40-99 locked for later P0 relation calibration.

Use original repository model AZConv2d (anza_v1) and standard conv U-Net
with matching capacity, same fixed patches and train budget. Fault bool
masks are expert-marked pixels, NOT complete geological negatives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from models.segmentation_v2 import build_comparable_model
from scripts.cracks_real_oracle_gap_audit import paths_from_mask
from scripts.tver_v26_train_real_segmenters import predict_patch
from scripts.tver_v26_predicted_cracks_e2e import (
    _world,sbpp_proposals,is_true_relation,ROI,
)
from scripts.tver_v28_topology_calibration import (
    choose,TOPOLOGY_THRESHOLDS,
)

TRAIN_STEPS=600
TRAIN_SAMPLE_SEED=20263377
PARAM_WIDTHS=(4,8,12,16)
CAL_PATCHES=96
MIN_SOURCE_LENGTH=80
N_GAPS_PER_SECTION=2
N_TOPOLOGY_TRAIN_SOURCE_SECTIONS=20
N_TOPOLOGY_CAL_SOURCE_SECTIONS=10

def read_memmap(root,kind):
    p=Path(root)/kind
    if not p.is_file():raise RuntimeError("STOP Thebe V33 converted source absent")
    x=np.load(p,mmap_mode="r",allow_pickle=False)
    if x.shape!=(100,3174,1537):
        raise RuntimeError("STOP Thebe V33 section-major source dimensions")
    return x


def amplitude_scale(train_x):
    rng=np.random.default_rng(TRAIN_SAMPLE_SEED)
    samples=[]
    for sid in range(100):
        y=rng.integers(0,3174,size=1024)
        x=rng.integers(0,1537,size=1024)
        samples.append(np.asarray(train_x[sid,y,x],np.float32))
    value=np.concatenate(samples)
    if not np.isfinite(value).all():
        raise RuntimeError("STOP invalid TRAIN seismic amplitude")
    floor,ceil=np.quantile(value,[.01,.99]).tolist()
    if ceil-floor<1e-6:raise RuntimeError("STOP degenerate TRAIN seismic dynamic range")
    norm=np.clip((value-floor)/(ceil-floor),0,1)
    return {
        "TRAIN_only_1pct":float(floor),"TRAIN_only_99pct":float(ceil),
        "normalized_channel_mean":float(np.mean(norm)),
        "normalized_channel_std":float(max(np.std(norm),.03)),
        "sample_seed":TRAIN_SAMPLE_SEED,
        "samples_per_train_section":1024,
        "uncalibrated_seismic_float64_converted_float32":True,
    }


def _scaled_gray(raw,scale):
    x=np.asarray(raw,np.float32)
    return np.ascontiguousarray(np.clip(
        (x-scale["TRAIN_only_1pct"])/
        (scale["TRAIN_only_99pct"]-scale["TRAIN_only_1pct"]),0,1),
        np.float32)


def _rgb(raw,scale):
    x=_scaled_gray(raw,scale)
    return np.repeat(x[:,:,None],3,axis=2)


def _safe_patch(x,sid,y,x0,half=32):
    # Tiny sample only: 64x64 raw source mmap to avoid a 3.9 GB load.
    y0=int(np.clip(y-half,0,x.shape[1]-2*half))
    left=int(np.clip(x0-half,0,x.shape[2]-2*half))
    return x[sid,y0:y0+2*half,left:left+2*half]


def _make_patches(seismic,label,scale,*,count,seed,selected_sections,positive_fraction):
    rng=np.random.default_rng(seed)
    ids=list(selected_sections)
    if not ids or count<len(ids):
        raise RuntimeError("STOP no valid patch source section IDs")
    available={}
    for sid in ids:
        positive=np.flatnonzero(label[sid])
        if len(positive)==0:
            raise RuntimeError(f"STOP Thebe source section {sid} contains no fault pixels")
        available[sid]=positive
    patches=[];stats={"positive_center_count":0,"negative_center_count":0}
    width=label.shape[2]
    for k in range(count):
        sid=ids[k%len(ids)]
        mask=label[sid]
        positive=rng.random()<positive_fraction
        if positive:
            flat=int(rng.choice(available[sid]))
            cy,cx=divmod(flat,width)
            cy+=int(rng.integers(-10,11))
            cx+=int(rng.integers(-10,11))
        else:
            for attempt in range(70):
                cy=int(rng.integers(32,mask.shape[0]-32))
                cx=int(rng.integers(32,mask.shape[1]-32))
                if not np.any(mask[cy-12:cy+13,cx-12:cx+13]):
                    break
            else:
                raise RuntimeError("STOP could not sample unmarked train patch")
        patch=_safe_patch(seismic,sid,cy,cx)
        truth=np.asarray(_safe_patch(label,sid,cy,cx),np.float32)
        observed=_rgb(patch,scale)
        valid=np.ones_like(truth,np.float32)
        # There is no guaranteed natural nonfault negative here.
        pseudo=np.zeros_like(truth,np.float32)
        if rng.random()<.5:
            observed=observed[::-1];truth=truth[::-1]
        if rng.random()<.5:
            observed=observed[:,::-1];truth=truth[:,::-1]
        patches.append((
            np.ascontiguousarray(observed),
            np.ascontiguousarray(truth),
            valid,
            pseudo))
        stats["positive_center_count" if positive else "negative_center_count"]+=1
    return patches,stats


def _convert_training_image(image,scale):
    mean=scale["normalized_channel_mean"]
    std=scale["normalized_channel_std"]
    return torch.from_numpy(np.ascontiguousarray(
        ((image-mean)/std).transpose(2,0,1)[None],np.float32))


def _fit_model(name,seed,patches,calpatches,scale,out):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=build_comparable_model(name,widths=PARAM_WIDTHS)
    optim=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    model.train();loss_history=[]
    for k,(x,y,valid,pseudo) in enumerate(patches):
        tx=_convert_training_image(x,scale)
        ty=torch.from_numpy(np.ascontiguousarray(y[None,None]))
        v=torch.ones_like(ty)
        logits=model(tx)
        weights=(1.+14.*ty)*v
        bce=(F.binary_cross_entropy_with_logits(logits,ty,reduction="none")
             *weights).sum()/weights.sum().clamp_min(1.)
        proba=torch.sigmoid(logits)
        dice=1.-(2*(proba*ty).sum()+1.)/((proba*v).sum()+(ty*v).sum()+1.)
        loss=bce+.2*dice
        if not torch.isfinite(loss):raise RuntimeError("STOP nonfinite real Thebe seg train loss")
        optim.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        optim.step();loss_history.append(float(loss.item()))
        if (k+1)%150==0:
            print("THEBE_SEG_FIT",name,"step",k+1,
                  "last150loss",round(np.mean(loss_history[-150:]),5),flush=True)
    model.eval()
    with torch.inference_mode():
        predicted=np.stack([
            predict_patch(
                model,image,
                [scale["normalized_channel_mean"]]*3,
                [scale["normalized_channel_std"]]*3)
            for image,*_ in calpatches])
    labels=np.stack([y for _,y,_,_ in calpatches])
    scores={}
    for threshold in TOPOLOGY_THRESHOLDS:
        binary=predicted>=threshold
        tp=int(np.logical_and(binary,labels>.5).sum())
        fp=int(np.logical_and(binary,labels<=.5).sum())
        fn=int(np.logical_and(~binary,labels>.5).sum())
        scores[str(threshold)]={
            "tp":tp,"fp_unmarked":fp,"fn":fn,
            "mask_f1_descriptive":2*tp/max(2*tp+fp+fn,1)}
    path=out/(name+"_new_thebe_checkpoint.pt")
    torch.save({
        "model":model.state_dict(),
        "architecture":name,
        "widths":PARAM_WIDTHS,
        "train_mean":[scale["normalized_channel_mean"]]*3,
        "train_std":[scale["normalized_channel_std"]]*3,
        "original_historical_pretrained_checkpoint_used":False,
        "thebe_real_expert_train_steps":len(patches),
    },path)
    return model,{
        "checkpoint":path.name,
        "SHA256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_params":sum(p.numel() for p in model.parameters()),
        "train_loss_first150":float(np.mean(loss_history[:150])),
        "train_loss_last150":float(np.mean(loss_history[-150:])),
        "same_patches_as_other_frontend":True,
        "original_CRACKS_weights_NOT_imported":True,
        "train_steps":len(patches),
        "pixel_CAL_on_unmarked_background_not_geological_precision":scores,
    }


def _gaps(mask,section_id):
    """From expert label only choose synthetic intervention and true scoring."""
    paths=sorted(
        (np.asarray(p,dtype=np.int32) for p in paths_from_mask(mask)
         if len(p)>=MIN_SOURCE_LENGTH),
        key=lambda p:(-len(p),int(p[0,0]),int(p[0,1])))
    out=[]
    for path in paths:
        i=len(path)//2
        hidden=path[i-6:i+6]
        if len(hidden)!=12:continue
        y0=int(hidden[:,0].min()-3);y1=int(hidden[:,0].max()+4)
        x0=int(hidden[:,1].min()-3);x1=int(hidden[:,1].max()+4)
        h,w=mask.shape
        if y0<9 or x0<9 or y1>=h-9 or x1>=w-9:continue
        inside=lambda p:y0<=p[0]<y1 and x0<=p[1]<x1
        a=next((p for p in path[:i][::-1] if not inside(p)),None)
        b=next((p for p in path[i:] if not inside(p)),None)
        if a is None or b is None or np.linalg.norm(a-b)<6:continue
        out.append({
            "id":f"thebe:{section_id}:{len(out)}",
            "section":section_id,"rectangle":(y0,y1,x0,x1),
            "true_ends":(tuple(map(float,a)),tuple(map(float,b))),
            "hidden_centerline":[tuple(map(int,z)) for z in hidden],
        })
        if len(out)>=N_GAPS_PER_SECTION:break
    return out


def _roi_gray(raw,rect,scale):
    h,w=raw.shape
    y0,y1,x0,x1=rect
    cy=(y0+y1)//2;cx=(x0+x1)//2
    oy=int(np.clip(cy-ROI//2,0,h-ROI))
    ox=int(np.clip(cx-ROI//2,0,w-ROI))
    patch=np.asarray(raw[oy:oy+ROI,ox:ox+ROI],np.float32)
    if patch.shape!=(ROI,ROI):
        raise RuntimeError("STOP bad Thebe ROI")
    return _rgb(patch,scale),(oy,ox),(y0-oy,y1-oy,x0-ox,x1-ox)


def _precompute_cases(seismic,labels,sids,scale,model):
    result=[]
    for sid in sids:
        for gap in _gaps(np.asarray(labels[sid],bool),sid):
            rgb,origin,rect=_roi_gray(seismic[sid],gap["rectangle"],scale)
            prob=predict_patch(
                model,rgb,
                [scale["normalized_channel_mean"]]*3,
                [scale["normalized_channel_std"]]*3)
            result.append({
                "id":gap["id"],"origin":origin,"rect":rect,
                "ends":gap["true_ends"],"prob":prob,
                "gray":rgb.mean(axis=2),
            })
    return result


def _threshold_result(cases,threshold):
    records=[];sources=0;ncands=ntrue=0;present=0;dense=0
    for case in cases:
        visible=_world(case["prob"],case["rect"],case["ends"],
                       case["origin"],"LINK")
        pred=sbpp_proposals(visible,case["gray"],case["rect"],threshold)
        true=sum(is_true_relation(p,case["ends"],case["origin"])
                 for p in pred["proposals"])
        sources+=int(pred["source_count"]>0)
        ncands+=len(pred["proposals"]);ntrue+=true
        present+=int(true>0)
        dense+=int(pred["status"]=="DEGENERATE_DENSE_PREDICTED_MASK")
        records.append({"id":case["id"],"status":pred["status"],
                        "true_candidates":true,
                        "proposed_candidates":len(pred["proposals"])})
    return {
        "n_controlled_gaps":len(cases),
        "n_sources":sources,"n_candidates":ncands,
        "n_true_pairs":ntrue,
        "n_true_pair_gaps":present,
        "n_false_pairs":ncands-ntrue,
        "n_dense_rejected":dense,
        "case_rows":records,
    }


def execute(root,out):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    train_x=read_memmap(root,"train_seismic_float32.npy")
    train_y=read_memmap(root,"train_fault_bool.npy")
    val_x=read_memmap(root,"val_seismic_float32.npy")
    val_y=read_memmap(root,"val_fault_bool.npy")
    if train_x.dtype!=np.float32 or val_x.dtype!=np.float32:
        raise RuntimeError("STOP type mismatch from verified Thebe source")
    if train_y.dtype!=np.bool_ or val_y.dtype!=np.bool_:
        raise RuntimeError("STOP expert labels no longer Boolean")
    scale=amplitude_scale(train_x)
    patches,train_counts=_make_patches(
        train_x,train_y,scale,count=TRAIN_STEPS,
        seed=TRAIN_SAMPLE_SEED+1,selected_sections=range(100),
        positive_fraction=.72)
    calpatches,cal_counts=_make_patches(
        val_x,val_y,scale,count=CAL_PATCHES,
        seed=TRAIN_SAMPLE_SEED+2,selected_sections=range(30),
        positive_fraction=.50)
    report={
        "status":"V33_THEBE_SEG_TRAIN_AND_TOPOLOGY_CAL_STARTED",
        "source":"original Thebe DATAVERSE expert masks",
        "real_TRAIN_sections_count":100,
        "SEG_CAL_section_indices":list(range(30)),
        "excluded_valid_indices":list(range(30,40)),
        "REL_CAL_section_indices_locked_unopened":list(range(40,100)),
        "heldout_THEBE_TEST7_NOT_OPENED":True,
        "training_steps_each":TRAIN_STEPS,"patch_counts":train_counts,
        "CAL_pixel_patches":CAL_PATCHES,
        "normalization":scale,
        "threshold_grid":TOPOLOGY_THRESHOLDS,
        "frontends":{},
    }
    topology_train_ids=list(range(0,100,5))
    topology_cal_ids=list(range(0,30,3))
    for name,seed in (("unet",20263312),("anza_v1",20263311)):
        model,receipt=_fit_model(name,seed,patches,calpatches,scale,out)
        train_cases=_precompute_cases(
            train_x,train_y,topology_train_ids,scale,model)
        cal_cases=_precompute_cases(
            val_x,val_y,topology_cal_ids,scale,model)
        tr={};ca={}
        for q in TOPOLOGY_THRESHOLDS:
            tr[str(q)]=_threshold_result(train_cases,q)
            ca[str(q)]=_threshold_result(cal_cases,q)
        chosen=choose(tr,ca)
        report["frontends"][name]={
            **receipt,"topology_threshold_frozen":chosen,
            "topology_status":"TRAIN_CAL_FEASIBLE" if chosen is not None else
                "STOP_NO_TRAIN_CAL_CORRECT_PREDICTED_SBPP",
            "train_topology_source_indices":topology_train_ids,
            "cal_topology_source_indices":topology_cal_ids,
            "train_cases":len(train_cases),"cal_cases":len(cal_cases),
            "TRAIN_by_threshold":tr,"SEG_CAL_by_threshold":ca,
        }
    feasible=all(
        x["topology_threshold_frozen"] is not None
        for x in report["frontends"].values())
    report["status"]=(
        "V33_THEBE_REAL_SEGMENTERS_AND_TOPOLOGY_TRAIN_CAL_COMPLETE"
        if feasible else
        "V33_STOP_THEBE_NEW_FRONTEND_PREDICTED_SBPP_INSUFFICIENT")
    report["next"]="Fit independent 5-conv P0 and matched logistic on actual Thebe predicted TRAIN pairs, CAL risk on VAL40..99, then TEST7 in another frozen run" if feasible else "Do not open TEST; diagnose front-end under Thebe domain"
    path=out/"v33_THEBE_frontends_topology_TRAIN_CAL_frozen.json"
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "status":report["status"],
        "frontends":{name:{
            "weight_SHA":data["SHA256"],
            "selected_topology":data["topology_threshold_frozen"],
            "train_cases":data["train_cases"],
            "cal_cases":data["cal_cases"],
            "CAL_selected":data["SEG_CAL_by_threshold"].get(
                str(data["topology_threshold_frozen"]),{}).get(
                    "n_true_pair_gaps",0),
        } for name,data in report["frontends"].items()},
    },indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mmap-root",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    execute(a.mmap_root,a.out_dir)
