"""V55 frozen comparison: whole-section 2-D vs 2.5-D fault segmentation.

All model training uses official Thebe TRAIN1; full-section evaluation uses
TRAIN1 and already-inspected VAL1. Never read Thebe TEST7.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from models.segmentation_v2 import build_comparable_model
from scripts.tver_v33_thebe_train_segmenters import read_memmap
from scripts.tver_v34_anza_p0_relation_cal import _state_report
from scripts.tver_v54_full_section_infer import tile_starts
from trace_extraction.coverage import annotated_trace_coverage

SEED=20261122
WIDTHS=(8,16,32,48)
TRAIN_STEPS=800
TRAIN_BATCH=6
PATCH=64
TILE=96
STRIDE=64
VALIDATION=(("TRAIN1",20),("VAL1_ALREADY_SEEN",43),("VAL1_ALREADY_SEEN",77))
PIXEL_BUDGETS=(.01,.02,.05,.10)


def _normalize(x,scale):
    lo=float(scale["TRAIN_only_1pct"])
    hi=float(scale["TRAIN_only_99pct"])
    mean=float(scale["normalized_channel_mean"])
    std=float(scale["normalized_channel_std"])
    if not (np.isfinite([lo,hi,mean,std]).all() and lo<hi and std>0):
        raise ValueError("Invalid frozen TRAIN-only normalization")
    return ((np.clip((np.asarray(x,np.float32)-lo)/(hi-lo),0,1)-mean)/std).astype(np.float32)


def _patch(scene,labels,sid,cy,cx,mode,scale,size=PATCH):
    h,w=labels.shape[1:]
    y=int(np.clip(cy-size//2,0,h-size))
    x=int(np.clip(cx-size//2,0,w-size))
    ids=(sid,sid,sid) if mode=="2d" else (sid-1,sid,sid+1)
    if mode not in ("2d","2p5d") or min(ids)<0 or max(ids)>=len(scene):
        raise ValueError("Invalid source/neighbor arrangement")
    fields=np.stack([scene[z,y:y+size,x:x+size] for z in ids])
    target=np.asarray(labels[sid,y:y+size,x:x+size],np.float32)
    return _normalize(fields,scale),target.copy()


def _draw_batch(images,masks,scale,rng,mode,batch_size,positives):
    count=len(images)
    if count<3 or masks.shape!=images.shape:
        raise ValueError("Requires aligned consecutive TRAIN volumes and labels")
    h,w=images.shape[1:]
    batch_x=np.zeros((batch_size,3,PATCH,PATCH),np.float32)
    batch_y=np.zeros((batch_size,1,PATCH,PATCH),np.float32)
    for i in range(batch_size):
        sid=int(rng.integers(1,count-1))
        if rng.random()<.65:
            if sid not in positives:
                positives[sid]=np.flatnonzero(np.asarray(masks[sid],bool))
            ids=positives[sid]
            if not len(ids):
                raise RuntimeError("TRAIN section contains no expert positive pixels")
            flat=int(ids[rng.integers(len(ids))])
            cy,cx=divmod(flat,w)
            cy+=int(rng.integers(-8,9));cx+=int(rng.integers(-8,9))
        else:
            # Pseudo-negative centers must be distant from annotated lines;
            # unmarked pixels are not claimed to be verified geological negatives.
            for _ in range(40):
                cy=int(rng.integers(PATCH//2,h-PATCH//2))
                cx=int(rng.integers(PATCH//2,w-PATCH//2))
                if not np.any(masks[sid,cy-12:cy+13,cx-12:cx+13]):
                    break
        x,y=_patch(images,masks,sid,cy,cx,mode,scale)
        if rng.random()<.5:
            x=x[:,:,::-1];y=y[:,::-1]
        if rng.random()<.5:
            x=x[:,::-1];y=y[::-1]
        batch_x[i]=np.ascontiguousarray(x)
        batch_y[i,0]=np.ascontiguousarray(y)
    return batch_x,batch_y


def train_model(train_images,train_labels,scale,mode,*,steps=TRAIN_STEPS,
                batch_size=TRAIN_BATCH,seed=SEED):
    import torch
    import torch.nn.functional as F

    if mode not in ("2d","2p5d") or steps<1 or batch_size<1:
        raise ValueError("Invalid frozen model training schedule")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng=np.random.default_rng(seed)
    net=build_comparable_model("unet",widths=WIDTHS)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    net.train()
    positives={}
    losses=[]
    for k in range(steps):
        inputs,truth=_draw_batch(
            train_images,train_labels,scale,rng,mode,batch_size,positives)
        x=torch.from_numpy(inputs)
        y=torch.from_numpy(truth)
        logits=net(x)
        if logits.shape!=y.shape:
            raise RuntimeError("V55 unexpected train logits shape")
        weights=1.+14.*y
        bce=(F.binary_cross_entropy_with_logits(logits,y,reduction="none")*
             weights).sum()/weights.sum().clamp_min(1.)
        prob=torch.sigmoid(logits)
        dice=1.-(2*(prob*y).sum()+1.)/((prob+y).sum()+1.)
        loss=bce+.2*dice
        if not torch.isfinite(loss):
            raise RuntimeError("V55 nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (k+1)%200==0:
            print("V55_TRAIN",mode,"step",k+1,
                  "last100loss",float(np.mean(losses[-100:])),flush=True)
    net.eval()
    return net,{
        "mode":mode,"seed":seed,"steps":steps,"batch_size":batch_size,
        "architecture":"unet","widths":list(WIDTHS),
        "n_trainable_parameters":sum(p.numel() for p in net.parameters()),
        "train_patch_size":PATCH,"mean_last100_training_loss":
            float(np.mean(losses[-100:])),
        "source_sections_exclude_neighbor_split_boundary":True,
        "no_CAL_used_for_training_or_checkpoint_selection":True,
    }


def predict_complete(source,model,scale,mode,*,tile=TILE,stride=STRIDE,batch=8):
    """Overlap-add a three-channel image stack, never expert annotations."""
    import torch

    data=np.asarray(source)
    if data.ndim!=3 or data.shape[0]!=3 or min(data.shape[1:])<tile or not np.isfinite(data).all():
        raise ValueError("Expect three aligned finite 2-D seismic source channels")
    if mode not in ("2d","2p5d"):
        raise ValueError("Unknown model input mode")
    height,width=data.shape[1:]
    ys=tile_starts(height,tile,stride)
    xs=tile_starts(width,tile,stride)
    coords=[(y,x) for y in ys for x in xs]
    numerator=np.zeros((height,width),np.float32)
    denominator=np.zeros((height,width),np.float32)
    wave=np.hanning(tile)
    blend=np.maximum(np.outer(wave,wave),.05).astype(np.float32)
    model.eval()
    with torch.inference_mode():
        for offset in range(0,len(coords),batch):
            current=coords[offset:offset+batch]
            samples=np.stack([
                _normalize(data[:,y:y+tile,x:x+tile],scale)
                for y,x in current
            ])
            logits=model(torch.from_numpy(samples))
            if logits.shape!=(len(current),1,tile,tile):
                raise RuntimeError("V55 unexpected full-section logits shape")
            score=torch.sigmoid(logits).cpu().numpy()[:,0]
            if not np.isfinite(score).all():
                raise RuntimeError("V55 nonfinite full-section probability")
            for (y,x),p in zip(current,score):
                numerator[y:y+tile,x:x+tile]+=p*blend
                denominator[y:y+tile,x:x+tile]+=blend
    if np.any(denominator<=0):
        raise RuntimeError("V55 tile seams left uncovered pixels")
    return numerator/denominator


def _visual_preview(prob,path):
    from PIL import Image
    factor=max(1,math.ceil(max(prob.shape)/1500))
    p=np.asarray(prob[::factor,::factor],np.float32)
    # Equal-area colors, chosen from prediction alone, never reference.
    high,mid,low=np.quantile(p,(.99,.95,.80))
    span=max(float(high-low),1e-4)
    g=(np.clip((p-low)/span,0,1)*130).astype(np.uint8)
    rgb=np.repeat(g[...,None],3,axis=2)
    rgb[p>=low]=[98,52,150]
    rgb[p>=mid]=[230,166,38]
    rgb[p>=high]=[20,218,88]
    Image.fromarray(rgb).save(path)
    return int(factor)


def exact_top_fraction_mask(probability,fraction):
    """Deterministic exact-pixel budget, even for tied/quantized model scores."""
    p=np.asarray(probability,np.float32)
    if (p.ndim!=2 or not np.isfinite(p).all()
            or not 0<fraction<=1):
        raise ValueError("Invalid exact positive area budget")
    count=max(1,int(round(p.size*fraction)))
    flat=p.ravel()
    cutoff=float(np.partition(flat,flat.size-count)[flat.size-count])
    chosen=(flat>cutoff)
    needed=count-int(chosen.sum())
    if needed:
        ties=np.flatnonzero(flat==cutoff)
        if len(ties)<needed:
            raise RuntimeError("Unstable exact-area ranking")
        chosen[ties[:needed]]=True
    result=chosen.reshape(p.shape)
    if int(result.sum())!=count:
        raise RuntimeError("Exact-area candidate budget not met")
    return result,cutoff


def _profile(prob,truth):
    output=[]
    for budget in PIXEL_BUDGETS:
        mask,cutoff=exact_top_fraction_mask(prob,budget)
        row=annotated_trace_coverage(mask.astype(np.float32),truth,(0.5,))[0]
        row.pop("threshold",None)
        output.append({
            "target_foreground_budget":budget,
            "actual_foreground_budget":float(mask.mean()),
            "model_score_cutoff":cutoff,
            "tie_handling":"row_major_deterministic_exact_area",
            **row,
        })
    return output


def run(mmap_root,frontends_dir,out_dir):
    import torch

    out=Path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    state=_state_report(frontends_dir)
    if not state["heldout_THEBE_TEST7_NOT_OPENED"]:
        raise RuntimeError("Source authenticity/TEST7 guard failed")
    scale=state["normalization"]
    train_x=read_memmap(mmap_root,"train_seismic_float32.npy")
    train_y=read_memmap(mmap_root,"train_fault_bool.npy")
    models={}
    receipts={}
    for mode in ("2d","2p5d"):
        model,receipt=train_model(train_x,train_y,scale,mode)
        path=out/f"V55_{mode}_train_only.pt"
        torch.save({
            "mode":mode,"seed":SEED,"architecture":"unet",
            "widths":WIDTHS,"model":model.state_dict(),
            "normalization":scale,"steps":TRAIN_STEPS,
        },path)
        receipt["model_SHA256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        models[mode]=model
        receipts[mode]=receipt
        print("V55_MODEL_FITTED",mode,json.dumps(receipt),flush=True)
    del train_x,train_y
    data={}
    reports=[]
    for cohort,sid in VALIDATION:
        dataset="train" if cohort=="TRAIN1" else "val"
        if dataset not in data:
            data[dataset]=(
                read_memmap(mmap_root,dataset+"_seismic_float32.npy"),
                read_memmap(mmap_root,dataset+"_fault_bool.npy"),
            )
        sources,reference=data[dataset]
        section_out=out/f"{cohort}_{sid}"
        section_out.mkdir(exist_ok=True)
        truth=np.array(reference[sid],dtype=bool,order="C",copy=True)
        for mode in ("2d","2p5d"):
            ids=(sid,sid,sid) if mode=="2d" else (sid-1,sid,sid+1)
            stack=np.stack([sources[i] for i in ids])
            prob=predict_complete(stack,models[mode],scale,mode)
            del stack
            np.save(section_out/f"{mode}_probability.npy",prob,allow_pickle=False)
            figure_scale=_visual_preview(prob,section_out/f"{mode}_review_preview.png")
            profile=_profile(prob,truth)
            report={
                "cohort":cohort,"section":sid,"mode":mode,
                "source_indices":list(ids),
                "mean_prediction":float(prob.mean()),
                "std_prediction":float(prob.std()),
                "preview_downsample_factor":figure_scale,
                "equal_area_budget_profile":profile,
                "checkpoint_SHA256":receipts[mode]["model_SHA256"],
                "no_TEST7":True,
                "label_used_only_after_complete_prediction":True,
            }
            reports.append(report)
            print("V55_SECTION_COMPARE",cohort,sid,mode,
                  json.dumps(profile),flush=True)
    summary={
        "status":"V55_MATCHED_2D_2P5D_FULLSECTION_EXPLORATORY",
        "train_schedule_frozen":dict(steps=TRAIN_STEPS,
            batch_size=TRAIN_BATCH,patch_size=PATCH,seed=SEED,
            architecture="unet",widths=WIDTHS),
        "training_receipts":receipts,
        "section_reports":reports,
        "no_TEST7_source_or_labels":True,
        "partial_expert_labels_not_complete_negatives":True,
        "no_independent_field_generalization_claim":True,
        "no_deployment_threshold_fitted_on_VAL":True,
    }
    (out/"V55_RESULTS.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    return summary


def main():
    p=argparse.ArgumentParser()
    for arg in ("mmap-root","frontends-dir","out-dir"):
        p.add_argument("--"+arg,type=Path,required=True)
    opts=p.parse_args()
    run(opts.mmap_root,opts.frontends_dir,opts.out_dir)


if __name__=="__main__":
    main()
