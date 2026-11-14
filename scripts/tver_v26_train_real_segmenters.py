"""V26 actual CRACKS expert-TRAIN image segmentation: new genuine AZConv and U-Net.

Fitted from scratch on same 64x64 real seismic patches and original expert
TRAIN annotations; no historical weights are available. Unknown white
pixels are not ground-truth negatives. White >12px from annotated BLUE/GREEN
is a declared *assumed/pseudo-negative* solely for pilot model training.
CAL-only threshold before ever opening TEST pixels/labels.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
import torch.nn.functional as F

from scripts.cracks_real_oracle_gap_audit import sections,COLORS
from scripts.tver_v26_cracks_inventory import split_map
from models.segmentation_v2 import build_comparable_model

S=64
STEPS=120
WIDTHS=(4,8,12,16)
MODELS=(("unet",20262612),("anza_v1",20262611))
THRESHOLDS=(.2,.3,.4,.5,.6,.7)


def _rgb(archive,path):
    with Image.open(io.BytesIO(archive.read(path))) as im:
        return np.asarray(im.convert("RGB"),np.uint8)


def _read(zi,za,images,expert,sid):
    image=_rgb(zi,images[sid]).astype(np.float32)/255.
    rgb=_rgb(za,expert[sid])
    blue=np.all(rgb==COLORS["blue"],axis=2)
    green=np.all(rgb==COLORS["green"],axis=2)
    orange=np.all(rgb==COLORS["orange"],axis=2)
    white=np.all(rgb==COLORS["white"],axis=2)
    if not np.all(blue|green|orange|white):
        raise RuntimeError("STOP unknown annotation RGB")
    pseudo=white&~ndi.binary_dilation(blue|green,iterations=12)
    target=blue.astype(np.float32)
    valid=(blue|orange|pseudo).astype(np.float32)
    known=blue|orange
    return {"image":image,"mask":rgb,"blue":blue,
            "green":green,"orange":orange,"white":white,
            "target":target,"valid":valid,"pseudo":pseudo,"known":known}


def _patch(array,cy,cx,size=S):
    h,w=array.shape[:2];half=size//2
    y0=max(0,min(h-size,int(cy)-half))
    x0=max(0,min(w-size,int(cx)-half))
    if h>=size and w>=size:
        return array[y0:y0+size,x0:x0+size]
    padding=((0,max(0,size-h)),(0,max(0,size-w)))
    if array.ndim==3:padding+=( (0,0), )
    return np.pad(array,padding,mode="reflect")[:size,:size]


def _samples(data,*,count,seed,pos_fraction=.72):
    rng=np.random.default_rng(seed)
    ids=sorted(data)
    samples=[];counters=Counter()
    for i in range(count):
        sid=ids[i%len(ids)]
        source=data[sid]
        use_positive=(rng.random()<pos_fraction)
        candidates=np.argwhere(
            source["blue"] if use_positive else source["pseudo"])
        if len(candidates)==0:
            candidates=np.argwhere(source["valid"]>0)
        if len(candidates)==0:continue
        cy,cx=map(int,candidates[int(rng.integers(len(candidates)))])
        cy+=int(rng.integers(-12,13))
        cx+=int(rng.integers(-12,13))
        image=_patch(source["image"],cy,cx)
        label=_patch(source["target"],cy,cx)
        valid=_patch(source["valid"],cy,cx)
        pseudo=_patch(source["pseudo"],cy,cx)
        if rng.random()<.5:
            image=image[:,::-1];label=label[:,::-1]
            valid=valid[:,::-1];pseudo=pseudo[:,::-1]
        if rng.random()<.5:
            image=image[::-1];label=label[::-1]
            valid=valid[::-1];pseudo=pseudo[::-1]
        samples.append((image.copy(),label.copy(),
                        valid.copy(),pseudo.copy()))
        counters["positive_pixels"]+=int(label.sum())
        counters["pseudo_negative_pixels"]+=int(pseudo.sum())
        counters["known_or_pseudo_valid_pixels"]+=int(valid.sum())
    if len(samples)!=count:
        raise RuntimeError("STOP insufficient supervised real-CRACKS patches")
    return samples,dict(counters)


def _normalize(image,mean,std):
    return np.ascontiguousarray(
        ((image-np.asarray(mean).reshape(1,1,3))/
         np.asarray(std).reshape(1,1,3)).transpose(2,0,1),np.float32)


def predict_patch(model,image,mean,std):
    x=torch.from_numpy(_normalize(image,mean,std)[None])
    with torch.inference_mode():
        out=torch.sigmoid(model(x)).squeeze().cpu().numpy()
    if out.shape!=image.shape[:2] or not np.isfinite(out).all():
        raise RuntimeError("STOP model probability shape/nonfinite")
    return out.astype(np.float32)


def fit_one(model_name,seed,training,cal_samples,mean,std,out):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=build_comparable_model(model_name,widths=WIDTHS)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    model.train()
    history=[]
    for k,(image,target,valid,pseudo) in enumerate(training):
        x=torch.from_numpy(_normalize(image,mean,std)[None])
        y=torch.from_numpy(np.ascontiguousarray(target[None,None]))
        v=torch.from_numpy(np.ascontiguousarray(valid[None,None]))
        logits=model(x)
        if logits.shape!=y.shape:raise RuntimeError("Segmentation output unexpected shape")
        weights=(1.+14.*y)*v
        bce=(F.binary_cross_entropy_with_logits(logits,y,reduction="none")*
             weights).sum()/weights.sum().clamp_min(1.)
        p=torch.sigmoid(logits)
        dice=1.-(2*(p*y*v).sum()+1.)/((p*v).sum()+(y*v).sum()+1.)
        loss=bce+.20*dice
        if not torch.isfinite(loss):raise RuntimeError("STOP nonfinite segmentation training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        optimizer.step()
        history.append(float(loss.item()))
        if (k+1)%30==0:
            print("phase=V26_REAL_TRAIN",model_name,
                  "step",k+1,"loss",round(float(np.mean(history[-30:])),5),
                  flush=True)
    model.eval()
    with torch.inference_mode():
        preds=np.stack([predict_patch(model,x,mean,std)
                        for x,*_ in cal_samples])
    labels=np.stack([t for _,t,_,_ in cal_samples])
    valid=np.stack([v for _,_,v,_ in cal_samples]).astype(bool)
    known=np.stack([v for _,_,v,_ in cal_samples]) # includes pseudo; disclosed
    scores={}
    for threshold in THRESHOLDS:
        pred=(preds>=threshold)&valid
        tp=int(np.sum(pred&(labels>.5)))
        fp=int(np.sum(pred&(labels<=.5)&valid))
        fn=int(np.sum(~pred&(labels>.5)&valid))
        precision=tp/max(tp+fp,1)
        recall=tp/max(tp+fn,1)
        scores[str(threshold)]={
            "tp":tp,"fp_pseudo_or_orange":fp,"fn":fn,
            "precision_on_assumed_label_mask":precision,
            "recall_on_annotated_blue":recall,
            "f1_on_assumed_label_mask":2*tp/max(2*tp+fp+fn,1)}
    # Fixed CAL objective. Not a guaranteed real-geology segmentation score.
    selected=max(THRESHOLDS,key=lambda q:(
        scores[str(q)]["f1_on_assumed_label_mask"],q))
    path=out/(model_name+"_new_checkpoint.pt")
    torch.save({
        "model":model.state_dict(),
        "architecture":model_name,"widths":WIDTHS,
        "train_mean":mean.tolist(),"train_std":std.tolist(),
        "selected_CAL_threshold":selected,
        "historical_pretrained_checkpoint_used":False,
        "training_steps":STEPS},path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "checkpoint":path.name,"checkpoint_SHA256":digest,
        "train_loss_initial30":float(np.mean(history[:30])),
        "train_loss_final30":float(np.mean(history[-30:])),
        "model_parameters":sum(p.numel() for p in model.parameters()),
        "selected_CAL_threshold":float(selected),
        "CAL_scores":scores,
        "CAL_uses_pseudo_negative_white_unlabeled":True}


def execute(images,annotations,out,*,split_mapping=None):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(images) as zi,zipfile.ZipFile(annotations) as za:
        im=sections(zi);ex=sections(za,expert=True)
        ids=sorted(set(im)&set(ex))
        mapping=split_map(ids) if split_mapping is None else dict(split_mapping)
        if set(mapping).difference(ids):raise RuntimeError('External cohort contains unknown expert IDs')
        # Do not open TEST expert mask or image before TRAIN and CAL fit.
        train={sid:_read(zi,za,im,ex,sid)
               for sid in ids if mapping[sid]=="TRAIN"}
        cal={sid:_read(zi,za,im,ex,sid)
             for sid in ids if mapping[sid]=="CAL"}
    if len(train)<(8 if split_mapping is None else 5) or len(cal)<2:
        raise RuntimeError("STOP insufficient real spatial train/CAL experts")
    allpix=np.concatenate([x["image"].reshape(-1,3)
                           for x in train.values()],axis=0)
    mean=allpix.mean(axis=0)
    std=np.maximum(allpix.std(axis=0),.03)
    del allpix
    patches,counts=_samples(train,count=STEPS,seed=20262677)
    cal_patches,_=_samples(cal,count=48,seed=20262678,pos_fraction=.5)
    receipt={
        "status":"V26_NEW_SEGMENTERS_TRAINED_ON_REAL_CRACKS",
        "V27_source_eligible_BLUE_cohort":bool(split_mapping is not None),
        "seismic_source":"official_Zenodo_CRACKS_v2",
        "TRAIN_section_ids":sorted(train),
        "CAL_section_ids":sorted(cal),
        "TEST_pixel_arrays_or_labels_loaded_before_freeze":False,
        "reserved_TEST_source_sections":[sid for sid in ids if mapping[sid]=="TEST"],
        "buffer_source_sections":[sid for sid in ids if mapping[sid]=="BUFFER"],
        "split_only_source_filenames_before_labels":True,
        "synthetic_continuity_model_pretraining":False,
        "historical_ANZA_checkpoint_recovered":False,
        "partial_BLUE_ORANGE_labels_and_mined_unknown_white":True,
        "white_is_NOT_annotated_ground_truth_negative":True,
        "pseudo_negative_min_distance_from_annotated_trace_px":12,
        "train_patch_counts":counts,
        "train_normalization":{"mean":mean.tolist(),"std":std.tolist()},
        "training_budget":{"steps":STEPS,"patch_hw":[S,S],
                           "batch":1,"widths":WIDTHS,
                           "optimizer":"AdamW_lr_0.002_weight_decay_0.0001"},
        "frontends":{}}
    for name,seed in MODELS:
        receipt["frontends"][name]=fit_one(
            name,seed,patches,cal_patches,mean,std,out)
    (out/"v26_segmenter_frozen.json").write_text(
        json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "status":receipt["status"],
        "train_sections":len(train),"cal_sections":len(cal),
        "models":{name:{"sha":item["checkpoint_SHA256"],
                         "threshold":item["selected_CAL_threshold"],
                         "parameters":item["model_parameters"],
                         "last_loss":item["train_loss_final30"]}
                  for name,item in receipt["frontends"].items()},
    },indent=2))
    return receipt


def restore_model(root,name,receipt):
    item=receipt["frontends"][name]
    path=Path(root)/item["checkpoint"]
    if not path.is_file():
        raise RuntimeError("STOP missing newly-trained segmenter")
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=item["checkpoint_SHA256"]:
        raise RuntimeError("STOP checkpoint SHA256 drift")
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload["architecture"]!=name or tuple(payload["widths"])!=WIDTHS:
        raise RuntimeError("STOP checkpoint architecture mismatch")
    model=build_comparable_model(name,widths=WIDTHS)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--images",type=Path,required=True)
    p.add_argument("--annotations",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    execute(a.images,a.annotations,a.out)
