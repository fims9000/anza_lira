#!/usr/bin/env python3
"""Qualitative H50Z demo with exact-source checkpoints. No real-film ground truth."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s7_original_retrain import RunConfig,build_models,verify_sources

def predict_tiled(model,image:np.ndarray,tile:int=64,step:int=48)->np.ndarray:
    if image.ndim!=2 or image.dtype!=np.uint8:raise ValueError("Expected grayscale uint8")
    h,w=image.shape
    if h<tile or w<tile or tile%16:raise ValueError("Insufficient crop size")
    ys=list(range(0,h-tile+1,step))
    xs=list(range(0,w-tile+1,step))
    if ys[-1]!=h-tile:ys.append(h-tile)
    if xs[-1]!=w-tile:xs.append(w-tile)
    scores=np.zeros((h,w),np.float32)
    counts=np.zeros((h,w),np.float32)
    model.eval()
    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch=(255-image[y:y+tile,x:x+tile]).astype(np.float32)/255.0
                inp=torch.from_numpy(patch[None,None])
                pred=model(inp)
                if isinstance(pred,dict):pred=pred["main"]
                probs=pred.sigmoid()[0,0].numpy()
                scores[y:y+tile,x:x+tile]+=probs
                counts[y:y+tile,x:x+tile]+=1
    return scores/counts

def run(images:list[Path],checkpoints:Path,out:Path):
    source=verify_sources(ROOT)
    torch.set_num_threads(2)
    cfg=RunConfig(size=64,widths=(4,6,8,12),num_rules=2,az_residual=True)
    models=build_models(cfg)
    thresholds={}
    for name,model in models.items():
        ckpt=torch.load(checkpoints/name/"best_final.pt",map_location="cpu",weights_only=False)
        if ckpt["source_blobs"]!=source:raise ValueError("Original model source mismatch")
        model.load_state_dict(ckpt["state_dict"])
        thresholds[name]=float(ckpt["validation_selected_threshold"])
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for filename in images:
        raw=cv2.imread(str(filename),cv2.IMREAD_GRAYSCALE)
        if raw is None:raise FileNotFoundError(filename)
        h,w=raw.shape
        size=(min(256,h,w)//16)*16
        y=(h-size)//2;x=(w-size)//2
        crop=raw[y:y+size,x:x+size]
        prob={name:predict_tiled(model,crop) for name,model in models.items()}
        masks={name:value>=thresholds[name] for name,value in prob.items()}
        fig,axes=plt.subplots(1,5,figsize=(16,3.8))
        axes[0].imshow(crop,cmap="gray",vmin=0,vmax=255)
        axes[0].set_title("Real H50Z (unlabelled)")
        for j,name in enumerate(("plain_unet","original_azunet")):
            axes[1+j].imshow(prob[name],cmap="viridis",vmin=0,vmax=1)
            axes[1+j].set_title(name+" probability")
            axes[3+j].imshow(crop,cmap="gray",vmin=0,vmax=255)
            axes[3+j].contour(masks[name],levels=[.5],colors="red",linewidths=.5)
            axes[3+j].set_title(name+" candidate tracks")
        for ax in axes:ax.axis("off")
        fig.suptitle("No real film pixel masks available; NOT an accuracy evaluation")
        fig.tight_layout()
        dest=out/(filename.stem+"_compare.png")
        fig.savefig(dest,dpi=145,bbox_inches="tight")
        plt.close(fig)
        records.append({"image":filename.name,"roi_xyxy_in_crop":[x,y,x+size,y+size],
                        "models":{name:{"validation_threshold":thresholds[name],
                                       "fraction_foreground":float(masks[name].mean()),
                                       "mean_probability":float(prob[name].mean())} for name in models},
                        "real_labels_present":False,"evidence_level":"QUALITATIVE_ONLY","visualization":dest.name})
    (out/"real_unlabeled_inference_manifest.json").write_text(json.dumps(records,indent=2)+"\n")
    return records

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--images",type=Path,nargs="+",required=True)
    p.add_argument("--checkpoints",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.images,a.checkpoints,a.out_dir),indent=2))
