"""Official real CRACKS V26 supervised-label inventory and CPU smoke.

No model TEST metrics are calculated. Only source-section inventory, label
sparsity and one synthetic microbatch backward pass on each ACTUAL
repository architecture. This is prerequisite for an honest new training.
"""
from __future__ import annotations

from collections import Counter,defaultdict
import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.cracks_real_oracle_gap_audit import sections,COLORS,paths_from_mask


TRAIN={0,1,2,3,4}
CAL={5}
BUFFER={6}
TEST={7}


def _split(sid):
    block=int(sid)//50
    if block in TRAIN:return "TRAIN"
    if block in CAL:return "CAL"
    if block in BUFFER:return "BUFFER"
    if block in TEST:return "TEST"
    return "UNREGISTERED"


def inventory(images,annotations):
    with zipfile.ZipFile(images) as zi,zipfile.ZipFile(annotations) as za:
        ims=sections(zi)
        ann=sections(za,expert=True)
        ids=sorted(set(ims)&set(ann))
        if len(ids)!=40:
            raise ValueError("STOP: expected exact 40 official expert F3 sections")
        result={"status":"V26_OFFICIAL_EXPERT_INVENTORY",
                "n_expert_sections":len(ids),
                "sections_by_split":defaultdict(list),
                "color_pixels_by_split":defaultdict(Counter),
                "long_blue_paths_by_split":Counter(),
                "image_shapes":Counter(),
                "train_patch_background_unknown":True,
                "test_labels_not_used_for_model_selection":True}
        for sid in ids:
            split=_split(sid)
            with Image.open(io.BytesIO(zi.read(ims[sid]))) as im:
                picture=np.asarray(im.convert("RGB"))
            with Image.open(io.BytesIO(za.read(ann[sid]))) as am:
                target=np.asarray(am.convert("RGB"))
            if picture.shape!=target.shape:
                raise ValueError("STOP: image/annotation shape differs")
            result["image_shapes"][str(picture.shape)]+=1
            result["sections_by_split"][split].append(sid)
            for name,color in COLORS.items():
                pixels=int(np.all(target==color,axis=-1).sum())
                result["color_pixels_by_split"][split][name]+=pixels
            blue=np.all(target==COLORS["blue"],axis=-1)
            green=np.all(target==COLORS["green"],axis=-1)
            from scipy.ndimage import binary_dilation
            count=sum(len(p)>=70 for p in paths_from_mask(
                blue&~binary_dilation(green,iterations=5)))
            result["long_blue_paths_by_split"][split]+=count
        result["sections_by_split"]=dict(result["sections_by_split"])
        result["color_pixels_by_split"]={
            k:dict(v) for k,v in result["color_pixels_by_split"].items()}
        for k in ("TRAIN","CAL","TEST"):
            if not result["sections_by_split"].get(k):
                raise ValueError("STOP: no expert sections in required "+k)
        result["long_blue_paths_by_split"]=dict(result["long_blue_paths_by_split"])
        result["image_shapes"]=dict(result["image_shapes"])
        return result


def _smoke():
    import torch
    from models.segmentation_v2 import build_comparable_model
    torch.set_num_threads(2)
    torch.manual_seed(26011)
    results={}
    # Dummy input only to test whether ORIGINAL ANZA operator is trainable.
    for name in ("unet","anza_v1"):
        model=build_comparable_model(name,widths=(4,8,12,16))
        batch=torch.randn(1,3,64,64)
        loss=model(batch).square().mean()
        start=time.monotonic()
        loss.backward()
        elapsed=time.monotonic()-start
        results[name]={
            "architecture":"models.segmentation_v2.build_comparable_model",
            "parameters":sum(x.numel() for x in model.parameters()),
            "one_microbatch_forward_backward_sec":elapsed,
            "can_backpropagate_all_parameters":any(
                x.grad is not None for x in model.parameters()),
            "original_pretrained_weights_loaded":False,
        }
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--images",type=Path,required=True)
    p.add_argument("--annotations",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    report=inventory(a.images,a.annotations)
    report["CPU_64px_model_trainability_smoke"]=_smoke()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
