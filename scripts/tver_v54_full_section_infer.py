"""Frozen ANZA inference and coverage audit across complete seismic sections.

TRAIN1/VAL1 source arrays and frozen V33 model are used. TEST7 is never
opened. This audits full-slice *segmentation evidence*, not fault identity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from trace_extraction.coverage import (
    annotated_trace_coverage, mode_layers,
)

THRESHOLDS=(.90,.80,.70,.60,.50,.40,.30)
# Frozen before full-slice outcomes, and never optimized on held-out labels.
TRAIN_SECTIONS=(20,80)
VAL_ALREADY_SEEN_SECTIONS=(43,77)
MODEL_SHA256="ba0c3b8060fd0e3cb1c15f6c1598b94a977677c2b075663b5414207df382c25e"


def tile_starts(length, size, stride):
    if not (isinstance(length,int) and isinstance(size,int) and isinstance(stride,int)
            and size>=16 and 1<=stride<=size and length>=size):
        raise ValueError("Invalid tiling dimensions")
    indices=list(range(0,length-size+1,stride))
    last=length-size
    if indices[-1]!=last:
        indices.append(last)
    return indices


def tiled_probability(seismic, model, normalization, *,
                      tile_size=96, stride=64, batch_size=8):
    """Batch and overlap-add model logits WITHOUT any reference annotations.

    Seismic amplitudes are scaled with the frozen TRAIN-only percentiles.
    Raised-cosine weights reduce seam artifacts; every pixel has nonzero weight.
    """
    import torch

    image=np.asarray(seismic)
    if image.ndim!=2 or min(image.shape)<tile_size or not np.isfinite(image).all():
        raise ValueError("Expected finite complete 2-D seismic image")
    if not isinstance(batch_size,int) or not 1<=batch_size<=256:
        raise ValueError("Invalid tile batch size")
    ys=tile_starts(int(image.shape[0]),tile_size,stride)
    xs=tile_starts(int(image.shape[1]),tile_size,stride)
    lo=float(normalization["TRAIN_only_1pct"])
    hi=float(normalization["TRAIN_only_99pct"])
    mean=float(normalization["normalized_channel_mean"])
    std=float(normalization["normalized_channel_std"])
    if not (np.isfinite([lo,hi,mean,std]).all() and lo<hi and std>0):
        raise ValueError("Invalid frozen train normalization")

    wave=np.hanning(tile_size)
    window=np.maximum(np.outer(wave,wave),.05).astype(np.float32)
    numerator=np.zeros(image.shape,np.float32)
    denominator=np.zeros(image.shape,np.float32)
    coords=[(y,x) for y in ys for x in xs]
    model.eval()
    with torch.inference_mode():
        for offset in range(0,len(coords),batch_size):
            current=coords[offset:offset+batch_size]
            prepared=[]
            for y,x in current:
                raw=np.asarray(image[y:y+tile_size,x:x+tile_size],np.float32)
                scaled=np.clip((raw-lo)/(hi-lo),0,1)
                input_field=np.ascontiguousarray((scaled-mean)/std)
                prepared.append(np.repeat(input_field[None],3,axis=0))
            tensor=torch.from_numpy(np.stack(prepared))
            raw_output=model(tensor)
            if raw_output.shape!=(len(current),1,tile_size,tile_size):
                raise RuntimeError("Full-section model output has unexpected shape")
            scores=torch.sigmoid(raw_output).cpu().numpy()[:,0]
            if not np.isfinite(scores).all():
                raise RuntimeError("Nonfinite predicted full-section score")
            for (y,x),prob in zip(current,scores):
                numerator[y:y+tile_size,x:x+tile_size]+=prob*window
                denominator[y:y+tile_size,x:x+tile_size]+=window
    if (denominator<=0).any():
        raise RuntimeError("Full-section tile coverage has holes")
    probability=numerator/denominator
    return probability.astype(np.float32),{
        "tile_size":tile_size,"stride":stride,"batch_size":batch_size,
        "n_tiles":len(coords),"model_state_unchanged":True,
        "uses_expert_truth_as_input":False,
    }


def _small_preview(probability, output_path):
    """Downsampled evidence image for human inspection, no GT overlay."""
    from PIL import Image

    height,width=probability.shape
    factor=max(1,int(np.ceil(max(height,width)/1600)))
    sample=np.asarray(probability[::factor,::factor],dtype=np.float32)
    gray=(np.clip(sample,0,1)*255).astype(np.uint8)
    # Green/high, amber/weak-supported, magenta/weak-to-review.
    layers=mode_layers(sample,"coverage")
    rgb=np.repeat(gray[...,None],3,axis=-1)
    rgb[layers.high_confidence]=[24,220,92]
    rgb[layers.supported_weak]=[245,180,45]
    rgb[layers.review_weak]=[180,75,235]
    Image.fromarray(rgb).save(output_path)
    return factor


def evaluate_full_sections(memmap_root,frontends_dir,out_dir,
                           *,train_sections=TRAIN_SECTIONS,
                           val_sections=VAL_ALREADY_SEEN_SECTIONS,
                           tile_size=96,stride=64,batch_size=8):
    import torch
    from scripts.tver_v34_anza_p0_relation_cal import _state_report
    from scripts.tver_v33_thebe_relation_cal import _load_model
    from scripts.tver_v33_thebe_train_segmenters import read_memmap

    if (any(not 0<=i<100 for i in train_sections)
            or any(not 40<=i<100 for i in val_sections)
            or not train_sections or not val_sections):
        raise ValueError("Source sections outside allowed TRAIN/inspected VAL sets")
    out=Path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    state=_state_report(frontends_dir)
    if state["frontends"]["anza_v1"]["SHA256"]!=MODEL_SHA256:
        raise RuntimeError("Original frozen Thebe ANZA changed")
    source=Path(frontends_dir)/state["frontends"]["anza_v1"]["checkpoint"]
    if hashlib.sha256(source.read_bytes()).hexdigest()!=MODEL_SHA256:
        raise RuntimeError("Original frozen Thebe checkpoint file mismatch")
    model=_load_model(frontends_dir,"anza_v1",state)
    torch.set_num_threads(2)
    records=[]
    for cohort,source_name,expert_name,identities in (
        ("TRAIN1","train_seismic_float32.npy","train_fault_bool.npy",train_sections),
        ("VAL1_ALREADY_SEEN","val_seismic_float32.npy","val_fault_bool.npy",val_sections),
    ):
        full=read_memmap(memmap_root,source_name)
        mask=read_memmap(memmap_root,expert_name)
        for section in identities:
            predicted,receipt=tiled_probability(
                full[section],model,state["normalization"],
                tile_size=tile_size,stride=stride,batch_size=batch_size)
            base=out/f"{cohort}_{section}"
            base.mkdir(exist_ok=True)
            np.save(base/"probability.npy",predicted,allow_pickle=False)
            profile=annotated_trace_coverage(
                predicted,mask[section],THRESHOLDS,tolerance_px=2)
            scale=_small_preview(predicted,base/"coverage_preview.png")
            annotated=np.asarray(mask[section],bool)
            for row in profile:
                records.append({
                    "cohort":cohort,"section":int(section),
                    "n_annotated_pixels":int(annotated.sum()),
                    **row,
                })
            (base/"receipt.json").write_text(json.dumps({
                "cohort":cohort,"section":int(section),
                "shape":list(predicted.shape),
                "original_checkpoint_sha256":MODEL_SHA256,
                "tiling":receipt,"preview_downsample":scale,
                "coverage_profile":profile,
                "source_is_real_seismic":True,
                "gaps_automatically_detected":False,
                "no_expert_intervention_rectangle_required":True,
                "labels_only_used_after_inference":True,
            },indent=2)+"\n",encoding="utf8")
            print("V54_FULL_SECTION_PROFILE",cohort,int(section),
                  json.dumps(profile),flush=True)
        del full,mask
    summary={
        "status":"V54_FROZEN_ANZA_FULL_SECTION_COVERAGE_EXPLORATORY",
        "train_sections":list(train_sections),
        "previously_inspected_val_sections":list(val_sections),
        "thresholds_frozen":list(THRESHOLDS),
        "model_sha256":MODEL_SHA256,
        "no_TEST7_access":True,
        "does_not_establish_natural_fault_instance_recall":True,
        "no_automatic_continuation_acceptance":True,
        "no_threshold_selected_using_evaluation_masks":True,
        "partial_expert_labels_not_complete_negative_truth":True,
        "records":records,
    }
    with (out/"V54_FULL_SECTION_COVERAGE.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (out/"V54_FULL_SECTION_SUMMARY.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    return summary


def main():
    parser=argparse.ArgumentParser()
    for arg in ("mmap-root","frontends-dir","out-dir"):
        parser.add_argument("--"+arg,required=True,type=Path)
    parser.add_argument("--train-sections",type=int,nargs="+",
                        default=list(TRAIN_SECTIONS))
    parser.add_argument("--val-sections",type=int,nargs="+",
                        default=list(VAL_ALREADY_SEEN_SECTIONS))
    parser.add_argument("--tile-size",type=int,default=96)
    parser.add_argument("--stride",type=int,default=64)
    parser.add_argument("--batch-size",type=int,default=8)
    a=parser.parse_args()
    evaluate_full_sections(a.mmap_root,a.frontends_dir,a.out_dir,
        train_sections=a.train_sections,val_sections=a.val_sections,
        tile_size=a.tile_size,stride=a.stride,batch_size=a.batch_size)


if __name__=="__main__":
    main()
