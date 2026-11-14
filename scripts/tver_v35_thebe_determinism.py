"""Thebe V35: bitwise real TRAIN model replay, test7 embargo.

Frozen AFTER inspecting V33 repeat instability. Explicit MKLDNN off and
one CPU thread; two retrains *within the same process* for each architecture
on the same real TRAIN patches. Do not infer across-run stability yet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.tver_v33_thebe_train_segmenters import (
    TRAIN_STEPS,TRAIN_SAMPLE_SEED,
    read_memmap,amplitude_scale,_make_patches,_fit_model,
)


def tensor_state_hash(state):
    digest=hashlib.sha256()
    for name in sorted(state):
        x=state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(x.dtype).encode())
        digest.update(str(tuple(x.shape)).encode())
        digest.update(x.tobytes(order="C"))
    return digest.hexdigest()


def dataset_patch_hash(patches):
    h=hashlib.sha256()
    for i,sample in enumerate(patches):
        h.update(i.to_bytes(4,"little"))
        for part in sample:
            p=np.ascontiguousarray(part)
            h.update(str(p.dtype).encode())
            h.update(str(tuple(p.shape)).encode())
            h.update(p.tobytes())
    return h.hexdigest()


def execute(root,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)
    train_x=read_memmap(root,"train_seismic_float32.npy")
    train_y=read_memmap(root,"train_fault_bool.npy")
    val_x=read_memmap(root,"val_seismic_float32.npy")
    val_y=read_memmap(root,"val_fault_bool.npy")
    scale=amplitude_scale(train_x)
    patches,_=_make_patches(
        train_x,train_y,scale,count=TRAIN_STEPS,
        seed=TRAIN_SAMPLE_SEED+1,selected_sections=range(100),
        positive_fraction=.72)
    cal_patches,_=_make_patches(
        val_x,val_y,scale,count=96,
        seed=TRAIN_SAMPLE_SEED+2,selected_sections=range(30),
        positive_fraction=.50)
    report={
        "status":"V35_THEBE_REAL_RETRAIN_DETERMINISM_AUDIT",
        "same_original_official_thebe_TRAIN1_and_VAL1":True,
        "external_TEST7_bytes_masks_scores_NEVER_OPENED":True,
        "training_patches_SHA256":dataset_patch_hash(patches),
        "SEG_CAL_patches_SHA256":dataset_patch_hash(cal_patches),
        "train_normalization":scale,
        "num_CPU_threads":torch.get_num_threads(),
        "num_interop_threads":torch.get_num_interop_threads(),
        "MKLDNN_disabled":not torch.backends.mkldnn.enabled,
        "torch_deterministic_algorithms_enabled":torch.are_deterministic_algorithms_enabled(),
        "torch_version":torch.__version__,
        "numpy_version":np.__version__,
        "model_results":{},
    }
    for name,seed in (("unet",20263312),("anza_v1",20263311)):
        repeats=[]
        for repeat in range(2):
            folder=out/f"{name}_repeat{repeat+1}"
            folder.mkdir(parents=True,exist_ok=True)
            model,r=_fit_model(name,seed,patches,cal_patches,scale,folder)
            model.eval()
            sha=tensor_state_hash(model.state_dict())
            repeats.append({
                "model_tensor_state_SHA256":sha,
                "saved_checkpoint_SHA256":r["SHA256"],
                "train_loss_first150":r["train_loss_first150"],
                "train_loss_last150":r["train_loss_last150"],
                "pixel_CAL_score_receipt":r[
                    "pixel_CAL_on_unmarked_background_not_geological_precision"],
                "original_model_parameters":r["n_params"],
            })
            print("V35_REPLAY",name,repeat+1,sha,flush=True)
            del model
        equal_states=(
            repeats[0]["model_tensor_state_SHA256"]==
            repeats[1]["model_tensor_state_SHA256"])
        equal_CALS=(
            repeats[0]["pixel_CAL_score_receipt"]==
            repeats[1]["pixel_CAL_score_receipt"])
        equal_losses=all(
            repeats[0][k]==repeats[1][k] for k in
            ("train_loss_first150","train_loss_last150"))
        report["model_results"][name]={
            "both_replays":repeats,
            "bitwise_model_state_equal":equal_states,
            "equal_CAL_pixel_scores":equal_CALS,
            "equal_train_loss_aggregates":equal_losses,
            "original_train_budget_steps_each":TRAIN_STEPS,
        }
    report["status"]=(
        "V35_THEBE_DETERMINISTIC_WITHIN_ONE_RUN_TEST7_UNOPENED"
        if all(all(x[k] for k in (
            "bitwise_model_state_equal",
            "equal_CAL_pixel_scores",
            "equal_train_loss_aggregates"))
            for x in report["model_results"].values()) else
        "V35_STOP_REAL_THEBE_BITWISE_TRAINING_NONDETERMINISM")
    path=out/"v35_thebe_real_training_determinism.json"
    path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({
        "status":report["status"],
        "patch_sha":report["training_patches_SHA256"],
        "models":{n:{
            "model_state_equal":x["bitwise_model_state_equal"],
            "pixel_CAL_scores_equal":x["equal_CAL_pixel_scores"]}
            for n,x in report["model_results"].items()},
        "external_TEST7_accessed":False,
    },indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mmap-root",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    a=p.parse_args()
    execute(a.mmap_root,a.out_dir)
