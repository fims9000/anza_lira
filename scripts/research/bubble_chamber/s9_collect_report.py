#!/usr/bin/env python3
"""Reconstruct complete S9 comparison from metrics and checkpoints, with provenance."""
from __future__ import annotations
import argparse,hashlib,json,statistics
from pathlib import Path

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(65536),b""):h.update(b)
    return h.hexdigest()

def collect(root:Path,output:Path,seeds=(20261024,20261025,20261026)):
    items=[];data_sha=None
    for seed in seeds:
        r=root/"results"/f"s9_replication_seed{seed}"
        m=json.loads((r/"metrics.json").read_text())
        f=json.loads((root/"results"/f"s9_frangi_seed{seed}.json").read_text())
        if m["experiment"]!="S7_ORIGINAL_LOCKED_AZUNET_VS_UNET":raise ValueError("Unknown experimental run")
        if m["source_data"]["real_pixel_labels"] is not False:raise ValueError("Unexpected real-mask claim")
        if m["source_data"]["film_ids"]!=["070"]:raise ValueError("Wrong film/run")
        if m["source_data"]["test_examples"]["frames"]<20:raise ValueError("Test not frame-disjoint")
        if m["source_data"]["test_examples"]["sha256"]!=f["source_fingerprints"]["test"]["sha256"]:
            raise ValueError("Frangi and neural test datasets differ")
        if m["source_data"]["val_examples"]["sha256"]!=f["source_fingerprints"]["val"]["sha256"]:
            raise ValueError("Frangi and neural validation datasets differ")
        if not m["config"]["robust_train"] or not m["config"]["az_residual"]:
            raise ValueError("Wrong ablation configuration")
        if any(v["trained_epochs"]!=8 for v in m["results"].values()):
            raise ValueError("Incomplete neural training")
        actual_sha=m["source_data"]["audit_sha256"]
        if data_sha is None:data_sha=actual_sha
        if data_sha!=actual_sha:raise ValueError("Source measurements changed")
        models={}
        for label,name in (("UNet","plain_unet"),("AZUNet+residual","original_azunet")):
            v=m["results"][name];checkpoint=r/name/"best_final.pt"
            models[label]={
                "dice_clean":v["test_clean"]["dice"],"dice_stress":v["test_stress"]["dice"],
                "precision_clean":v["test_clean"]["precision"],"precision_stress":v["test_stress"]["precision"],
                "recall_clean":v["test_clean"]["recall"],"recall_stress":v["test_stress"]["recall"],
                "cldice_clean":v["test_clean_topology"]["macro_cldice"],
                "cldice_stress":v["test_stress_topology"]["macro_cldice"],
                "threshold_val":v["selected_validation_threshold"],
                "parameters":v["parameters"],"checkpoint_sha256":sha(checkpoint),
                "best_epoch":v["best_epoch"]
            }
        models["Frangi"]={
            "dice_clean":f["clean"]["dice"],"dice_stress":f["stress"]["dice"],
            "precision_clean":f["clean"]["precision"],"precision_stress":f["stress"]["precision"],
            "recall_clean":f["clean"]["recall"],"recall_stress":f["stress"]["recall"],
            "threshold_val":f["threshold_selected_on_validation"]
        }
        items.append({"seed":seed,"train_images":m["source_data"]["train_examples"]["n"],
                      "train_frames":m["source_data"]["train_examples"]["frames"],
                      "val_images":m["source_data"]["val_examples"]["n"],
                      "val_frames":m["source_data"]["val_examples"]["frames"],
                      "test_images":m["source_data"]["test_examples"]["n"],
                      "test_frames":m["source_data"]["test_examples"]["frames"],
                      "models":models,"paired_az_minus_unet_bootstrap":m["paired"]})
    methods=("UNet","AZUNet+residual","Frangi")
    means={name:{key:float(statistics.mean(x["models"][name][key] for x in items))
                 for key in ("dice_clean","dice_stress")} for name in methods}
    delta=[x["models"]["AZUNet+residual"]["dice_stress"]-x["models"]["UNet"]["dice_stress"] for x in items]
    summary={
        "experiment":"S9_EXPLORATORY_THREE_SEED_ORIGINAL_AZCONV",
        "data_provenance_sha256":data_sha,"film_id":"070",
        "real_pixels_used_for_training":False,"synthetic_labels_only":True,
        "real_camera_performance_validated":False,"model_sources":m["original_source_git_blobs"],
        "model_implementation":"Original AZUNet and BaselineUNet, built-in AZConv2d residual enabled",
        "augmentation":"Identical deterministic train-only contrast/noise for neural models",
        "validation_threshold":"Selected per model on validation for balanced clean/stress Dice",
        "frangi":"Frangi ridges, threshold selected on matching validation",
        "comparison_unit":"Whole source frames; no independent film/run available",
        "training_epochs":8,"widths":[4,6,8,12],"rules":2,
        "per_seed":items,"arithmetic_means_exploratory_only":means,
        "az_minus_unet_stress_dice_differences":delta,
        "status":"NO_STABLE_AZUNET_SUPERIORITY_ON_SYNTHETIC_FRAME_HELD_OUT_TEST",
        "limitations":[
            "Seed changes both frame split and synthetic rendering, not just model initialisation",
            "Only three seeds; means are descriptive, not a statistical superiority test",
            "All source tracks originate from film 070; cannot test transfer to propane/hydrogen",
            "Masks and intensities are synthetic from real expert-clicked sparse 2D tracks",
            "No human-confirmed pixel-level truth exists for supplied H50Z real image crops"
        ]}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    r=collect(a.root,a.output)
    print(json.dumps({"status":r["status"],"mean_dice":r["arithmetic_means_exploratory_only"]},indent=2))
