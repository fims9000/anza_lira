#!/usr/bin/env python3
"""Reproducible ANZA-vs-CNN study on CT28 TRAIN patients with 3D mask outputs.

One command can prepare caches, OOF scores, train clean relation HGB heads,
create a controlled-damage K10 pilot, run both neural 3D CCTA models and
measure paired Dice/clDice on the SAME damaged input. Every stage is create-
only and resumable only when previous artifacts pass basic checksum checks.

This is a development study, NOT an independent clinical/test evaluation.
It does not open any of CT28 held-out test patients.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccta_3d_an_patch_cache import build_patient_cache, DEFAULT_CACHE
from ccta_3d_an_oof import build_oof, sha256
from ccta_3d_an_relation import run_relation_training
from ccta_3d_an_cli import run_neural
from ccta_3d_train_pilot import execute_train_pilot, SELECTED_PACK, DEFAULT_ROOT
from ccta_3d_evaluate import evaluate_files
from extract_ct28_scene_candidate_features import EXPECTED_PATIENT_IDS


TRAIN=sorted(EXPECTED_PATIENT_IDS["train"])
VAL=sorted(EXPECTED_PATIENT_IDS["val"])
DEFAULT_STUDY=DEFAULT_ROOT/"anza_vs_cnn_development"


def _load_json(path:Path)->dict:
    with path.open(encoding="utf-8") as reader:
        return json.load(reader)


def _verified_cache(cache:Path,patient:int)->bool:
    path=cache/f"patches_{patient}.npz"
    report=cache/f"patches_{patient}.json"
    if not path.exists() and not report.exists():
        return False
    if not path.is_file() or not report.is_file():
        raise ValueError(f"Partial cached CT patch evidence for patient {patient}")
    meta=_load_json(report)
    split="train" if patient in TRAIN else "val"
    if meta.get("patient_id")!=patient or meta.get("split")!=split or meta.get("cache_sha256")!=sha256(path):
        raise ValueError(f"Corrupted or patient/split-mismatched CT patch cache {patient}")
    return True


def _verified_oof(path:Path,kind:str,variant:str)->bool:
    if not path.exists():
        return False
    report=path/"oof_protocol.json"
    scores=path/f"{kind}_{variant}_oof_val.csv"
    if not report.exists() or not scores.exists():
        raise ValueError(f"Partial OOF model evidence: {path}")
    m=_load_json(report)
    if m.get("status")!="train_patient_OOF_experimental_CT_evidence_not_promoted_K10" or (
        m.get("variant"),m.get("candidate_type")
    )!=(variant,kind) or m.get("score_sha256")!=sha256(scores):
        raise ValueError(f"OOF model score protocol changed: {path}")
    if m.get("n_train_patients")!=17 or m.get("n_validation_patients")!=5 or m.get("held_out_test_accessed") is not False:
        raise ValueError("Invalid train/val OOF cohort")
    checkpoint=path/f"{kind}_{variant}_fulltrain.pt"
    if not checkpoint.exists() or sha256(checkpoint)!=m.get("fulltrain_model_sha256"):
        raise ValueError("OOF neural model weights changed")
    return True


def _verified_relation(path:Path,variant:str)->bool:
    if not path.exists():
        return False
    manifest=path/"relation_protocol.json"
    model=path/"clean_relation_head.joblib"
    pred=path/"clean_relation_predictions_val.csv"
    if not all(x.exists() for x in (manifest,model,pred)):
        raise ValueError(f"Partial relation-model evidence: {path}")
    m=_load_json(manifest)
    if m.get("status")!="experimental_clean_relation_head_no_consistency_gate" or m.get("variant")!=variant:
        raise ValueError("Wrong experimental relation variant")
    if m.get("model_sha256")!=sha256(model) or m.get("predictions_sha256")!=sha256(pred):
        raise ValueError("Relation model/prediction hash mismatch")
    return True


def ensure_patch_cache(cache:Path)->None:
    for patient in TRAIN+VAL:
        if _verified_cache(cache,patient):
            continue
        build_patient_cache(patient=patient,output_dir=cache,
                            include_val=patient in VAL)
        print(f"CT28 3D patches complete: {patient}",flush=True)


def ensure_oof(cache:Path,root:Path,*,epochs:int,device:str,folds:int,batch_size:int)->None:
    for kind in ("PAIR","JUNCTION"):
        for variant in ("conv3d","anza3d"):
            out=root/f"oof_{kind.lower()}_{variant}"
            if _verified_oof(out,kind,variant):
                continue
            build_oof(
                cache_dir=cache,out_dir=out,candidate_type=kind,
                variant=variant,train_patients=TRAIN,val_patients=VAL,
                folds=folds,epochs=epochs,batch_size=batch_size,
                device=device,
            )
            print(f"Patient-OOF candidate evidence: {kind}/{variant}",flush=True)


def ensure_relations(root:Path)->None:
    for variant in ("conv3d","anza3d"):
        path=root/f"relation_{variant}"
        if _verified_relation(path,variant):
            continue
        run_relation_training(
            pair_oof=root/f"oof_pair_{variant}",
            junction_oof=root/f"oof_junction_{variant}",
            variant=variant,output_dir=path,
        )
        print(f"Clean 84-feature relation trained: {variant}",flush=True)


def ensure_pilot(root:Path,*,patient:int)->Path:
    if patient not in TRAIN:
        raise ValueError("Full-volume ANZA comparison requires a train patient")
    output=root/f"pilot_{patient}"
    if output.exists():
        manifest=output/"pilot_manifest.json"
        if not manifest.exists():
            raise ValueError("Partial previous K10 pilot exists")
        meta=_load_json(manifest)
        if meta.get("status")!="completed_experimental_train_pilot" or meta.get("patient_id")!=patient or meta.get("mode")!="k10":
            raise ValueError("Wrong previous K10 patient pilot")
    else:
        execute_train_pilot(
            scan_id=patient,output_dir=output,mode="k10",
            gaps=3,gap_radius_mm=1.5,
            repair_radius_mm=0.,max_new_voxels=250,
        )
    return output


def ensure_neural_inference(root:Path,*,pilot:Path,patient:int,device:str)->None:
    for variant in ("conv3d","anza3d"):
        repaired=root/f"{variant}_repaired.nii.gz"
        additions=root/f"{variant}_additions.nii.gz"
        decisions=root/f"{variant}_decisions.json"
        metrics=root/f"{variant}_posthoc.json"
        artifacts=(repaired,additions,decisions,metrics)
        if any(path.exists() for path in artifacts):
            if not all(path.is_file() for path in artifacts):
                raise ValueError(f"Partial neural volume evidence for {variant}")
            provenance=_load_json(decisions)
            if provenance.get("experimental_evidence",{}).get("variant")!=variant or provenance.get("patient_label")!=str(patient):
                raise ValueError("Wrong neural volume inference provenance")
            evaluation=_load_json(metrics)
            if evaluation.get("patient_id")!=str(patient):
                raise ValueError("Wrong posthoc evaluation patient")
            continue
        run_neural(
            ct=pilot/"ct.nii.gz",
            mask=pilot/"broken_mask.nii.gz",
            patient_id=str(patient),
            base_k10=SELECTED_PACK,
            relation_dir=root/f"relation_{variant}",
            pair_oof=root/f"oof_pair_{variant}",
            junction_oof=root/f"oof_junction_{variant}",
            variant=variant,output_mask=repaired,
            output_additions=additions,
            report=decisions,device=device,
            radius_mm=0.,max_new_voxels=250,
        )
        evaluate_files(
            patient_id=str(patient),
            before=pilot/"broken_mask.nii.gz",
            after=repaired,
            reference=(
                Path(__file__).resolve().parents[3]/
                f"varvara/anza_lira_local_data/Varvara_CT28_junction_source_pack/segmentations/{patient}.coronary.nii.gz"
            ),
            output=metrics,
        )
        print(f"3D neural CT inference and Dice/clDice computed: {variant}",flush=True)


def summarize(root:Path,*,pilot:Path,patient:int)->dict:
    k10_metrics=_load_json(pilot/"posthoc_metrics.json")
    inputs=_load_json(pilot/"decisions.json")
    records={}
    for variant in ("K10","conv3d","anza3d"):
        m=k10_metrics if variant=="K10" else _load_json(root/f"{variant}_posthoc.json")
        decisions=inputs if variant=="K10" else _load_json(root/f"{variant}_decisions.json")
        if decisions.get("patient_label")!=str(patient) or m.get("patient_id")!=str(patient):
            raise ValueError(f"Comparative segmentation evidence patient mismatch: {variant}")
        if not all(isinstance(m.get(key),(int,float)) for key in (
            "dice_before","dice_after","dice_delta",
            "cldice_before","cldice_after","cldice_delta",
        )):
            raise ValueError(f"Missing paired 3D mask metrics for {variant}")
        if variant!="K10":
            if decisions.get("experimental_evidence",{}).get("variant")!=variant:
                raise ValueError("Neural encoder model provenance mismatch")
            if m["before_sha256"]!=k10_metrics["before_sha256"]:
                raise ValueError("Not the same broken 3D segmentation input")
            if m["reference_sha256"]!=k10_metrics["reference_sha256"]:
                raise ValueError("Not the same reference volume in posthoc comparison")
        records[variant]={
            "accepted_repairs":decisions["accepted_decisions"],
            "added_voxels":decisions["added_voxels"],
            "dice_before":m["dice_before"],
            "dice_after":m["dice_after"],
            "delta_dice":m["dice_delta"],
            "cldice_before":m["cldice_before"],
            "cldice_after":m["cldice_after"],
            "delta_cldice":m["cldice_delta"],
            "reference_unsupported_added_voxels":m["reference_unsupported_added_voxels"],
        }
    result={
        "status":"train_only_exploratory_single_patient_volume_comparison",
        "patient":patient,
        "source":"controlled damaged ImageCAS-X reference segmentation",
        "same_ct_mask_and_reference_for_all":True,
        "held_out_test_accessed":False,
        "models":records,
        "limitations":[
            "ANZA/CNN compare clean relation HGB heads, not 15-copy augmented K10 head",
            "Single training patient is not generalization evidence",
            "Real natural predicted mask errors remain untested",
            "ANZA advantage requires independent subjects and matched train augmentations",
        ],
    }
    report=root/f"train_{patient}_three_model_comparison.json"
    md=root/f"train_{patient}_three_model_comparison.md"
    if report.exists() or md.exists():
        raise FileExistsError("Study summary already exists")
    report.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lines=[
        f"# CT28 CCTA 3D train-only controlled gap comparison — patient {patient}",
        "",
        "| Variant | Accepted repairs | Added voxels | False added voxels | Dice before | Dice after | clDice before | clDice after |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant,row in records.items():
        lines.append(f"| {variant} | {row['accepted_repairs']} | {row['added_voxels']} | "
                     f"{row['reference_unsupported_added_voxels']} | "
                     f"{row['dice_before']:.5f} | {row['dice_after']:.5f} | "
                     f"{row['cldice_before']:.5f} | {row['cldice_after']:.5f} |")
    lines.extend(["","## Limitations",""]+[f"- {x}" for x in result["limitations"]])
    md.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("patches","oof","relations","pilot","volume","all"),default="all")
    parser.add_argument("--root",type=Path,default=DEFAULT_STUDY)
    parser.add_argument("--cache-dir",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--patient-id",type=int,default=953)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--epochs",type=int,default=8)
    parser.add_argument("--folds",type=int,default=5)
    parser.add_argument("--batch-size",type=int,default=8)
    args=parser.parse_args()
    if args.patient_id not in TRAIN:
        raise ValueError("Held-out or validation patient forbidden in end-to-end study")
    args.root.mkdir(parents=True,exist_ok=True)
    if args.stage in ("patches","all"):
        ensure_patch_cache(args.cache_dir)
    if args.stage in ("oof","all"):
        ensure_oof(args.cache_dir,args.root,epochs=args.epochs,device=args.device,
                   folds=args.folds,batch_size=args.batch_size)
    if args.stage in ("relations","all"):
        ensure_relations(args.root)
    pilot=None
    if args.stage in ("pilot","volume","all"):
        pilot=ensure_pilot(args.root,patient=args.patient_id)
    if args.stage in ("volume","all"):
        ensure_neural_inference(args.root,pilot=pilot,patient=args.patient_id,device=args.device)
        result=summarize(args.root,pilot=pilot,patient=args.patient_id)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    else:
        print(json.dumps({"status":"stage_complete","stage":args.stage},indent=2))


if __name__=="__main__":
    main()
