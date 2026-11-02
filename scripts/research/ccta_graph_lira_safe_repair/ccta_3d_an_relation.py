#!/usr/bin/env python3
"""Train a clean 84-feature CT28 relation head from patient-OOF neural CT scores.

This exploratory relation experiment keeps the selected geometry score
stream and PAIR/JUNCTION identity scoring frozen, but replaces ONLY the
relation head's local 'ct_score' stream with 3D CNN/ANZA candidate scores.

Crucially the CT score used for any train scene must be produced by a
candidate model trained on DIFFERENT patients (patient OOF), while 5 validation
patients use a full-train candidate model. All labels and patients stay in
the historical 17 train / 5 val CT28 split.

This produces CLEAN relation-head predictions, *not* the frozen K10/15
15-perturbation full structural gate. Do not mix its metrics with 52/0 or
promote it as an end-to-end Graph-LIRA replacement.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,log_loss

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import run_ct28_graph_lira as graph
import train_ct28_relation_head as training
from ccta_3d_an_oof import sha256

SCENES=ROOT/"artifacts/varvara/ct28_graph_lira"
SELECTED=ROOT/(
    "varvara/anza_lira_local_results/ct28_copy_count_ablations/"
    "joint_check/fusion_10_relation_15"
)
MODEL_SHA="1693b6e9b00fd74f2d443905351f534e6bdf745a5cdc00d0057823646aa09630"
TEST_IDS={954,958,972,973,980,984}


def _validated_oof_scores(
    directory:Path,variant:str,kind:str
)->pd.DataFrame:
    file=directory/f"{kind}_{variant}_oof_val.csv"
    protocol=json.loads((directory/"oof_protocol.json").read_text())
    if protocol.get("status")!="train_patient_OOF_experimental_CT_evidence_not_promoted_K10":
        raise ValueError("OOF protocol status is not the expected candidate evidence")
    if protocol["candidate_type"]!=kind or protocol["variant"]!=variant:
        raise ValueError("OOF candidate type or encoder variant mismatch")
    if protocol["score_sha256"]!=sha256(file):
        raise ValueError("OOF candidate score file checksum mismatch")
    if protocol.get("held_out_test_accessed") is not False:
        raise ValueError("Held-out test evidence present")
    if protocol["n_train_patients"]!=17 or protocol["n_validation_patients"]!=5:
        raise ValueError("OOF candidate models require all 17 train and 5 validation patients")
    for fold in protocol["fold_assignments"]:
        if set(fold["train_patients"]) & set(fold["held_patients"]):
            raise ValueError("OOF fold trained on a held patient")
        if set(fold["held_patients"]) & TEST_IDS:
            raise ValueError("Held-out test patient in OOF candidate evidence")
    frame=pd.read_csv(file)
    required={"scene_id","candidate_id","candidate_type","scan_id","split","score","oof_fold"}
    if not required <= set(frame):
        raise ValueError("Candidate scores lack the required identity/provenance")
    if frame[["scene_id","candidate_id"]].duplicated().any():
        raise ValueError("Duplicate candidate in learned score evidence")
    if set(frame.candidate_type)!={kind}:
        raise ValueError("OOF output unexpectedly mixes candidate kinds")
    if not np.isfinite(frame.score.to_numpy()).all():
        raise ValueError("Nonfinite learned CT candidate probabilities")
    if not frame.score.between(0,1).all():
        raise ValueError("Learned scores outside probability interval")
    if set(frame.scan_id.astype(int)) & TEST_IDS:
        raise ValueError("Held-out test patient in candidate scores")
    if (frame.loc[frame.split.eq("train"),"oof_fold"]<0).any() or (
        frame.loc[frame.split.eq("val"),"oof_fold"]!=-1
    ).any():
        raise ValueError("Train/validation candidate OOF fold labels incorrect")
    return frame


def substitute_neural_relation_stream(
    plans:pd.DataFrame,
    candidates:pd.DataFrame,
    baseline_scores:pd.DataFrame,
    neural_scores:pd.DataFrame,
)->tuple[list[dict],list[str],pd.DataFrame]:
    required={"scene_id","candidate_id","candidate_type"}
    if not required <= set(candidates) or not required|{"geometry_score","ct_score"} <= set(baseline_scores):
        raise ValueError("Missing baseline candidate identity or score streams")
    if not required|{"score","scan_id","split","oof_fold"} <= set(neural_scores):
        raise ValueError("Missing neural patient OOF score data")
    if neural_scores[["scene_id","candidate_id"]].duplicated().any():
        raise ValueError("Duplicate neural candidate scores")
    keys=plans[["scene_id","scan_id","split"]].copy()
    combined=candidates[["scene_id","candidate_id","candidate_type"]].merge(
        keys,on="scene_id",validate="many_to_one"
    )
    if len(combined)!=len(candidates) or combined[["scan_id","split"]].isna().any().any():
        raise ValueError("Candidate scenes do not align with train/val plan")
    merged=combined.merge(
        neural_scores,
        on=["scene_id","candidate_id","candidate_type","scan_id","split"],
        validate="one_to_one",suffixes=("","_neural"),
    )
    if len(merged)!=len(candidates):
        raise ValueError("Neural OOF scores do not cover every frozen candidate")
    if ((merged.split=="train")&(merged.oof_fold<0)).any() or (
        (merged.split=="val")&(merged.oof_fold!=-1)
    ).any():
        raise ValueError("Neural score fold provenance violates patient OOF")
    data=baseline_scores.copy()
    if data[["scene_id","candidate_id"]].duplicated().any():
        raise ValueError("Duplicate baseline local score rows")
    source=data.merge(
        merged[["scene_id","candidate_id","score"]],
        on=["scene_id","candidate_id"],validate="one_to_one"
    )
    if len(source)!=len(baseline_scores):
        raise ValueError("Baseline and neural candidate scores have different keys")
    source["ct_score"]=source["score"].astype(float)
    source=source.drop(columns=["score"])
    feature_rows,feature_columns=graph.relation_feature_rows(
        plans,candidates,source,("geometry_score","ct_score"),
    )
    if len(feature_rows)!=len(plans) or len(feature_columns)!=84:
        raise ValueError("Historical 84-feature relation representation did not reproduce")
    return feature_rows,feature_columns,source


def run_relation_training(
    *,
    pair_oof:Path,
    junction_oof:Path,
    variant:str,
    output_dir:Path,
    baseline_dir:Path=SELECTED,
    scene_dir:Path=SCENES,
)->dict:
    if variant not in {"anza3d","conv3d"}:
        raise ValueError("Expected ANZA or CNN learned CT evidence")
    if output_dir.exists():
        raise FileExistsError("Relation comparison output is immutable")
    pack_path=baseline_dir/"relation_models.joblib"
    if sha256(pack_path)!=MODEL_SHA:
        raise ValueError("Selected K10/15 HGB model SHA mismatch")
    pack=joblib.load(pack_path)
    if len(pack["ct_conditioned_relation_features"])!=84:
        raise ValueError("Frozen relation head has incorrect feature dimension")
    pair=_validated_oof_scores(pair_oof,variant,"PAIR")
    junction=_validated_oof_scores(junction_oof,variant,"JUNCTION")
    scores=pd.concat([pair,junction],ignore_index=True)
    plans=pd.read_csv(scene_dir/"scene_plan_train_val.csv")
    candidates=pd.read_csv(scene_dir/"scene_candidates_train_val.csv")
    baseline_scores=pd.read_csv(baseline_dir/"relation_local_scores_train_val.csv")
    if set(plans.scan_id.astype(int))&TEST_IDS or set(scores.scan_id.astype(int))&TEST_IDS:
        raise ValueError("No held-out test data permitted")
    if set(plans.split)!={"train","val"}:
        raise ValueError("Unexpected frozen split")
    for split,expected in training.EXPECTED_PATIENT_IDS.items():
        if set(plans.loc[plans.split==split,"scan_id"].astype(int))!=expected:
            raise ValueError("Train/val patient split is not the frozen CT28 protocol")
    features,columns,updated_scores=substitute_neural_relation_stream(
        plans,candidates,baseline_scores,scores
    )
    if columns!=list(pack["ct_conditioned_relation_features"]):
        raise ValueError("Features reordered versus frozen 84 feature-head schema")
    frame=pd.DataFrame(features)
    X=frame[columns].to_numpy(dtype=float)
    labels=frame.relation_class.map({
        name:i for i,name in enumerate(training.RELATION_CLASSES)
    })
    if labels.isna().any():
        raise ValueError("Unexpected structural relation class")
    train=frame.split.eq("train").to_numpy()
    val=frame.split.eq("val").to_numpy()
    if not train.any() or not val.any() or not np.isfinite(X).all():
        raise ValueError("Incomplete relation feature inputs")
    # Same HGB model family/hyperparameters. This uses *clean OOF* scores,
    # NOT the selected K10 15-copy relation augmentation; there is no
    # valid shortcut to reproduce augmented NN patches without retraining.
    head=clone(pack["relation_heads"]["ct_conditioned"])
    head.fit(X[train],labels[train].astype(int))
    predictions=head.predict_proba(X[val])
    classes=list(head.classes_)
    if classes!=[0,1,2,3]:
        raise ValueError("Relation head lost a structural class")
    raw=np.argmax(predictions,axis=1)
    confidence=predictions.max(axis=1)
    tau=np.where(confidence>=.85,raw,0)
    true=labels[val].to_numpy(dtype=int)
    metrics={
        "raw_relation_accuracy":float(accuracy_score(true,raw)),
        "raw_balanced_accuracy":float(balanced_accuracy_score(true,raw)),
        "raw_log_loss":float(log_loss(true,predictions,labels=[0,1,2,3])),
        "tau_relation_accuracy":float(accuracy_score(true,tau)),
        "raw_confusion":confusion_matrix(true,raw,labels=[0,1,2,3]).tolist(),
        "tau_confusion":confusion_matrix(true,tau,labels=[0,1,2,3]).tolist(),
        "n_validation_scenes":int(val.sum()),
    }
    output_dir.mkdir(parents=True,exist_ok=False)
    data=frame.loc[val,["scene_id","scan_id","scene_family","relation_class"]].copy()
    data["raw_relation"]=[training.RELATION_CLASSES[i] for i in raw]
    data["tau_relation"]=[training.RELATION_CLASSES[i] for i in tau]
    data["confidence"]=confidence
    for i,label in enumerate(training.RELATION_CLASSES):
        data["probability_"+label.lower()]=predictions[:,i]
    predictions_path=output_dir/"clean_relation_predictions_val.csv"
    data.to_csv(predictions_path,index=False)
    model_path=output_dir/"clean_relation_head.joblib"
    joblib.dump(head,model_path)
    manifest={
        "status":"experimental_clean_relation_head_no_consistency_gate",
        "variant":variant,
        "source_K10_model_sha256":MODEL_SHA,
        "pair_oof_sha256":sha256(pair_oof/f"PAIR_{variant}_oof_val.csv"),
        "junction_oof_sha256":sha256(junction_oof/f"JUNCTION_{variant}_oof_val.csv"),
        "geometry_stream":"frozen K10/15 train-OOF geometry_score",
        "image_stream":"patient-OOF learned 3D candidate scores",
        "candidate_identity":"frozen K10/15, not modified",
        "train_count":int(train.sum()),
        "validation_count":int(val.sum()),
        "relation_head_family":type(head).__name__,
        "relation_feature_count":len(columns),
        "metrics":metrics,
        "model_sha256":sha256(model_path),
        "predictions_sha256":sha256(predictions_path),
        "held_out_test_accessed":False,
        "warnings":[
            "Clean relation-head comparison only; K10/15 has 15 augmentation copies",
            "No 15x perturbation consistency, no full structural repair/mask metrics",
            "Do not promote ANZA based on clean val alone",
        ],
    }
    (output_dir/"relation_protocol.json").write_text(
        json.dumps(manifest,indent=2)+"\n",encoding="utf-8"
    )
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",choices=("conv3d","anza3d"),required=True)
    parser.add_argument("--pair-oof",type=Path,required=True)
    parser.add_argument("--junction-oof",type=Path,required=True)
    parser.add_argument("--baseline-dir",type=Path,default=SELECTED)
    parser.add_argument("--scene-dir",type=Path,default=SCENES)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    result=run_relation_training(
        pair_oof=args.pair_oof,junction_oof=args.junction_oof,
        variant=args.variant,baseline_dir=args.baseline_dir,
        scene_dir=args.scene_dir,output_dir=args.output_dir
    )
    print(json.dumps({
        "status":result["status"],"variant":result["variant"],
        "metrics":result["metrics"],
    },indent=2))


if __name__=="__main__":
    main()
