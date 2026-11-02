#!/usr/bin/env python3
"""Auditable count-only CT28 structural-relation shortcut baseline.

Uses ONLY the committed 17 train + 5 validation scene plan, NOT any CT,
segmentations, model weights, test labels, or CT-derived features. Critical
negative control against cardinality shortcuts in the 84-feature HGB head.

The benchmark is a train-count lookup -> train-majority relation, with a
global train-majority fallback on previously unseen val count pairs. No
validation label is used to fit the classifier or select a hyperparameter.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random

import pandas as pd
from sklearn.metrics import (
    accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,
)

ROOT=Path(__file__).resolve().parents[3]
SCENES=ROOT/"artifacts/varvara/ct28_graph_lira/scene_plan_train_val.csv"
TRAIN_IDS={953,955,956,959,960,963,964,967,969,970,971,975,976,977,979,982,983}
VAL_IDS={957,961,965,966,974}
TEST_IDS={954,958,972,973,980,984}
CLASSES=("NONE","PAIR","JUNCTION","BOTH")


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as data:
        for chunk in iter(lambda:data.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen(path:Path=SCENES)->tuple[pd.DataFrame,pd.DataFrame]:
    frame=pd.read_csv(path)
    required={"scene_id","scan_id","split","relation_class",
              "pair_candidate_count","junction_candidate_count"}
    if not required<=set(frame):
        raise ValueError("Missing canonical CT28 scene-plan columns")
    if frame.scene_id.duplicated().any() or frame.isna().any().any() and (
        frame[list(required)].isna().any().any()
    ):
        raise ValueError("Duplicate or missing canonical scenes/required columns")
    if not set(frame.relation_class)<=set(CLASSES):
        raise ValueError("Unexpected structural class")
    if set(frame.split)!={"train","val"}:
        raise ValueError("Only frozen train/validation scene plan allowed")
    train=frame[frame.split.eq("train")].copy()
    val=frame[frame.split.eq("val")].copy()
    if set(train.scan_id.astype(int))!=TRAIN_IDS or set(val.scan_id.astype(int))!=VAL_IDS:
        raise ValueError("Patient split differs; held-out access remains blocked")
    for col in ("pair_candidate_count","junction_candidate_count"):
        if frame[col].lt(0).any() or not frame[col].notna().all():
            raise ValueError("Invalid candidate cardinality")
    if len(train)!=1036 or len(val)!=368:
        raise ValueError("Unexpected train/validation scene count")
    return train,val


def majority(label_counts:Counter)->str:
    # A deterministic tie policy, fixed before viewing validation labels.
    return max(CLASSES,key=lambda label:(label_counts[label],-CLASSES.index(label)))


def compute_baseline(
    train:pd.DataFrame,val:pd.DataFrame,*,bootstrap:int=5000,seed:int=20261102
)->tuple[dict,list[dict]]:
    if bootstrap<100:
        raise ValueError("At least 100 bootstrap draws required")
    contingency=defaultdict(Counter)
    for row in train.itertuples(index=False):
        contingency[(int(row.pair_candidate_count),
                     int(row.junction_candidate_count))][row.relation_class]+=1
    fallback=majority(Counter(train.relation_class))
    output=[]
    for row in val.itertuples(index=False):
        pair=int(row.pair_candidate_count); junction=int(row.junction_candidate_count)
        pattern=(pair,junction)
        seen=pattern in contingency
        label=majority(contingency[pattern]) if seen else fallback
        counts=contingency[pattern] if seen else Counter()
        ambiguity=len([x for x in CLASSES if counts[x]>0])>1
        output.append({
            "scene_id":row.scene_id,"scan_id":int(row.scan_id),
            "true_relation":row.relation_class,"pred_relation":label,
            "pair_candidate_count":pair,"junction_candidate_count":junction,
            "pattern_seen_in_train":int(seen),
            "train_pattern_frequency":sum(counts.values()),
            "train_pattern_has_multiple_classes":int(ambiguity),
            "correct":int(label==row.relation_class),
        })
    actual=[row["true_relation"] for row in output]
    predicted=[row["pred_relation"] for row in output]
    patients=sorted({row["scan_id"] for row in output})
    by_patient={}
    for pid in patients:
        subset=[row for row in output if row["scan_id"]==pid]
        by_patient[str(pid)]={
            "scenes":len(subset),
            "correct":sum(row["correct"] for row in subset),
            "accuracy":sum(row["correct"] for row in subset)/len(subset),
        }
    rng=random.Random(seed)
    resampled=[]
    patient_groups={
        pid:[x for x in output if x["scan_id"]==pid]
        for pid in patients
    }
    for _ in range(bootstrap):
        group=[item for _ in patients for item in patient_groups[rng.choice(patients)]]
        resampled.append(sum(row["correct"] for row in group)/len(group))
    resampled.sort()
    def quantile(frac):
        at=(len(resampled)-1)*frac
        a=int(at); b=min(a+1,len(resampled)-1)
        return resampled[a]+(at-a)*(resampled[b]-resampled[a])
    # Conditional class entropy of each count pattern (estimated from train).
    pattern_stats=[]
    for (pair,junction),c in sorted(contingency.items()):
        n=sum(c.values())
        entropy=-sum((k/n)*math.log(k/n,2) for k in c.values() if k)
        pattern_stats.append({
            "pair_count":pair,
            "junction_count":junction,
            "train_n":n,
            "train_label_entropy_bits":entropy,
            "train_classes":dict(c),
        })
    report={
        "status":"negative_control_train_count_only_val",
        "source":"frozen_CT28_train_val_scene_plan",
        "train_scenes":len(train),
        "validation_scenes":len(val),
        "train_subjects":len(set(train.scan_id)),
        "validation_subjects":len(patients),
        "fallback_relation":fallback,
        "training_count_patterns":len(contingency),
        "validation_patterns_seen_in_train":sum(x["pattern_seen_in_train"] for x in output),
        "validation_patterns_unseen_in_train":sum(not x["pattern_seen_in_train"] for x in output),
        "validation_accuracy":float(accuracy_score(actual,predicted)),
        "validation_correct":int(sum(x["correct"] for x in output)),
        "validation_balanced_accuracy":float(balanced_accuracy_score(actual,predicted)),
        "validation_macro_f1":float(f1_score(actual,predicted,labels=list(CLASSES),average="macro",zero_division=0)),
        "confusion_matrix_class_order":list(CLASSES),
        "validation_confusion_matrix":confusion_matrix(actual,predicted,labels=list(CLASSES)).tolist(),
        "per_patient":by_patient,
        "patient_cluster_bootstrap_accuracy_descriptive_95pct":[quantile(.025),quantile(.975)],
        "bootstrap_draws":bootstrap,
        "bootstrap_seed":seed,
        "train_count_patterns":pattern_stats,
        "held_out_test_accessed":False,
        "claim_limits":[
            "Count-only performance is generator shortcut evidence, not clinical utility.",
            "Validation is shared with the K10/15 model-selection process; descriptive only.",
            "This predicts NONE/PAIR/JUNCTION/BOTH, not candidate identity or 3D mask repair.",
            "Patient-level bootstrap uses only five validation subjects and is not confirmatory.",
        ],
    }
    return report,output


def write_report(output_dir:Path,report:dict,rows:list[dict]):
    if output_dir.exists():
        raise FileExistsError("Immutable benchmark output directory exists")
    output_dir.mkdir(parents=True,exist_ok=False)
    (output_dir/"count_only_baseline.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    with (output_dir/"count_only_val_predictions.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-plan",type=Path,default=SCENES)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--bootstrap",type=int,default=5000)
    parser.add_argument("--seed",type=int,default=20261102)
    args=parser.parse_args()
    train,val=load_frozen(args.scene_plan)
    report,rows=compute_baseline(train,val,bootstrap=args.bootstrap,seed=args.seed)
    report["scene_plan_sha256"]=sha(args.scene_plan)
    write_report(args.output_dir,report,rows)
    print(json.dumps({
        "status":report["status"],"val_accuracy":report["validation_accuracy"],
        "correct":report["validation_correct"],
        "n":report["validation_scenes"],
        "macro_f1":report["validation_macro_f1"],
        "unseen_count_patterns":report["validation_patterns_unseen_in_train"],
    },indent=2))


if __name__=="__main__":
    main()
