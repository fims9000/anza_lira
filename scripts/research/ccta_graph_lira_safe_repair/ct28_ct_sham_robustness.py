#!/usr/bin/env python3
"""Repeated patient-wise CT-feature permutation controls for frozen CT28 candidates.

Pre-declared 25 permuted CT controls, each with the *same* model family,
feature dimension, training patients, endpoint geometry and val patients as
the aligned-CT model. This tests sensitivity to CT/candidate spatial alignment.

One repeated shuffle is not a proof of causality, and multiple shuffles on
the same five post-selected development-val patients are not an independent
p-value; report the observed sham distribution descriptively.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score

from ct28_ct_signal_benchmark import (
    DEFAULT_SOURCE,KINDS,TRAIN,VAL,
    read_frozen,expected_columns,make_matrix,fit_predict,sha,
)


def describe(values:list[float])->dict:
    a=np.asarray(values,dtype=float)
    if a.size==0 or not np.isfinite(a).all():
        raise ValueError("No finite shuffled CT metric values")
    return {
        "repetitions":int(a.size),
        "mean":float(a.mean()),
        "min":float(a.min()),
        "median":float(np.median(a)),
        "max":float(a.max()),
        "p05":float(np.quantile(a,.05)),
        "p95":float(np.quantile(a,.95)),
    }


def run_repeated_sham(
    *,
    input_csv:Path,
    output_dir:Path,
    repetitions:int=25,
    seed:int=20261102,
    max_iter:int=100,
)->dict:
    if not (3<=repetitions<=200) or max_iter<10:
        raise ValueError("Sham count 3..200; HGB iterations >=10")
    if output_dir.exists():
        raise FileExistsError("Sham research outputs are immutable")
    frame=read_frozen(input_csv)
    report={}
    rows=[]
    for kind in KINDS:
        subset=frame[frame.candidate_type.eq(kind)].copy()
        geometry,ct=expected_columns(subset,kind)
        train=subset.split.eq("train").to_numpy()
        val=subset.split.eq("val").to_numpy()
        ytrain=subset.loc[train,"y"].to_numpy(dtype=int)
        yval=subset.loc[val,"y"].to_numpy(dtype=int)
        val_patients=subset.loc[val,"scan_id"].to_numpy(dtype=int)
        if set(val_patients)!=VAL:
            raise ValueError("Validation patient count changed")
        x=make_matrix(
            subset,geometry=geometry,ct=ct,
            variant="geometry_plus_ct",seed=seed,
        )
        reference,reference_seconds=fit_predict(
            x[train],ytrain,x[val],max_iter=max_iter
        )
        real_ap=float(average_precision_score(yval,reference))
        real_auc=float(roc_auc_score(yval,reference))
        sham_ap=[]
        sham_auc=[]
        sham_per_patient_delta={str(patient):[] for patient in sorted(VAL)}
        for repetition in range(repetitions):
            control_seed=seed+1009*(repetition+1)
            permuted=make_matrix(
                subset,geometry=geometry,ct=ct,
                variant="geometry_plus_shuffled_ct",
                seed=control_seed,
            )
            prediction,duration=fit_predict(
                permuted[train],ytrain,permuted[val],max_iter=max_iter
            )
            ap=float(average_precision_score(yval,prediction))
            auc=float(roc_auc_score(yval,prediction))
            sham_ap.append(ap)
            sham_auc.append(auc)
            patient_diffs={}
            for patient in sorted(VAL):
                choose=val_patients==patient
                yy=yval[choose]
                if len(np.unique(yy))!=2:
                    raise ValueError("Insufficient true/false candidates for patient-specific AP")
                diff=(
                    average_precision_score(yy,reference[choose])
                    -average_precision_score(yy,prediction[choose])
                )
                patient_diffs[str(patient)]=float(diff)
                sham_per_patient_delta[str(patient)].append(float(diff))
            rows.append({
                "candidate_type":kind,
                "repeat":repetition,
                "control_seed":control_seed,
                "AP":ap,"AUROC":auc,
                "real_AP":real_ap,
                "delta_AP_real_minus_sham":real_ap-ap,
                "model_train_infer_seconds":duration,
                **{"patient_"+p+"_delta_AP":v for p,v in patient_diffs.items()},
            })
        report[kind]={
            "val_candidates":len(yval),
            "val_positive":int(yval.sum()),
            "aligned_real_CT": {
                "AP":real_ap,"AUROC":real_auc,
                "model_train_infer_seconds":reference_seconds,
            },
            "patientwise_sham_CT":{
                "AP":describe(sham_ap),
                "AUROC":describe(sham_auc),
                "per_patient_real_minus_sham_AP":{
                    patient:describe(deltas)
                    for patient,deltas in sham_per_patient_delta.items()
                },
                "repetitions_with_sham_AP_at_least_real":sum(x>=real_ap for x in sham_ap),
                "description":"Preserve each patient's joint CT-feature vectors while permuting assignments among candidates",
            },
        }
    output_dir.mkdir(parents=True,exist_ok=False)
    csvfile=output_dir/"sham_repetitions.csv"
    with csvfile.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result={
        "status":"completed_frozen_CT28_precomputed_real_CT_vs_repeated_sham_development_val",
        "source_feature_sha256":sha(input_csv),
        "repetitions":repetitions,
        "seed":seed,
        "max_iter":max_iter,
        "held_out_test_accessed":False,
        "train_patients":sorted(TRAIN),
        "val_patients":sorted(VAL),
        "results":report,
        "sham_csv_sha256":sha(csvfile),
        "interpretation_limitations":[
            "Descriptive negative-control distribution on previously selected five-patient validation; not an independent formal significance test.",
            "Patientwise joint CT permutations preserve patient distribution but intentionally destroy CCTA spatial alignment.",
            "Outcomes are candidate AP/AUC from saved CCTA-derived numeric features; no raw 3D CT was processed.",
            "No information here establishes superiority of AZConv3d over CNN or safety of 3D mask repair.",
        ],
    }
    (output_dir/"sham_robustness.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"
    )
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=DEFAULT_SOURCE)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--repetitions",type=int,default=25)
    p.add_argument("--seed",type=int,default=20261102)
    p.add_argument("--max-iter",type=int,default=100)
    args=p.parse_args()
    r=run_repeated_sham(
        input_csv=args.source,output_dir=args.output_dir,
        repetitions=args.repetitions,seed=args.seed,max_iter=args.max_iter,
    )
    print(f"CT28 feature source sha256={r['source_feature_sha256']}")
    for kind in KINDS:
        s=r["results"][kind]
        original=s["aligned_real_CT"]
        sham=s["patientwise_sham_CT"]
        print(
            f"{kind}: real AP={original['AP']:.6f}; "
            f"sham AP mean={sham['AP']['mean']:.6f} "
            f"range=[{sham['AP']['min']:.6f}, {sham['AP']['max']:.6f}]; "
            f"sham>=real {sham['repetitions_with_sham_AP_at_least_real']}/{r['repetitions']}"
        )
        for patient,details in sham["per_patient_real_minus_sham_AP"].items():
            print(f" {kind} patient {patient}: real minus sham AP mean={details['mean']:+.6f}")
    print(f"report SHA256={sha(args.output_dir/'sham_robustness.json')}")


if __name__=="__main__":
    main()
