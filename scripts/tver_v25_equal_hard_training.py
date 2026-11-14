"""Tver V25: genuinely equal-hard-negative TRAIN for learned 2.5D baselines.

Train all supervised models on 200 NEW independent triplets with switched
NO_LINK negatives. Use another 100 triplets to CAL selective risk. Then, and
only then, open an entirely NEW 160-triplet TEST. The unchanged V24
two-neighbor consensus rule is compared against (i) same-observation HGB,
(ii) genuinely trained image+oracle-endpoint 2.5D CNN, two seeds 11/23,
and (iii) 2D controls. Historical ANZA/SBPP/LIRA weights are NOT loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.tver_v24_switched_neighbor_synthetic import (
    render_hard_v24,_visible_scores,
)
from scripts.tver_v23_25d_counterfactual_synthetic import (
    _fit_learned,_calibrate,_predict,_metric,
    _paired_base_bootstrap,
)
from scripts.tver_v25_cnn import (
    TORCH_SEEDS,fit_cnn_v25,score_cnn_v25,
    train_image_statistics,
)

BASE={"train":202683010,"cal":202693010,"test":202703010}
N={"train":200,"cal":100,"test":160}
HEUR=("mean_v23","consensus_v24")
CLASSIC=("learned_2d_hgb","learned_25d_hgb")
CONV=tuple(
    f"cnn_{scope}_seed{seed}"
    for seed in TORCH_SEEDS for scope in ("2d","25d"))
METHODS=HEUR+CLASSIC+CONV
PROTOCOL="TV_V25_EQUAL_SWITCHED_NEGATIVE_TRAIN_CNN_HGB"


def _build(split):
    rows=[]
    for i in range(N[split]):
        for scene in render_hard_v24(
            i,base_seed=BASE[split],total_groups=N[split]):
            features=_visible_scores(scene)
            rows.append({
                "base":scene["base"],"world":scene["world"],
                "label":int(scene["label"]),
                "central_sha256":hashlib.sha256(
                    scene["central"].tobytes()).hexdigest(),
                "near_minus_sha256":hashlib.sha256(
                    scene["adjacent"][0].tobytes()).hexdigest(),
                "near_plus_sha256":hashlib.sha256(
                    scene["adjacent"][1].tobytes()).hexdigest(),
                "scene":scene,"features":features,
                "center_features":features["center_features"],
                "all_features":features["all_features"],
            })
    groups=defaultdict(list)
    for row in rows:groups[row["base"]].append(row)
    if len(groups)!=N[split]:
        raise AssertionError("Nuisance base cohort size drift")
    for base,cases in groups.items():
        if (len(cases)!=3 or set(r["label"] for r in cases)!={0,1,2}
            or len(set(r["central_sha256"] for r in cases))!=1):
            raise AssertionError("Broken 3-world exact central counterfactual")
        neg=next(r for r in cases if r["label"]==2)
        positives=[r for r in cases if r["label"]!=2]
        nminus=[r for r in positives
                if r["near_minus_sha256"]==neg["near_minus_sha256"]]
        nplus=[r for r in positives
               if r["near_plus_sha256"]==neg["near_plus_sha256"]]
        if (len(nminus)!=1 or len(nplus)!=1 or
            nminus[0]["label"]==nplus[0]["label"]):
            raise AssertionError("Hybrid negative has a single-slice marginal leakage")
    return rows


def _scores(rows,hgb,convs,mean,std):
    out={name:[] for name in METHODS}
    for row in rows:
        f=row["features"]
        out["mean_v23"].append(f["continuity_25d"])
        out["consensus_v24"].append(f["consensus_v24"])
        for method,orig in (
            ("learned_2d_hgb","learned_2d"),
            ("learned_25d_hgb","learned_25d"),
        ):
            model,key=hgb[orig]
            p=model.predict_proba(row[key][None,:])[0]
            c=dict(zip(model.classes_,p))
            out[method].append((c[0]-c[2],c[1]-c[2]))
    for name,(model,mode) in convs.items():
        out[name]=score_cnn_v25(
            model,[r["scene"] for r in rows],
            mode=mode,mean=mean,std=std)
    for key,value in out.items():
        arr=np.asarray(value,float)
        if arr.shape!=(len(rows),2) or not np.isfinite(arr).all():
            raise AssertionError(f"Source-only prediction wrong {key}")
        out[key]=arr
    return out


def execute(out):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    if any(set(range(BASE[a],BASE[a]+N[a]))&
           set(range(BASE[b],BASE[b]+N[b]))
           for a in BASE for b in BASE if a!=b):
        raise AssertionError("Train/Cal/TEST nuisance base overlap")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train=_build("train")
    cal=_build("cal")
    hgb=_fit_learned(train)
    mu,sd=train_image_statistics([r["scene"] for r in train])
    cnns={}
    passports={}
    for seed in TORCH_SEEDS:
        for scope,mode in (("2d","central_2d"),
                           ("25d","neighbors_25d")):
            name=f"cnn_{scope}_seed{seed}"
            model,receipt=fit_cnn_v25(
                [r["scene"] for r in train],
                mode=mode,seed=seed,mean=mu,std=sd,
                output_dir=out/"models")
            cnns[name]=(model,mode)
            passports[name]=receipt
    # True stochastic distinct init fingerprints (can still converge,
    # which is an outcome to report; do not assume distinct performance).
    if (passports["cnn_25d_seed11"]["checkpoint_SHA256"]==
        passports["cnn_25d_seed23"]["checkpoint_SHA256"]):
        raise AssertionError("Two declared CNN seeds duplicated one set of trained parameters")
    cal_scores=_scores(cal,hgb,cnns,mu,sd)
    ycal=np.asarray([r["label"] for r in cal])
    thresholds={m:_calibrate(ycal,cal_scores[m]) for m in METHODS}
    frozen={
        "status":"V25_EQUAL_HARD_TRAIN_CAL_LOCKED_BEFORE_TEST",
        "protocol":PROTOCOL,"train_base_start":BASE["train"],
        "cal_base_start":BASE["cal"],
        "reserved_unopened_test_base_start":BASE["test"],
        "registered_N_base_groups":N,
        "risk_CAL_false_accept_max":.02,
        "risk_CAL_wrong_branch_max":.01,
        "registered_CNN_seeds":list(TORCH_SEEDS),
        "two_seeds_are_NOT_duplicated_TEST_people_or_base_groups":True,
        "observed_only_CNN_train_global_mean":mu,
        "observed_only_CNN_train_global_std":sd,
        "CNN_CE_train_passports":passports,
        "HGB_same_observed_features_trained_on_equal_hard_NOLINK":True,
        "CNN_only_fits_from_same_600_hard_TRAIN_worlds":True,
        "cal_acceptance_thresholds":thresholds,
        "V25_TEST_exposed_at_lock_time":False,
        "no_real_seismic_or_historical_ANZA_LIRA_E2E":True,
    }
    lock=out/"v25_train_cal_frozen.json"
    lock.write_text(json.dumps(frozen,indent=2)+"\n")
    lock_sha=hashlib.sha256(lock.read_bytes()).hexdigest()
    # Exactly FIRST render of fresh test after all CNN/HGB learned weights
    # and every acceptance/margin risk threshold are written and sealed.
    test=_build("test")
    scores=_scores(test,hgb,cnns,mu,sd)
    predictions={name:_predict(scores[name],thresholds[name])
                 for name in METHODS}
    reports={name:_metric(test,predictions[name]) for name in METHODS}
    contrasts={
        name:_paired_base_bootstrap(
            test,predictions["consensus_v24"],predictions[name])
        for name in METHODS if name!="consensus_v24"}
    c=reports["consensus_v24"]
    comparators=("learned_25d_hgb","cnn_25d_seed11","cnn_25d_seed23")
    beyond_all=all(
        contrasts[k]["base_group_cluster_CI95"][0]>0
        for k in comparators)
    admission=bool(
        c["false_bridge_rate"]<=.02 and
        c["wrong_branch_rate"]<=.01 and
        c["positive_correct_accepted_rate"]>=.15 and
        beyond_all)
    csv_path=out/"v25_case_scores.csv"
    evidence=[]
    for i,row in enumerate(test):
        item={x:row[x] for x in (
            "base","world","label","central_sha256",
            "near_minus_sha256","near_plus_sha256")}
        for name in METHODS:
            item[name+"_choice"]=int(predictions[name][i])
            item[name+"_score0"]=float(scores[name][i,0])
            item[name+"_score1"]=float(scores[name][i,1])
        evidence.append(item)
    with csv_path.open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(evidence[0]))
        writer.writeheader();writer.writerows(evidence)
    report={
        "status":"EXECUTED_V25_EQUAL_HARD_NEGATIVE_TRAIN",
        "protocol":PROTOCOL,
        "prior_V24_and_V23_heldout_outcomes_not_reused_for_model_selection":True,
        "all_train_CAL_and_TEST_hard_negative_distribution_equal":True,
        "actual_historical_ANZA_LIRA_not_loaded":True,
        "TEST_generated_after_CNN_HGB_and_CAL_lock":True,
        "n_independent_test_base_groups":N["test"],
        "n_counterfactual_test_worlds":len(test),
        "train_CAL_sha256":lock_sha,
        "method_results":reports,
        "consensus_minus_all_methods_group_CI":contrasts,
        "strict_V25_SUPERIORITY_to_all_equal_hard_training_baselines":admission,
        "strict_V25_empirical_error_budget_met":bool(
            c["false_bridge_rate"]<=.02 and
            c["wrong_branch_rate"]<=.01 and
            c["positive_correct_accepted_rate"]>=.15),
        "case_evidence_sha256":hashlib.sha256(
            csv_path.read_bytes()).hexdigest(),
        "models_have_oracle_endpoints":True,
        "real_geological_fault_instances_NOT_tested":True,
        "published_3D_seismic_CNNs_not_run":True,
    }
    (out/"v25_results.json").write_text(
        json.dumps(report,indent=2)+"\n")
    print(json.dumps({
        "status":report["status"],
        "superiority_to_all_same_train":admission,
        "selected_results":{
            name:{key:row[key] for key in (
                "correct_accepted","wrong_branch_accepted",
                "false_accepted","all_3way_accuracy")}
            for name,row in reports.items()},
        "consensus_v_HGB_group_interval":
            contrasts["learned_25d_hgb"]["base_group_cluster_CI95"],
        "consensus_v_CNN11_group_interval":
            contrasts["cnn_25d_seed11"]["base_group_cluster_CI95"],
        "consensus_v_CNN23_group_interval":
            contrasts["cnn_25d_seed23"]["base_group_cluster_CI95"],
    },indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    execute(a.out)
