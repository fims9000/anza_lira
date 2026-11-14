"""V34: new native 5-conv P0 on THEBE real ANZA-predicted SBPP candidates.

Uses EXACT passed V33 ANZA checkpoint and topology threshold .65, no
segmentation retraining, no oracle pair injection and no Thebe TEST7
images/masks/scores. U-Net stays STOP at V33 candidate feasibility gate.
CAL here uses VAL1 sections 40..99, physically same Thebe 3D volume:
one-sided section-CP assumptions are NOT validated natural geology risk.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from scripts.tver_v33_thebe_train_segmenters import read_memmap
from scripts.tver_v33_thebe_relation_cal import (
    _candidate_scenes,_load_model,_full_train_logistic,
    _section_calibrate,_save_cal_cases,
    TRAIN_SOURCE_IDS,RELATION_CAL_SOURCE_IDS,
)
from scripts.tver_v30_p0_context import (
    SEEDS,paired_training_examples,fit_native_p0,fit_matched_logistic,
)
from scripts.tver_v30_real_p0_e2e import _scores,MODEL_NAMES

FROZEN_ANZA_SHA="ba0c3b8060fd0e3cb1c15f6c1598b94a977677c2b075663b5414207df382c25e"
FROZEN_UNET_SHA="93f91fbc24e90b8ed02346fcab3c535f2b4348397d1ae1be9bbd09e400f9ef6b"
FROZEN_ANZA_TOPOLOGY=.65
UNET_V33_POLICY="STOP_NO_TRAIN_CAL_CORRECT_PREDICTED_SBPP"


def _state_report(frontends_dir):
    p=Path(frontends_dir)/"v33_THEBE_frontends_topology_TRAIN_CAL_frozen.json"
    if not p.is_file():raise RuntimeError("STOP missing successful V33 source receipt")
    report=json.loads(p.read_text())
    a=report["frontends"]["anza_v1"]
    u=report["frontends"]["unet"]
    if (report["status"]!="V33_STOP_THEBE_NEW_FRONTEND_PREDICTED_SBPP_INSUFFICIENT"
        or a["SHA256"]!=FROZEN_ANZA_SHA
        or a["topology_threshold_frozen"]!=FROZEN_ANZA_TOPOLOGY
        or a["topology_status"]!="TRAIN_CAL_FEASIBLE"
        or u["SHA256"]!=FROZEN_UNET_SHA
        or u["topology_status"]!=UNET_V33_POLICY
        or u["topology_threshold_frozen"] is not None
        or report["SEG_CAL_section_indices"]!=list(range(30))
        or report["REL_CAL_section_indices_locked_unopened"]!=list(range(40,100))
        or not report["heldout_THEBE_TEST7_NOT_OPENED"]):
        raise RuntimeError("STOP V34 source/weights/topology provenance changed")
    return report


def execute(memmap_root,frontends_dir,out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    meta=_state_report(frontends_dir)
    original_receipt=(Path(frontends_dir)/
       "v33_THEBE_frontends_topology_TRAIN_CAL_frozen.json")
    report={
        "status":"V34_ANZA_THEBE_P0_RELATION_STARTED",
        "source":"real external Thebe Gas Field expert boolean fault masks",
        "frozen_V33_original_topology_receipt_SHA256":
            hashlib.sha256(original_receipt.read_bytes()).hexdigest(),
        "frontends_admission":{
            "anza_v1":"V33_TRAIN_CAL_TOPOLOGY_PASS_SELECTED_FOR_EXPLORATORY_P0",
            "unet":"V33_FAILED_TRAIN_CORRECT_CANDIDATE_FROZEN_EXCLUDED",
        },
        "ANZA_exact_checkpoint_SHA256":FROZEN_ANZA_SHA,
        "ANZA_exact_topology_threshold":FROZEN_ANZA_TOPOLOGY,
        "no_ANZA_U_NET_retraining_or_new_SEG_CAL_tuning":True,
        "genuine_original_P0_CNN_architecture_new_weights_not_historical":True,
        "TRAIN_source_sections":list(TRAIN_SOURCE_IDS),
        "RELATION_CAL_source_sections":list(RELATION_CAL_SOURCE_IDS),
        "CAL_40_99_source_availability_was_sampled_without_model_outcomes_in_V32":True,
        "external_test7_images_labels_model_scores_NEVER_OPENED":True,
        "hard_source_lineage_is_internal_to_expert_undivided_trace":True,
        "no_natural_geological_instance_identity":True,
    }
    train_x=read_memmap(memmap_root,"train_seismic_float32.npy")
    train_y=read_memmap(memmap_root,"train_fault_bool.npy")
    val_x=read_memmap(memmap_root,"val_seismic_float32.npy")
    val_y=read_memmap(memmap_root,"val_fault_bool.npy")
    model=_load_model(frontends_dir,"anza_v1",meta)
    threshold=FROZEN_ANZA_TOPOLOGY
    train,train_sources=_candidate_scenes(
        train_x,train_y,TRAIN_SOURCE_IDS,meta["normalization"],model,threshold)
    n_positive_train=sum(
        scene["has_true_proposal"] for scene in train if scene["kind"]=="LINK")
    report["TRAIN_actual_predicted_source_gap_count"]=len(train)//2
    report["TRAIN_correct_candidate_available_gap_count"]=n_positive_train
    report["TRAIN_sources_with_candidate_gaps"]=train_sources
    try:
        if n_positive_train<8:
            raise RuntimeError("STOP fewer than eight correct predicted TRAIN candidates")
        images,labels,features,passport=paired_training_examples(train,threshold)
        full,full_receipt=_full_train_logistic(train)
        matched=fit_matched_logistic(features,labels)
    except (RuntimeError,ValueError) as exc:
        report["status"]="V34_STOP_TRUE_PREDICTED_TRAIN_PAIR_INSUFFICIENT"
        report["reason"]=str(exc)
        (out/"v34_THEBE_P0_ANZA_source_CAL_frozen.json").write_text(
            json.dumps(report,indent=2)+"\n")
        return report
    models={
        "geometry":None,
        "full_train_logistic":full,
        "matched_logistic":matched,
    }
    p0_evidence={}
    for seed in SEEDS:
        neural,receipt=fit_native_p0(
            images,labels,seed=seed,
            root=out/"native_new_P0_weights",
            frontend="anza_v1")
        models[f"p0_seed{seed}"]=neural
        p0_evidence[str(seed)]=receipt
    report["TRAIN_balanced_predicted_only_candidate_pairs"]=passport
    report["TRAIN_full_negative_budget_logistic"]=full_receipt
    report["TRAIN_native_P0_two_distinct_checkpoint_receipts"]=p0_evidence
    # First and only access to REL-CAL 40..99 for this study happens now.
    cal,cal_sources=_candidate_scenes(
        val_x,val_y,RELATION_CAL_SOURCE_IDS,meta["normalization"],model,threshold)
    report["RELATION_CAL_sources"]=cal_sources
    report["RELATION_CAL_actual_positive_gap_count"]=sum(
        s["kind"]=="LINK" for s in cal)
    report["RELATION_CAL_actual_synthetic_NO_LINK_count"]=sum(
        s["kind"]=="NO_LINK" for s in cal)
    policies={};rows=[]
    for method in MODEL_NAMES:
        scores=[_scores(s,method,models[method],threshold) for s in cal]
        cal_receipt=_section_calibrate(cal,scores)
        policies[method]=cal_receipt
        rows.extend(_save_cal_cases(
            out,"anza_v1",method,cal,scores,cal_receipt["accept_threshold"]))
    report["RELATION_CAL_frozen_policies"]=policies
    if rows:
        with (out/"v34_THEBE_ANZA_P0_CAL_case_rows.csv").open(
            "w",encoding="utf8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    report["status"]=(
        "V34_THEBE_ANZA_NATIVE_P0_AND_LOGISTIC_SECTION_CAL_COMPLETE_TEST_UNOPENED"
        if all(v["status"]=="CAL_SECTION_RISK_CONDITIONALLY_FROZEN"
               for v in policies.values()) else
        "V34_STOP_THEBE_ANZA_RELATION_SECTION_CAL_INSUFFICIENT")
    report["interpretation"]={
        "this_is_selected_frontend_EXPLORATORY_after_SEG_CAL":True,
        "synthetic_NO_LINK_is_NOT_real_geological_wrong_identity":True,
        "section_CP_independence_not_verified_within_same_Thebe_field":True,
        "future_one_shot_TEST7_not_authorized_by_THIS_result":True,
        "zero_accepted_policies_have_zero_recovery_not_high_performance":True,
        "historical_ANZA_LIRA_original_weights_absent":True,
    }
    outpath=out/"v34_THEBE_P0_ANZA_source_CAL_frozen.json"
    outpath.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({
        "status":report["status"],
        "predicted_TRAIN_positive_gap_availability":n_positive_train,
        "n_relation_CAL_distinct_sections":len(set(
            s["section"] for s in cal if s["kind"]=="LINK")),
        "risk_CAL_method_policy":{
            k:{"status":v["status"],
               "selected":v["accept_threshold"],
               "n_sections":v.get("source_section_count",v.get("n_sections"))}
            for k,v in policies.items()},
        "external_TEST7_opened":False,
    },indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mmap-root",required=True,type=Path)
    p.add_argument("--frontends-dir",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    a=p.parse_args()
    execute(a.mmap_root,a.frontends_dir,a.out_dir)
