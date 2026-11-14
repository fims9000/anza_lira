"""V33 external Thebe 5-convolution P0 relation training and independent CAL.

Input source: real external Thebe TRAIN1 expert faults and seismic images,
new matched AZConv-v1 / U-Net checkpoints from V33 segmentation stage.
No Thebe TEST file/path/array opened anywhere in this module.

The expert centerline determines ONLY where to erase predicted segmentation
and how to label a candidate after inference; P0 sees original seismic and
SBPP-predicted source/candidate ports, never expert centerline/endpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta
import torch

from models.segmentation_v2 import build_comparable_model
from scripts.tver_v26_train_real_segmenters import predict_patch
from scripts.tver_v26_predicted_cracks_e2e import (
    _world,sbpp_proposals,is_true_relation,_path_success,
)
from scripts.tver_v30_p0_context import (
    SEEDS,paired_training_examples,fit_native_p0,
    fit_matched_logistic,context_scores,score_logistic,
)
from scripts.tver_v30_real_p0_e2e import (
    _scores,_decision,MODEL_NAMES,CAL_THRESHOLDS,
)
from scripts.tver_v33_thebe_train_segmenters import (
    read_memmap,_gaps,_roi_gray,PARAM_WIDTHS,
)

FRONTENDS=("unet","anza_v1")
TRAIN_SOURCE_IDS=tuple(range(100))
RELATION_CAL_SOURCE_IDS=tuple(range(40,100))
RELATION_CAL_MIN_ELIGIBLE_SECTIONS=29
FROZEN_SECTION_MAX_GAPS=2
MAX_SECTION_RISK=.10
ONE_SIDED_CONFIDENCE=.95


def upper_clopper_pearson(k,n,confidence=ONE_SIDED_CONFIDENCE):
    """95% one-sided exact-binomial upper CI (conditional on independent sections)."""
    if n<=0 or not 0<=k<=n:
        raise ValueError("Invalid distinct section-event numerator/denominator")
    return 1. if k==n else float(beta.ppf(confidence,k+1,n-k))


def _candidate_scenes(seismic,expert,sections,scale,segmenter,threshold):
    """Two artificial expert-selected interventions per source section.

    World creation only receives expert truth for controlled erasure; candidate
    generation uses a predicted mask and real seismic pixels ONLY. Labels
    attached after inference, for TRAIN/CAL and never as network input.
    """
    scenes=[]
    source_eligibility=[]
    for sid in sections:
        mask=np.asarray(expert[sid],bool)
        gaps=_gaps(mask,sid)
        source_eligibility.append({
            "section":int(sid),"reference_gap_count":len(gaps),
            "eligible":bool(gaps)})
        for gap in gaps[:FROZEN_SECTION_MAX_GAPS]:
            rgb,origin,rect=_roi_gray(seismic[sid],gap["rectangle"],scale)
            prob=predict_patch(
                segmenter,rgb,
                [scale["normalized_channel_mean"]]*3,
                [scale["normalized_channel_std"]]*3)
            for world in ("LINK","NO_LINK"):
                visible=_world(prob,rect,gap["true_ends"],origin,world)
                pred=sbpp_proposals(visible,rgb.mean(axis=2),rect,threshold)
                proposals=pred["proposals"]
                for pair in proposals:
                    pair["y"]=int(world=="LINK" and is_true_relation(
                        pair,gap["true_ends"],origin))
                scenes.append({
                    "id":gap["id"],"section":int(sid),"kind":world,
                    "proposals":proposals,"source_count":pred["source_count"],
                    "branch_count":pred["hard_branches"],
                    "status":pred["status"],
                    "has_true_proposal":bool(any(x["y"] for x in proposals)),
                    "original_rgb":rgb,"visible_prob":visible,
                    "gray":rgb.mean(axis=2),"rectangle":rect,
                    "origin":origin,"true_ends":gap["true_ends"],
                    "hidden_centerline":gap["hidden_centerline"],
                })
    return scenes,source_eligibility


def _full_train_logistic(train):
    from scripts.tver_v26_predicted_cracks_e2e import _train_relation
    model,receipt=_train_relation(train)
    if model is None:
        raise RuntimeError("STOP Thebe full-data logistic insufficient predicted TRAIN positives")
    return model,receipt


def _section_calibrate(scenes,scores):
    """Calibration and risk on independent source section IDs, not pair rows.

    A section is an error if ANY proposed relation is accepted wrongly in
    any LINK or artificially candidate-absent NO_LINK case of that section.
    Nominal <=10% risk using a 95% CP upper confidence bound assumes
    independent source sections, which is UNVERIFIED within Thebe volume.
    """
    if len(scenes)!=len(scores):
        raise ValueError("Scores must correspond to every actual CAL scene")
    pos={s["section"] for s in scenes if s["kind"]=="LINK"}
    neg={s["section"] for s in scenes if s["kind"]=="NO_LINK"}
    sections=pos&neg
    if len(sections)<RELATION_CAL_MIN_ELIGIBLE_SECTIONS:
        return {
            "status":"STOP_INSUFFICIENT_INDEPENDENT_SOURCE_SECTION_CAL",
            "n_sections":len(sections),
            "min_sections":RELATION_CAL_MIN_ELIGIBLE_SECTIONS,
            "accept_threshold":1.01,
            "no_TEST_opened":True,
        }
    best=(-1,-1.,-1.,-1.);chosen=1.01
    by_threshold={}
    for threshold in CAL_THRESHOLDS:
        decisions=[_decision(s,score,threshold)
                   for s,score in zip(scenes,scores)]
        wrong_source=set();false_source=set();correct=0;accepted=0
        for s,p in zip(scenes,decisions):
            if s["kind"]=="LINK":
                correct+=int(p["correct"])
                accepted+=int(p["accepted"])
                if p["accepted"] and not p["correct"]:
                    wrong_source.add(s["section"])
            elif p["accepted"]:
                false_source.add(s["section"])
        bad_link=len(wrong_source)
        bad_nolink=len(false_source)
        up_link=upper_clopper_pearson(bad_link,len(sections))
        up_nolink=upper_clopper_pearson(bad_nolink,len(sections))
        admitted=(up_link<=MAX_SECTION_RISK and up_nolink<=MAX_SECTION_RISK)
        by_threshold[str(threshold)]={
            "correctly_accepted_positive_gap_cases":correct,
            "total_accepted_positive_gap_cases":accepted,
            "wrong_accepted_LINK_source_sections":bad_link,
            "accepted_candidate_absent_NOLINK_source_sections":bad_nolink,
            "wrong_link_CP_upper_95":up_link,
            "nolink_CP_upper_95":up_nolink,
            "risk_gate_admitted_under_independent_source_assumption":admitted,
        }
        rank=(correct,-bad_link,-bad_nolink,threshold)
        if admitted and rank>best:
            best=rank;chosen=float(threshold)
    selected=by_threshold[str(chosen)]
    return {
        "status":"CAL_SECTION_RISK_CONDITIONALLY_FROZEN",
        "accept_threshold":chosen,
        "source_section_count":len(sections),
        "source_section_ids":sorted(sections),
        "n_positive_gap_cases":sum(s["kind"]=="LINK" for s in scenes),
        "n_fake_candidate_absent_nolink":sum(s["kind"]=="NO_LINK" for s in scenes),
        "CAL_only_section_risk_metrics":selected,
        "scan_by_threshold":by_threshold,
        "assumption":"distinct numeric sections within SAME Thebe field treated conditionally as independent; physical independence not established",
        "all_CI_one_sided_95pct_and_per_section":True,
        "synthetic_NO_LINK_not_geological_identity":True,
        "no_TEST_opened":True,
    }


def _load_model(frontend_dir,frontend,meta):
    item=meta["frontends"][frontend]
    path=Path(frontend_dir)/item["checkpoint"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item["SHA256"]:
        raise RuntimeError("STOP newly trained Thebe frontend checksum mismatch")
    obj=torch.load(path,map_location="cpu",weights_only=False)
    if obj.get("architecture")!=frontend or tuple(obj["widths"])!=PARAM_WIDTHS:
        raise RuntimeError("STOP Thebe segmentation checkpoint does not match architecture")
    model=build_comparable_model(frontend,widths=PARAM_WIDTHS)
    model.load_state_dict(obj["model"])
    model.eval()
    return model


def _save_cal_cases(out,frontend,method,scenes,scores,threshold):
    rows=[]
    for scene,score in zip(scenes,scores):
        dec=_decision(scene,score,threshold)
        rows.append({
            "frontend":frontend,"model":method,
            "section":scene["section"],"artificial_gap_id":scene["id"],
            "world":scene["kind"],
            "predicted_source_port_count":scene["source_count"],
            "predicted_candidate_count":len(scene["proposals"]),
            "at_least_one_true_candidate":scene["has_true_proposal"],
            "top1_correct_BEFORE_risk_gate":dec["top1_correct"],
            "top1_score":dec["top1_score"],
            "correct_accept_AFTER_risk_gate":dec["correct"],
            "accepted_ANY":dec["accepted"],
            "calibration_accept_threshold":threshold,
        })
    return rows


def execute(memmap_root,frontend_dir,out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    original=Path(frontend_dir)/"v33_THEBE_frontends_topology_TRAIN_CAL_frozen.json"
    if not original.is_file():
        raise RuntimeError("STOP Thebe frontends/topology freeze receipt missing")
    meta=json.loads(original.read_text())
    train_x=read_memmap(memmap_root,"train_seismic_float32.npy")
    train_y=read_memmap(memmap_root,"train_fault_bool.npy")
    val_x=read_memmap(memmap_root,"val_seismic_float32.npy")
    val_y=read_memmap(memmap_root,"val_fault_bool.npy")
    report={
        "status":"V33_RELATION_FIT_START",
        "model_and_thresholds_frozen_before_REL_CAL":True,
        "new_Thebe_TEST7_images_labels_NEVER_opened":True,
        "segmentation_CAL_separate_from_relation_CAL":True,
        "train_real_source_sections":list(TRAIN_SOURCE_IDS),
        "relation_CAL_source_sections":list(RELATION_CAL_SOURCE_IDS),
        "topology_SEG_CAL_indices":meta["SEG_CAL_section_indices"],
        "REL_CAL_min_source_sections":RELATION_CAL_MIN_ELIGIBLE_SECTIONS,
        "frontends":{},
    }
    if (meta["status"]!="V33_THEBE_REAL_SEGMENTERS_AND_TOPOLOGY_TRAIN_CAL_COMPLETE"
        or meta["REL_CAL_section_indices_locked_unopened"]!=list(RELATION_CAL_SOURCE_IDS)):
        raise RuntimeError("STOP Thebe front-end topology stage NOT eligible for relation training")
    frozen_thresholds={
        frontend:meta["frontends"][frontend]["topology_threshold_frozen"]
        for frontend in FRONTENDS}
    for frontend in FRONTENDS:
        seg=_load_model(frontend_dir,frontend,meta)
        th=frozen_thresholds[frontend]
        train,train_sources=_candidate_scenes(
            train_x,train_y,TRAIN_SOURCE_IDS,meta["normalization"],seg,th)
        train_pos=sum(x["has_true_proposal"] for x in train if x["kind"]=="LINK")
        if train_pos<8:
            report["frontends"][frontend]={
                "status":"STOP_INSUFFICIENT_TRUE_PREDICTED_TRAIN_PAIRS",
                "correct_candidate_train_cases":train_pos,
                "no_oracle_positive_candidate_injected":True}
            continue
        try:
            images,labels,features,passport=paired_training_examples(train,th)
            full,full_receipt=_full_train_logistic(train)
            matched=fit_matched_logistic(features,labels)
        except (RuntimeError,ValueError) as error:
            report["frontends"][frontend]={
                "status":"STOP_PREDICTED_TRAIN_PAIR_MATCHING_OR_RELATION_DATA",
                "true_predicted_train_cases":train_pos,
                "reason":str(error),
                "no_oracle_pairs_injected":True}
            continue
        models={
            "geometry":None,"full_train_logistic":full,
            "matched_logistic":matched}
        p0_receipts={}
        for seed in SEEDS:
            mdl,proof=fit_native_p0(images,labels,seed=seed,
                                    root=out/"new_thebe_P0",
                                    frontend=frontend)
            models[f"p0_seed{seed}"]=mdl
            p0_receipts[str(seed)]=proof
        # All relation models trained using ONLY TRAIN source before
        # any future relation CAL image or label pixels.
        cal,cal_sources=_candidate_scenes(
            val_x,val_y,RELATION_CAL_SOURCE_IDS,meta["normalization"],seg,th)
        policies={}
        case_rows=[]
        for method in MODEL_NAMES:
            scores=[_scores(s,method,models[method],th) for s in cal]
            admission=_section_calibrate(cal,scores)
            policies[method]=admission
            case_rows.extend(_save_cal_cases(
                out,frontend,method,cal,scores,admission["accept_threshold"]))
        if case_rows:
            with (out/(frontend+"_thebe_REL_CAL_case_rows.csv")).open(
                "w",encoding="utf8",newline="") as f:
                wr=csv.DictWriter(f,fieldnames=list(case_rows[0]))
                wr.writeheader();wr.writerows(case_rows)
        report["frontends"][frontend]={
            "status":"TRAINED_TRUE_PREDICTED_ONLY_NEW_LIRA_P0_AND_LOGISTICS",
            "frozen_segmentation_threshold":th,
            "genuine_original_P0_architecture_newly_trained_not_historical":True,
            "train_source_receipts":train_sources,
            "train_correct_candidate_cases":train_pos,
            "matched_train_source_pair_passport":passport,
            "full_logistic_train_counts":full_receipt,
            "new_5conv_P0_checkpoints":p0_receipts,
            "relation_CAL_source_receipts":cal_sources,
            "relation_CAL_method_gates":policies,
            "no_natural_fault_instance_identity_ground_truth":True,
        }
        print("THEBE_RELATION_CAL",frontend,{
            method:(v["status"],v["accept_threshold"])
            for method,v in policies.items()},flush=True)
    success=all(
        r["status"]=="TRAINED_TRUE_PREDICTED_ONLY_NEW_LIRA_P0_AND_LOGISTICS"
        for r in report["frontends"].values())
    all_eligible=success and all(
        all(g["status"]=="CAL_SECTION_RISK_CONDITIONALLY_FROZEN"
            for g in r["relation_CAL_method_gates"].values())
        for r in report["frontends"].values())
    report["status"]=(
        "V33_THEBE_RELATION_FIT_AND_SECTION_RISK_CAL_COMPLETE_TEST_UNOPENED"
        if all_eligible else
        "V33_STOP_THEBE_PREDICTED_PAIR_OR_SECTION_CAL_INSUFFICIENT")
    report["global_admission"]={
        "source_sections_in_one_THEBE_volume_not_proven_independent":True,
        "synthetic_candidate_absent_negative_not_real_geological_no_link":True,
        "never_link_1_01_remains_allowed_and_counts_as_zero_recovery":True,
        "external_TEST7_not_opened":True,
        "no_claim_natural_false_link_probability_le_10pct":True,
        "no_claim_ANZA_beats_UNET_or_original_historical_weights_recovered":True,
    }
    receipt=out/"v33_THEBE_ALL_RELATION_CAL_frozen.json"
    receipt.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],
                      "models":list(report["frontends"]),
                      "all_TEST_unopened":True},indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mmap-root",required=True,type=Path)
    p.add_argument("--frontends-dir",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    a=p.parse_args()
    execute(a.mmap_root,a.frontends_dir,a.out_dir)
