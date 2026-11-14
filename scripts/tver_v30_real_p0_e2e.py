"""Tver V30: real CRACKS ANZA→SBPP→new native P0 LIRA test, posthoc.

This runner is NOT historical weight reproduction. Section 108/124 TEST
outcomes were inspected in V27/V28, so V30 remains exploratory. Fresh
models and ALL CAL decisions freeze before *this run* first opens TEST.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
import shutil
from pathlib import Path

import numpy as np
import torch

from scripts.tver_v27_blue_cohort import (
    blue_available_sections,frozen_positive_split)
from scripts.tver_v26_train_real_segmenters import restore_model
from scripts.tver_v26_predicted_cracks_e2e import (
    _dataset,_train_relation,_rank,_path_success)
from scripts.cracks_real_oracle_gap_audit import sections
from scripts.tver_v30_p0_context import (
    SEEDS,paired_training_examples,fit_native_p0,
    context_scores,fit_matched_logistic,score_logistic)

FRONTENDS=("unet","anza_v1")
REGISTERED_SEG_THRESHOLD={"unet":.60,"anza_v1":.90}
REGISTERED_WEIGHT_SHA256={
    "unet":"908861df22265db189b4d18f53ca587634673b3c39a96c3f3752458e8516dc4b",
    "anza_v1":"14ecda83fbf327085390f1cc5665d30730e250fc4b2192f68ac1d25fa751e300",
}
CAL_THRESHOLDS=tuple(round(float(x),2) for x in
                     np.linspace(0.,1.,21))+(1.01,)
MODEL_NAMES=("geometry","full_train_logistic","matched_logistic",
             "p0_seed11","p0_seed23")


def _scores(scene,kind,model,seg_threshold):
    if kind=="geometry":
        _,v=_rank(scene,"geometry",None)
        if not scene["proposals"]:return np.empty(0)
        # Exact original V28 comparator per-pair geometry score.
        return np.asarray([
            1./(1.+np.exp(-np.clip(float(p["geometry"])/18.,-40,40)))
            for p in scene["proposals"]],float)
    if kind in ("full_train_logistic","matched_logistic"):
        return (score_logistic(model,scene) if model is not None
                else np.empty(0,float))
    if kind.startswith("p0_seed"):
        return context_scores(model,scene,seg_threshold)
    raise ValueError("Not a registered V30 method")


def _decision(scene,score,threshold):
    if len(score)==0:
        return {"accepted":False,"correct":False,
                "chosen":None,"top1_correct":False,
                "top1_score":None,"n_proposals":0}
    k=int(np.argmax(score))
    prob=float(score[k])
    accepted=bool(prob>=threshold)
    correct=bool(accepted and scene["kind"]=="LINK"
                 and scene["proposals"][k]["y"]==1)
    return {"accepted":accepted,"correct":correct,"chosen":k,
            "top1_correct":bool(scene["kind"]=="LINK" and
                                 scene["proposals"][k]["y"]==1),
            "top1_score":prob,"n_proposals":len(score)}


def _calibration(scenes,scores):
    assert len(scores)==len(scenes)
    positives=sum(s["kind"]=="LINK" for s in scenes)
    negatives=sum(s["kind"]=="NO_LINK" for s in scenes)
    if positives<2 or negatives<2:
        return {"status":"STOP_INSUFFICIENT_CAL_LINK_NOLINK",
                "positive":positives,"negative":negatives}
    best=None;choice=None
    for t in CAL_THRESHOLDS:
        pred=[_decision(s,z,t) for s,z in zip(scenes,scores)]
        true=sum(s["kind"]=="LINK" and p["correct"]
                 for s,p in zip(scenes,pred))
        wrong=sum(s["kind"]=="LINK" and p["accepted"] and not p["correct"]
                  for s,p in zip(scenes,pred))
        false=sum(s["kind"]=="NO_LINK" and p["accepted"]
                  for s,p in zip(scenes,pred))
        # Old V27/V28 10% gate means exactly zero among 4 CAL cases.
        if wrong/positives>.10 or false/negatives>.10:continue
        criterion=(true,-wrong,-false,t)
        if best is None or criterion>best:
            best=criterion;choice=t
    if best is None:raise AssertionError("Never-link must always be CAL-feasible")
    assert choice is not None
    return {
        "status":"CAL_FROZEN",
        "selected_threshold":float(choice),
        "n_positive":int(positives),"n_synthetic_NO_LINK":int(negatives),
        "n_correct_accept_CAL":int(best[0]),
        "n_wrong_accept_CAL":int(-best[1]),
        "n_false_accept_CAL":int(-best[2]),
        "CAL_only_frozen_before_reopening_OLD_TEST":True,
        "no_natural_geological_NO_LINK_labels":True,
    }


def _method_metrics(scenes,scored,threshold):
    records=[]
    for scene,scores in zip(scenes,scored):
        result=_decision(scene,scores,threshold)
        maximin_ok=False;straight_ok=False
        if result["correct"]:
            winner=scene["proposals"][result["chosen"]]
            maximin_ok=_path_success(scene,winner,method="maximin")
            straight_ok=_path_success(scene,winner,method="straight")
        records.append({
            "section":int(scene["section"]), "gap_id":scene["id"],
            "kind":scene["kind"],"SBPP_status":scene["status"],
            "n_predicted_source_ports":int(scene["source_count"]),
            "n_predicted_hard_branches":int(scene["branch_count"]),
            "n_predicted_candidates":int(result["n_proposals"]),
            "true_candidate_available":bool(scene["has_true_proposal"]),
            "ungated_top1_correct":bool(result["top1_correct"]),
            "ungated_top1_score":result["top1_score"],
            "accepted":bool(result["accepted"]),
            "correct_accepted":bool(result["correct"]),
            "chosen_index":result["chosen"],
            "maximin_path_success":bool(maximin_ok),
            "straight_path_success":bool(straight_ok),
            "all_candidate_scores_json":json.dumps(list(map(float,scores))),
            "all_candidate_TRAIN_or_TEST_truth_NOT_features":True,
        })
    positives=[r for r in records if r["kind"]=="LINK"]
    negatives=[r for r in records if r["kind"]=="NO_LINK"]
    return {
        "real_TEST_section_count":len(set(r["section"] for r in positives)),
        "positive_gap_count":len(positives),
        "synthetic_candidate_absent_negative_count":len(negatives),
        "predicted_source_available":sum(r["n_predicted_source_ports"]>0
                                         for r in positives),
        "predicted_SBPP_candidates_available":sum(
            r["n_predicted_candidates"]>0 for r in positives),
        "correct_predicted_candidate_available":sum(
            r["true_candidate_available"] for r in positives),
        "top1_correct_before_CAL_refusal":sum(
            r["ungated_top1_correct"] for r in positives),
        "correct_accepted_after_CAL_refusal":sum(
            r["correct_accepted"] for r in positives),
        "wrong_accepted_positive":sum(
            r["accepted"] and not r["correct_accepted"] for r in positives),
        "false_accepted_synthetic_NOLINK":sum(
            r["accepted"] for r in negatives),
        "maximin_correct_path":sum(
            r["maximin_path_success"] for r in positives),
        "straight_correct_path":sum(
            r["straight_path_success"] for r in positives),
        "independent_TEST_source_sections_already_inspected_in_V27_V28":True,
        "not_historical_LIRA_pretrained":True,
    },records


def execute(images,annotations,out,*,v28_reference):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    mapping,cohort=frozen_positive_split(
        blue_available_sections(images,annotations))
    if (cohort["status"]!="V27_FROZEN_BLUE_AVAILABILITY_SOURCE_SPLIT"
        or cohort["TEST_sections"]!=[108,124]
        or cohort["CAL_sections"]!=[43,67]):
        raise RuntimeError("STOP V30 cohort differs from already inspected V27")
    (out/"v30_source_cohort.json").write_text(
        json.dumps(cohort,indent=2)+"\n")
    # **Original V28 model weights**, not an approximation from re-training.
    # An otherwise identical re-training using PyTorch CPU single-thread
    # produces different segmentation weights and would require a NEW CAL
    # topology threshold, violating the V30 controlled P0-only protocol.
    reference=Path(v28_reference)
    original_model_root=reference/"new_train_model"
    original_receipt=original_model_root/"v26_segmenter_frozen.json"
    original_topology=reference/"v28_topology_TRAIN_CAL_frozen.json"
    if not original_receipt.is_file() or not original_topology.is_file():
        raise RuntimeError("STOP original successful V28 model artifacts unavailable")
    fitted=json.loads(original_receipt.read_text())
    topo=json.loads(original_topology.read_text())
    if (fitted["status"]!="V26_NEW_SEGMENTERS_TRAINED_ON_REAL_CRACKS"
        or topo["status"]!="V28_POSTHOC_TRAIN_CAL_ONLY_TOPOLOGY_SELECTION"
        or fitted["TRAIN_section_ids"]!=[10,13,17,22,24,35,41]
        or fitted["CAL_section_ids"]!=[43,67]
        or fitted["reserved_TEST_source_sections"]!=[108,124]
        or topo["reserved_TEST_section_ids"]!=[108,124]):
        raise RuntimeError("STOP V28 source/weights/cohort receipt mismatch")
    thresholds={n:topo["frontends"][n]["selected_threshold"]
                for n in FRONTENDS}
    if thresholds!=REGISTERED_SEG_THRESHOLD:
        raise RuntimeError(
            f"STOP registered V28 threshold drift: {thresholds!r}")
    trained=out/"newly_fitted_AZConv_UNet"
    trained.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(original_receipt,trained/"v26_segmenter_frozen.json")
    shutil.copyfile(original_topology,out/"v30_original_V28_topology_CAL.json")
    for name in FRONTENDS:
        item=fitted["frontends"][name]
        checkpoint=original_model_root/item["checkpoint"]
        if (not checkpoint.is_file()
            or hashlib.sha256(checkpoint.read_bytes()).hexdigest()
               !=item["checkpoint_SHA256"]
            or item["checkpoint_SHA256"]!=REGISTERED_WEIGHT_SHA256[name]
            or topo["frontends"][name]["unchanged_actual_model_SHA256"]
               !=REGISTERED_WEIGHT_SHA256[name]):
            raise RuntimeError("STOP original V28 trained weights SHA256 drift")
        shutil.copyfile(checkpoint,trained/item["checkpoint"])
    # No V30 retraining of segmentation or re-selection of topology.
    # V30 changes only newly fitted pair scoring; never claims a fresh TEST.
    test_ids=[s for s in mapping if mapping[s]=="TEST"]
    train_ids=[s for s in mapping if mapping[s]=="TRAIN"]
    cal_ids=[s for s in mapping if mapping[s]=="CAL"]
    trained_receipts={};cal_receipts={};models={};test_scores={}
    with zipfile.ZipFile(images) as zi,zipfile.ZipFile(annotations) as za:
        names_i=sections(zi);names_a=sections(za,expert=True)
        if not set(test_ids).issubset(set(names_i)&set(names_a)):
            raise RuntimeError("Missing registered original TEST sections")
        for frontend in FRONTENDS:
            seg=restore_model(trained,frontend,fitted)
            config={"train_normalization":fitted["train_normalization"],
                    "selected_threshold":thresholds[frontend]}
            train=_dataset(zi,za,seg,train_ids,config)
            cal=_dataset(zi,za,seg,cal_ids,config)
            imgs,labels,scalar,matched=paired_training_examples(
                train,thresholds[frontend])
            matched_logistic=fit_matched_logistic(scalar,labels)
            full_logistic,full_receipt=_train_relation(train)
            if full_logistic is None:
                raise RuntimeError("STOP full-training logistic missing positive pairs")
            instances={
                "geometry":None,
                "full_train_logistic":full_logistic,
                "matched_logistic":matched_logistic,
            }
            p0_receipts={}
            for seed in SEEDS:
                mdl,passport=fit_native_p0(
                    imgs,labels,seed=seed,
                    root=out/"new_LIRA_P0_checkpoints",
                    frontend=frontend)
                instances[f"p0_seed{seed}"]=mdl
                p0_receipts[str(seed)]=passport
            if (p0_receipts["11"]["checkpoint_SHA256"]==
                p0_receipts["23"]["checkpoint_SHA256"]):
                raise RuntimeError("STOP not genuinely distinct seeded P0 heads")
            gates={}
            for method in MODEL_NAMES:
                scores=[_scores(s,method,instances[method],
                                thresholds[frontend]) for s in cal]
                gates[method]=_calibration(cal,scores)
            trained_receipts[frontend]={
                "segmentation_checkpoint_SHA256":fitted["frontends"][
                    frontend]["checkpoint_SHA256"],
                "segmentation_threshold_V28_unchanged":thresholds[frontend],
                "matched_context_training":matched,
                "full_logistic_training":full_receipt,
                "new_native_P0_architecture_checkpoints":p0_receipts,
                "matched_vs_full_feature_logistic_train_budget_different":True,
                "source_and_all_proposals_predicted_only":True,
            }
            cal_receipts[frontend]=gates
            models[frontend]=(seg,instances)
            # Hold calibrated objects only; no TEST mask data loaded.
            del train,cal,imgs,labels,scalar
        frozen={
            "status":"V30_ALL_FIVE_METHODS_TWO_FRONTENDS_FROZEN_BEFORE_TEST",
            "original_historical_ANZA_LIRA_weights_not_available":True,
            "V30_IS_POSTHOC_after_V27_V28_TEST_inspection":True,
            "V30_TEST_not_decoded_in_this_run_before_freeze":True,
            "real_source_sections":{
                "train":sorted(train_ids),"cal":sorted(cal_ids),
                "buffer":cohort["BUFFER_sections"],
                "test_already_inspected_prior":sorted(test_ids)},
            "registered_V28_segmentation_thresholds":thresholds,
            "exact_persisted_V28_segmentation_weight_SHA256":REGISTERED_WEIGHT_SHA256,
            "never_retrained_or_rethresholded_original_V28_frontends":True,
            "frontends":trained_receipts,
            "calibration_only_acceptance_gates":cal_receipts,
            "no_expert_source_or_target_endpoint_as_any_inference_feature":True,
            "synthetic_NO_LINK_not_natural_fault_identity":True,
        }
        receipt=out/"v30_ALL_train_cal_frozen.json"
        receipt.write_text(json.dumps(frozen,indent=2)+"\n")
        locksha=hashlib.sha256(receipt.read_bytes()).hexdigest()
        results={};all_cases=[]
        for frontend in FRONTENDS:
            seg,instances=models[frontend]
            test=_dataset(zi,za,seg,test_ids,{
                "train_normalization":fitted["train_normalization"],
                "selected_threshold":thresholds[frontend]})
            by_method={}
            for method in MODEL_NAMES:
                scores=[_scores(s,method,instances[method],
                                thresholds[frontend]) for s in test]
                gate=cal_receipts[frontend][method]
                cutoff=(gate["selected_threshold"] if
                        gate["status"]=="CAL_FROZEN" else 1.01)
                metrics,cases=_method_metrics(test,scores,cutoff)
                by_method[method]=metrics
                all_cases.extend({"frontend":frontend,"method":method,**r}
                                 for r in cases)
            results[frontend]=by_method
    if not all_cases:
        raise RuntimeError("STOP no genuine source TEST cases")
    path=out/"v30_test_case_decisions.csv"
    with path.open("w",encoding="utf8",newline="") as f:
        fields=list(all_cases[0])
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(all_cases)
    report={
        "status":"V30_POSTHOC_REAL_CRACKS_PREDICTED_P0_E2E_EXECUTED",
        "study_is_new_P0_ARCHITECTURE_not_historical_pretrained_LIRA":True,
        "real_seismic_image_pixels_and_predicted_SBPP_ports":True,
        "TEST_original_cases_already_seen_in_V27_V28":True,
        "real_TEST_sections":sorted(test_ids),
        "all_model_frozen_SHA256":locksha,
        "case_csv_SHA256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "trained_CNN_2_seeds_each_frontend":True,
        "thresholds_predeclared_and_CAL_only":True,
        "methods":results,
        "not_natural_fault_gap_detection":True,
        "not_SOTA_against_published_seismic_models":True,
    }
    (out/"v30_results.json").write_text(
        json.dumps(report,indent=2)+"\n")
    print(json.dumps({
        "status":report["status"],"results":results,
        "lock_SHA":locksha,
    },indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--images",required=True,type=Path)
    p.add_argument("--annotations",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--v28-reference",required=True,type=Path)
    a=p.parse_args()
    execute(a.images,a.annotations,a.out,v28_reference=a.v28_reference)
