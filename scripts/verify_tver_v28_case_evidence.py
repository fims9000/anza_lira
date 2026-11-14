#!/usr/bin/env python3
"""Independent source/calibration/case audit of executed V28 real predicted E2E.

Reads saved TRAIN/CAL topology receipts and heldout per-case predictions,
not segmentation weights, raw seismic TEST pixels, or original expert masks.
V28's TEST 108/124 was already inspected in V27; this is posthoc evidence,
not a new confirmatory validation of ANZA superiority.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def _bool(x):
    if isinstance(x,bool):return x
    if x in ("True","true","1",1):return True
    if x in ("False","false","0",0):return False
    raise AssertionError(f"Unknown boolean indicator {x}")


def _sha(x):
    return hashlib.sha256(Path(x).read_bytes()).hexdigest()


def verify(root):
    root=Path(root)
    files={
        "summary":root/"v28_summary.json",
        "topology":root/"v28_topology_TRAIN_CAL_frozen.json",
        "cohort":root/"v28_source_cohort.json",
        "segment":root/"new_train_model/v26_segmenter_frozen.json",
        "allmodels":root/"predicted_e2e/v26_all_model_CAL_frozen.json",
        "scores":root/"predicted_e2e/v26_real_predicted_case_decisions.csv",
        "report":root/"predicted_e2e/v26_real_e2e_results.json",
    }
    all_files={k:json.loads(p.read_text()) for k,p in files.items()
               if k!="scores"}
    s=all_files["summary"];t=all_files["topology"]
    c=all_files["cohort"];seg=all_files["segment"]
    locked=all_files["allmodels"];out=all_files["report"]
    if (s["status"]!="V28_POSTHOC_TOPOLOGY_ADAPTED_REAL_PREDICTED_E2E_EXECUTED"
        or not s["original_V27_TEST_inspected_before_design"]
        or not t["posthoc_after_V27_TEST_inspection"]
        or not t["V28_TEST_images_labels_not_decoded_for_selection"]
        or not s["same_BLUE_selected_source_cohort_not_external"]
        or not s["model_weight_SHA256_same_as_V27"]
        or s["historical_weights_UNAVAILABLE"] is not True
        or not s["no_independent_SOTA_claim"]):
        raise AssertionError("Misstated posthoc V28 scientific claims")
    for file_name,json_name in (
        ("topology","topology_threshold_freeze_SHA256"),
        ("cohort","source_cohort_SHA256"),
        ("allmodels","all_model_CAL_freeze_sha256"),
    ):
        if _sha(files[file_name])!=s[json_name]:
            raise AssertionError("Immutable V28 file SHA mismatch "+file_name)
    if (_sha(files["allmodels"])!=out["frozen_SHA256"]
        or _sha(files["segment"])!=locked["segmentation_frozen_receipt_sha256"]
        or _sha(files["scores"])!=out["case_csv_SHA256"]):
        raise AssertionError("V28 model score and CAL replay SHA changed")
    train=[10,13,17,22,24,35,41]
    cal=[43,67]
    held=[108,124]
    if (c["TRAIN_sections"]!=train or c["CAL_sections"]!=cal
        or c["BUFFER_sections"]!=[106]
        or c["TEST_sections"]!=held
        or t["TRAIN_section_ids"]!=train or t["CAL_section_ids"]!=cal
        or t["reserved_TEST_section_ids"]!=held
        or locked["real_TR_sections"]!=train
        or locked["real_CAL_sections"]!=cal
        or locked["reserved_unseen_TEST_sections"]!=held):
        raise AssertionError("Real F3 fault-containing cohorts changed")
    # Re-derive selected topology thresholds independently, without
    # importing/running source candidate scoring or looking at TEST.
    verified={}
    for name in ("anza_v1","unet"):
        source=t["frontends"][name]
        feasible=[]
        for q in t["allowed_thresholds"]:
            tr=source["TRAIN_metrics_by_threshold"][str(q)]
            va=source["CAL_metrics_by_threshold"][str(q)]
            if (tr["n_true_pairs"]>=t["TRAIN_true_pair_min"]
                and tr["n_false_pairs"]>=t["TRAIN_false_pair_min"]
                and va["n_true_pair_gaps"]>=t["CAL_positive_gap_with_true_candidate_min"]):
                value=(va["n_true_pair_gaps"],-va["n_candidates"],
                       va["n_sources"],-va["n_dense_rejected"],q)
                feasible.append((value,float(q)))
        selected=max(feasible)[1] if feasible else None
        if (selected!=source["selected_threshold"]
            or s["thresholds_new"][name]!=selected
            or seg["frontends"][name]["selected_CAL_threshold"]!=selected
            or seg["frontends"][name]["checkpoint_SHA256"]!=
               locked["frontends"][name]["segmentation_checkpoint_SHA256"]):
            raise AssertionError(f"V28 {name} threshold chosen from heldout TEST or weight changed")
        status=locked["frontends"][name]["real_TRAIN_pairs"]
        expected_new_head=status["status"]=="TRAINED_PREDICTED_PAIR_RELATION"
        if expected_new_head:
            roothead=root/"predicted_e2e"/status["checkpoint_filename"]
            if not roothead.exists() or _sha(roothead)!=status["checkpoint_SHA256"]:
                raise AssertionError("Trained LIRA state not present/frozen")
        verified[name]={"selected_CAL_topology_threshold":selected,
                        "predicted_TRAIN_true_pairs":status["counts"]["1"],
                        "predicted_TRAIN_false_pairs":status["counts"]["0"],
                        "LIRA_fitted":expected_new_head,
                        "CAL_geometry":locked["frontends"][name]["CAL_thresholds"]["geometry"],
                        "CAL_new_LIRA":locked["frontends"][name]["CAL_thresholds"]["new_lira"]}
    with files["scores"].open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    expected={
        (n,m) for n in ("anza_v1","unet")
        for m in ("geometry","new_lira")}
    grouped=defaultdict(list)
    for row in rows:
        if (int(row["section"]) not in held or
            int(row["n_predictions_leaking_expert_endpoints"])!=0):
            raise AssertionError("Real TEST/source input leaked expert source geometry")
        if _bool(row["correct"]) and (
            not _bool(row["accepted"]) or row["kind"]!="LINK"):
            raise AssertionError("Invalid accepted correct-pair identity")
        if _bool(row["maximin_path_tol2_success"]) and not _bool(row["correct"]):
            raise AssertionError("Incorrect path counted despite wrong pair")
        grouped[row["frontend"],row["method"]].append(row)
    if set(grouped)!=expected:
        raise AssertionError("No actual AZConv/U-Net and LIRA/geometry decisions")
    for (name,method),batch in grouped.items():
        pos=[r for r in batch if r["kind"]=="LINK"]
        neg=[r for r in batch if r["kind"]=="NO_LINK"]
        if (len(pos)!=len(neg) or len(pos)!=2 or
            {int(x["section"]) for x in pos}!={108,124}):
            raise AssertionError("Wrong actual controlled test scene set")
        metrics={
            "n_source_available":sum(int(x["n_source_ports"])>0 for x in pos),
            "n_candidate_available":sum(int(x["n_sbpp_proposals"])>0 for x in pos),
            "n_true_candidate_available":sum(_bool(x["has_correct_candidate"]) for x in pos),
            "n_correct_accepted":sum(_bool(x["correct"]) for x in pos),
            "n_wrong_accepted":sum(_bool(x["accepted"]) and not _bool(x["correct"]) for x in pos),
            "n_false_accepted_synthetic_NO_LINK":sum(_bool(x["accepted"]) for x in neg),
            "n_correctly_completed_straight_path":sum(_bool(x["path_tol2_success"]) for x in pos),
            "n_correctly_completed_maximin_path":sum(_bool(x["maximin_path_tol2_success"]) for x in pos),
        }
        expected_metrics=out["models"][name]["methods"][method]
        for k,value in metrics.items():
            if value!=expected_metrics[k]:
                raise AssertionError(f"V28 false test model-quality claim {name}/{method}/{k}")
        verified[name][method+"_heldout_metrics"]=metrics
    record={
        "status":"V28_INDEPENDENT_POSTHOC_TOPOLOGY_REAL_E2E_CASE_AUDIT_PASS",
        "original_BLUE_TEST_was_exposed_in_V27":True,
        "V28_topology_selected_only_from_TRAIN_CAL_rows":True,
        "both_new_LIRA_heads_trained_from_predicted_ports":all(
            verified[name]["LIRA_fitted"] for name in verified),
        "F3_source_sections_heldout_from_V28_training":[108,124],
        "both_frontends_source_and_correct_candidates_available_on_TEST":all(
            verified[n]["new_lira_heldout_metrics"]["n_true_candidate_available"]==2
            for n in verified),
        "individual_REAL_F3_link_path_cases_not_invented":True,
        "no_geological_instance_truth_for_negative_controls":True,
        "not_original_historical_model":True,
        "models":verified,
    }
    output=root/"v28_independent_audit.json"
    output.write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "status":record["status"],
        "both_trained_LIRA":record["both_new_LIRA_heads_trained_from_predicted_ports"],
        "both_have_correct_test_candidates":
            record["both_frontends_source_and_correct_candidates_available_on_TEST"],
        "actual_accepted_correct":{
            n:record["models"][n]["new_lira_heldout_metrics"]["n_correct_accepted"]
            for n in verified}
    },indent=2))
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    a=p.parse_args()
    verify(a.root)
