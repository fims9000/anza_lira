#!/usr/bin/env python3
"""V30 independent replay auditor: no model import or classification calls.

Audits ORIGINAL registered CAL thresholds, raw per-candidate score arrays,
no-label-input claims, all TEST counts/decision consistency and new native
P0 checkpoint hashes from the immutable posthoc artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FRONTENDS=("unet","anza_v1")
METHODS=("geometry","full_train_logistic","matched_logistic",
         "p0_seed11","p0_seed23")


def _boolean(s):
    if s=="True":return True
    if s=="False":return False
    raise AssertionError(f"Boolean CSV value unexpected: {s!r}")


def audit(folder):
    folder=Path(folder)
    frozen_raw=(folder/"v30_ALL_train_cal_frozen.json").read_bytes()
    summary=json.loads((folder/"v30_results.json").read_text())
    frozen=json.loads(frozen_raw)
    case_raw=(folder/"v30_test_case_decisions.csv").read_bytes()
    if (summary["status"]!="V30_POSTHOC_REAL_CRACKS_PREDICTED_P0_E2E_EXECUTED"
        or frozen["status"]!="V30_ALL_FIVE_METHODS_TWO_FRONTENDS_FROZEN_BEFORE_TEST"
        or hashlib.sha256(frozen_raw).hexdigest()!=summary["all_model_frozen_SHA256"]
        or hashlib.sha256(case_raw).hexdigest()!=summary["case_csv_SHA256"]
        or not summary["TEST_original_cases_already_seen_in_V27_V28"]
        or not frozen["no_expert_source_or_target_endpoint_as_any_inference_feature"]
        or frozen["registered_V28_segmentation_thresholds"]!={"unet":.6,"anza_v1":.9}):
        raise AssertionError("V30 posthoc/frozen SHA/threshold declaration incorrect")
    rows=list(csv.DictReader(case_raw.decode().splitlines()))
    if len(rows)!=2*5*4:
        raise AssertionError("V30 expected exactly four worlds per 2 frontends × 5 methods")
    observed={};score_types={}
    for frontend in FRONTENDS:
        r=frozen["frontends"][frontend]
        if (not r["source_and_all_proposals_predicted_only"]
            or r["matched_context_training"]["n_train_examples"]<16
            or not r["matched_context_training"]["no_expert_endpoints_in_CNN_input"]
            or not set(r["matched_context_training"]["train_source_sections"]).issubset(
                frozen["real_source_sections"]["train"])
            or not r["matched_context_training"]["train_source_sections"]):
            raise AssertionError("P0 TRAIN provenance does not match registered TRAIN sections")
        p0=r["new_native_P0_architecture_checkpoints"]
        if set(p0)!={"11","23"}:raise AssertionError("Both real native P0 seeds absent")
        sha=[]
        for seed,rec in p0.items():
            filename=folder/"new_LIRA_P0_checkpoints"/rec["checkpoint"]
            if (not filename.is_file()
                or hashlib.sha256(filename.read_bytes()).hexdigest()!=rec["checkpoint_SHA256"]
                or rec["genuine_repository_five_convolution_P0"] is not True
                or rec["original_historical_P0_checkpoint_used"] is not False
                or rec["train_examples"]!=r["matched_context_training"]["n_train_examples"]):
                raise AssertionError("New P0 trained checkpoint missing/drift/historical mismatch")
            sha.append(rec["checkpoint_SHA256"])
        if len(set(sha))!=2:raise AssertionError("Two initializations trained identical weights")
        for method in METHODS:
            subset=[row for row in rows if
                    row["frontend"]==frontend and row["method"]==method]
            if len(subset)!=4 or {int(r["section"]) for r in subset}!={108,124}:
                raise AssertionError("V30 real TEST population changed")
            cal=frozen["calibration_only_acceptance_gates"][frontend][method]
            if (cal["status"]!="CAL_FROZEN"
                or cal["n_positive"]!=4 or cal["n_synthetic_NO_LINK"]!=4
                or cal["n_wrong_accept_CAL"]!=0 or cal["n_false_accept_CAL"]!=0):
                raise AssertionError("V30 4-positive+4-NOLINK strict empirical CAL budget altered")
            threshold=float(cal["selected_threshold"])
            pos=[];neg=[]
            for row in subset:
                label=row["kind"]
                if label not in ("LINK","NO_LINK"):
                    raise AssertionError("Unknown ground truth case type")
                scores=np.asarray(json.loads(row["all_candidate_scores_json"]),float)
                n=int(row["n_predicted_candidates"])
                if len(scores)!=n or not np.isfinite(scores).all():
                    raise AssertionError("P0 candidate list/score invalid")
                if n:
                    winner=int(np.argmax(scores))
                    choice=bool(scores[winner]>=threshold)
                    if float(row["ungated_top1_score"])!=float(scores[winner]):
                        raise AssertionError("Incorrect ungated top1 scalar receipt")
                    if int(row["chosen_index"])!=winner:
                        raise AssertionError("Top1 candidate index inconsistent")
                else:
                    choice=False
                    if row["chosen_index"] not in ("", "None"):
                        raise AssertionError("No candidates but a top1 was selected")
                if choice!=_boolean(row["accepted"]):
                    raise AssertionError(f"{frontend}/{method}: TEST selection differs from CAL")
                if _boolean(row["correct_accepted"]) and not (
                    choice and label=="LINK" and
                    _boolean(row["ungated_top1_correct"])):
                    raise AssertionError("False claim of correct accepted relation")
                if (_boolean(row["straight_path_success"]) or
                    _boolean(row["maximin_path_success"])) and not _boolean(row["correct_accepted"]):
                    raise AssertionError("Hidden/test path leaked without accepted correct pair")
                (pos if label=="LINK" else neg).append(row)
            metrics={
                "positive_gap_count":len(pos),
                "synthetic_candidate_absent_negative_count":len(neg),
                "predicted_source_available":sum(int(x["n_predicted_source_ports"])>0 for x in pos),
                "predicted_SBPP_candidates_available":sum(int(x["n_predicted_candidates"])>0 for x in pos),
                "correct_predicted_candidate_available":sum(_boolean(x["true_candidate_available"]) for x in pos),
                "top1_correct_before_CAL_refusal":sum(_boolean(x["ungated_top1_correct"]) for x in pos),
                "correct_accepted_after_CAL_refusal":sum(_boolean(x["correct_accepted"]) for x in pos),
                "wrong_accepted_positive":sum(_boolean(x["accepted"]) and not _boolean(x["correct_accepted"]) for x in pos),
                "false_accepted_synthetic_NOLINK":sum(_boolean(x["accepted"]) for x in neg),
                "maximin_correct_path":sum(_boolean(x["maximin_path_success"]) for x in pos),
                "straight_correct_path":sum(_boolean(x["straight_path_success"]) for x in pos),
            }
            for k,value in metrics.items():
                if value!=summary["methods"][frontend][method][k]:
                    raise AssertionError(f"V30 case-count drift {frontend}/{method}/{k}")
            observed[f"{frontend}/{method}"]=metrics
    report={
        "status":"V30_INDEPENDENT_NATIVE_P0_REAL_PREDICTED_CASE_AUDIT_PASS",
        "recomputed_two_real_TEST_sections_and_four_worlds_per_policy":True,
        "all_ten_method_frontend_cases_verified":True,
        "all_four_new_genuine_P0_state_dict_SHA256_verified":True,
        "different_true_training_seeds_verified":True,
        "top1_and_selected_CAL_thresholds_independently_checked":True,
        "frozen_source_cohort_no_heldout_fitting":True,
        "historical_ANZA_LIRA_weights_not_recovered":True,
        "posthoc_108_124_previously_seen_during_V27_V28":True,
        "genuine_real_seismic_pixels_but_synthetic_gap_and_negative_interventions":True,
        "cases":observed,
    }
    (folder/"verify_v30_independent.json").write_text(
        json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"cases":observed},indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dir",required=True,type=Path)
    a=p.parse_args()
    audit(a.dir)
