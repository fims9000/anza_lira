"""Independent V27 real CRACKS predicted source-port metrics and lock audit.

Does not load/train model, open official expert masks, propose source ports
or call LIRA. Recomputes every test count from the frozen case-level output.
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
    raise AssertionError("Nonboolean recorded frozen predicted decision")


def verify(root):
    root=Path(root)
    eligibility=(root/"v27_eligibility_frozen.json")
    frozen=(root/"predicted_e2e/v26_all_model_CAL_frozen.json")
    result=(root/"predicted_e2e/v26_real_e2e_results.json")
    report=root/"v27_summary.json"
    cases=(root/"predicted_e2e/v26_real_predicted_case_decisions.csv")
    receipt=json.loads(report.read_text())
    r=json.loads(result.read_text())
    f=json.loads(frozen.read_text())
    s=json.loads(eligibility.read_text())
    if (receipt["status"]!="V27_BLUE_ELIGIBLE_REAL_CRACKS_COMPLETE_EXPLORATORY"
        or s["status"]!="V27_FROZEN_BLUE_AVAILABILITY_SOURCE_SPLIT"
        or not receipt["source_selection_is_post_V26_label_availability"]
        or not r["real_TEST_image_accessed_after_ALL_fitted_models_frozen"]
        or not f["no_TEST_image_mask_decoded_for_supervised_learning"]
        or receipt["old_original_checkpoint_not_recovered"] is not True
        or not r["no_claim_ANZA_SOTA"]):
        raise AssertionError("V27 source cohort, heldout or historical model disclosure")
    cohort=s["cohort"]
    if (cohort["n_eligible"]<10
        or len(cohort["CAL_sections"])!=2
        or len(cohort["TEST_sections"])!=2
        or cohort["TEST_min_minus_CAL_max_F3_sections"]<3
        or receipt["test_section_ids"]!=cohort["TEST_sections"]
        or receipt["buffer_section_ids"]!=cohort["BUFFER_sections"]):
        raise AssertionError("The registered BLUE eligible cohort changed")
    if (hashlib.sha256(eligibility.read_bytes()).hexdigest()
            !=receipt["frozen_eligibility_sha256"]
        or hashlib.sha256(result.read_bytes()).hexdigest()
            !=receipt["e2e_score_sha256"]
        or hashlib.sha256(frozen.read_bytes()).hexdigest()
            !=r["frozen_SHA256"]
        or hashlib.sha256(cases.read_bytes()).hexdigest()
            !=r["case_csv_SHA256"]):
        raise AssertionError("V27 all frozen artifacts / per-case decisions not SHA locked")
    with cases.open(newline="",encoding="utf8") as stream:
        rows=list(csv.DictReader(stream))
    if not rows:
        raise AssertionError("No real TEST predicted-port cases in research artifact")
    expected=set(cohort["TEST_sections"])
    for row in rows:
        if int(row["section"]) not in expected:
            raise AssertionError("TRAIN/CAL section leaked into final heldout metrics")
        if int(row["n_predictions_leaking_expert_endpoints"])!=0:
            raise AssertionError("SBPP used expert oracle source/destination coordinates")
        if _bool(row["correct"]) and (
            not _bool(row["accepted"]) or row["kind"]!="LINK"):
            raise AssertionError("Incorrect verified success label")
        if _bool(row["maximin_path_tol2_success"]) and not _bool(row["correct"]):
            raise AssertionError("A path was marked success after wrong relation")
    grouped=defaultdict(list)
    for row in rows:grouped[row["frontend"],row["method"]].append(row)
    if set(grouped)!={
        ("unet","geometry"),("unet","new_lira"),
        ("anza_v1","geometry"),("anza_v1","new_lira")}:
        raise AssertionError("Real AZConv or U-Net, SBPP or LIRA baseline missing")
    audited={}
    for (name,method),subset in grouped.items():
        positives=[a for a in subset if a["kind"]=="LINK"]
        negatives=[a for a in subset if a["kind"]=="NO_LINK"]
        if not positives or len(positives)!=len(negatives):
            raise AssertionError("V27 synthetic candidate-absent negative pairing changed")
        if len({a["section"] for a in positives})!=2:
            raise AssertionError("Both physical TEST expert source sections must be represented")
        stats={
            "n_positive_gaps":len(positives),
            "n_synthetic_candidate_absent_NO_LINK":len(negatives),
            "n_source_available":sum(int(a["n_source_ports"])>0 for a in positives),
            "n_candidate_available":sum(int(a["n_sbpp_proposals"])>0 for a in positives),
            "n_true_candidate_available":sum(_bool(a["has_correct_candidate"]) for a in positives),
            "n_correct_accepted":sum(_bool(a["correct"]) for a in positives),
            "n_wrong_accepted":sum(_bool(a["accepted"]) and not _bool(a["correct"]) for a in positives),
            "n_false_accepted_synthetic_NO_LINK":sum(_bool(a["accepted"]) for a in negatives),
            "n_correctly_completed_straight_path":sum(_bool(a["path_tol2_success"]) for a in positives),
            "n_correctly_completed_maximin_path":sum(_bool(a["maximin_path_tol2_success"]) for a in positives),
            "n_real_TEST_source_sections":len({a["section"] for a in positives}),
        }
        claimed=r["models"][name]["methods"][method]
        for key,value in stats.items():
            if claimed[key]!=value:
                raise AssertionError(f"V27 inflated E2E count: {name}/{method}/{key}")
        audited[name+"/"+method]=stats
    record={
        "status":"V27_INDEPENDENT_REAL_PREDICTED_E2E_COUNT_AUDIT_PASS",
        "trained_anza_and_unet_fresh_real_images":True,
        "all_sha256_verified":True,
        "n_unique_source_positive_TEST_sections":2,
        "test_sections":[int(x) for x in cohort["TEST_sections"]],
        "test_selection_was_conditional_on_BLUE_ground_truth_availability":True,
        "no_independent_external_f3_volume":True,
        "oracle_endpoints_not_used_to_select_ports_or_pairs":True,
        "ground_truth_used_to_define_missing_segmentation_intervention":True,
        "candidate_absent_NEGATIVE_is_synthetic_not_natural_geology":True,
        "missing_true_LIRA_train_examples_means_STOP_not_fake_model":True,
        "historical_weights_UNAVAILABLE":True,
        "all_method_denominators_and_accepted_paths_recounted":True,
        "methods":audited,
    }
    (root/"verify_v27_independent.json").write_text(
        json.dumps(record,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(record,indent=2,ensure_ascii=False))
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True,type=Path)
    args=p.parse_args()
    verify(args.root)
