"""V27 full freshly trained real CRACKS AZConv→SBPP→LIRA controlled pilot.

The expert BLUE path *availability* determines the cohort in a transparent
post-V26 source diagnostic. This is NOT a confirmatory natural-data split.
No historical checkpoints; train new ANZA-v1 and same-budget U-Net, then
fit one new LIRA relation head on predicted SBPP proposals, freeze CAL,
and only then generate TEST predictions. No expert source/target oracle
is passed to model inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.tver_v27_blue_cohort import (
    blue_available_sections,frozen_positive_split,
)
from scripts.tver_v26_train_real_segmenters import execute as fit_segmenters
from scripts.tver_v26_predicted_cracks_e2e import execute as run_e2e


def execute(images,annotations,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=blue_available_sections(images,annotations)
    mapping,eligibility=frozen_positive_split(rows)
    source={
        "status":eligibility["status"],
        "cohort":eligibility,
        "all_original_expert_section_BLUE_availability":rows,
        "disclosure":"BLUE-eligible spatial TEST selected AFTER discovering empty V26 TEST, before any V27 model TEST performance",
        "no_TEST_labels_as_inference_features":True,
        "original_checkpoint_absent":True,
    }
    source_path=out/"v27_eligibility_frozen.json"
    source_path.write_text(json.dumps(source,indent=2,ensure_ascii=False)+"\n")
    if source["status"]!="V27_FROZEN_BLUE_AVAILABILITY_SOURCE_SPLIT":
        print(json.dumps({
            "status":"V27_STOP_SOURCE_LABEL_ELIGIBILITY",
            "reason":source["status"],
            "trained_historical_ANZA_LIRA":False},indent=2))
        return {"status":"V27_STOP_SOURCE_LABEL_ELIGIBILITY"}
    trained=out/"trained_AZConv_UNet"
    train_receipt=fit_segmenters(
        images,annotations,trained,split_mapping=mapping)
    if not train_receipt["V27_source_eligible_BLUE_cohort"]:
        raise RuntimeError("V27 train did not bind exact source eligibility")
    e2e_dir=out/"predicted_e2e"
    result=run_e2e(images,annotations,trained,e2e_dir,
                   split_mapping=mapping)
    report={
        "status":"V27_BLUE_ELIGIBLE_REAL_CRACKS_COMPLETE_EXPLORATORY",
        "frozen_eligibility_sha256":hashlib.sha256(
            source_path.read_bytes()).hexdigest(),
        "trained_segmentation_sha256":hashlib.sha256(
            (trained/"v26_segmenter_frozen.json").read_bytes()
        ).hexdigest(),
        "e2e_score_sha256":hashlib.sha256(
            (e2e_dir/"v26_real_e2e_results.json").read_bytes()
        ).hexdigest(),
        "test_section_ids":eligibility["TEST_sections"],
        "buffer_section_ids":eligibility["BUFFER_sections"],
        "new_real_segmenters":{"ANZA_v1":"real fresh genuine AZConv",
                               "UNet":"same size/fitting effort"},
        "e2e_results":result["models"],
        "new_LIRA_must_be_trained_on_actual_predicted_pairs":True,
        "old_original_checkpoint_not_recovered":True,
        "source_selection_is_post_V26_label_availability":True,
        "not_unseen_confirmatory_test":True,
        "no_real_natural_fault_false_instance_labels":True,
        "no_claim_ANZA_better_than_published_seismic_SEG":True,
    }
    (out/"v27_summary.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "status":report["status"],
        "test_section_ids":report["test_section_ids"],
        "source_spatial_buffer":report["buffer_section_ids"],
        "model_results":result["models"],
    },indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--images",type=Path,required=True)
    p.add_argument("--annotations",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    execute(a.images,a.annotations,a.out)
