"""V28 posthoc end-to-end runner: topology threshold, not new weights.

This run is explicitly designed AFTER negative V27 TEST outcomes.
Model and raw CRACKS source remain unchanged. Freeze TRAIN/CAL-only
predicted candidate threshold before opening 108/124 again, but their
past exposure means any changed score is exploratory engineering only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.tver_v27_blue_cohort import (
    blue_available_sections,frozen_positive_split,
)
from scripts.tver_v26_train_real_segmenters import execute as fit_seg
from scripts.tver_v28_topology_calibration import execute as topology
from scripts.tver_v26_predicted_cracks_e2e import execute as full_e2e


def execute(images,annotations,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=blue_available_sections(images,annotations)
    mapping,cohort=frozen_positive_split(rows)
    if cohort["status"]!="V27_FROZEN_BLUE_AVAILABILITY_SOURCE_SPLIT":
        raise RuntimeError("STOP V28 same eligible cohort unavailable")
    trace=out/"v28_source_cohort.json"
    trace.write_text(json.dumps(cohort,indent=2)+"\n")
    trained=out/"new_train_model"
    segmentation=fit_seg(images,annotations,trained,split_mapping=mapping)
    receipt=topology(
        images,annotations,trained,mapping,
        out/"v28_topology_TRAIN_CAL_frozen.json")
    selection={
        name:d["selected_threshold"]
        for name,d in receipt["frontends"].items()}
    good={name:x for name,x in selection.items() if x is not None}
    if len(good)<2:
        # Never relax fit requirements and never inject GT candidate
        # just to force the ANZA relational head to fit.
        result={
            "status":"V28_STOP_NOT_ALL_FRONTENDS_TOPOLOGY_TRAINABLE",
            "topology_thresholds":selection,
            "V28_TEST_images_not_opened":True,
            "original_source_and_newly_trained_weights_available":True,
            "not_a_historical_checkpoint_replay":True,
            "previous_V27_TEST_already_inspected":True,
        }
        (out/"v28_summary.json").write_text(
            json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))
        return result
    original_thresholds={
        name:segmentation["frontends"][name]["selected_CAL_threshold"]
        for name in selection}
    for name,threshold in good.items():
        segmentation["frontends"][name]["selected_CAL_threshold"]=threshold
        segmentation["frontends"][name][
            "threshold_selected_for_predicted_SBPP_on_TRAIN_CAL"]=True
        segmentation["frontends"][name][
            "original_pixel_F1_threshold_V27"]=original_thresholds[name]
    # Preserve original .pt weights, update only a separate explicit
    # V28 receipt read by the unmodified E2E engine.
    new_receipt=trained/"v26_segmenter_frozen.json"
    new_receipt.write_text(
        json.dumps(segmentation,indent=2,ensure_ascii=False)+"\n")
    e2e=full_e2e(
        images,annotations,trained,
        out/"predicted_e2e",split_mapping=mapping)
    result={
        "status":"V28_POSTHOC_TOPOLOGY_ADAPTED_REAL_PREDICTED_E2E_EXECUTED",
        "original_V27_TEST_inspected_before_design":True,
        "same_BLUE_selected_source_cohort_not_external":True,
        "model_weight_SHA256_same_as_V27":all(
            segmentation["frontends"][name]["checkpoint_SHA256"]==
            receipt["frontends"][name]["unchanged_actual_model_SHA256"]
            for name in selection),
        "source_cohort_SHA256":hashlib.sha256(trace.read_bytes()).hexdigest(),
        "all_model_CAL_freeze_sha256":e2e["frozen_SHA256"],
        "topology_threshold_freeze_SHA256":hashlib.sha256(
            (out/"v28_topology_TRAIN_CAL_frozen.json").read_bytes()
        ).hexdigest(),
        "thresholds_old":original_thresholds,
        "thresholds_new":selection,
        "e2e":e2e["models"],
        "historical_weights_UNAVAILABLE":True,
        "no_independent_SOTA_claim":True,
    }
    (out/"v28_summary.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "status":result["status"],
        "cal_model_thresholds":selection,
        "results":e2e["models"],
    },indent=2,ensure_ascii=False))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--images",type=Path,required=True)
    p.add_argument("--annotations",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    execute(a.images,a.annotations,a.out)
