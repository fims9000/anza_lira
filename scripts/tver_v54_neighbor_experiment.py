"""Compare frozen ANZA vs label-blind three-section consensus on Thebe.

Only original TRAIN1 and previously inspected VAL1 sections. Full-resolution
seismic amplitude remains observed; no expert gap or fault labels are passed
to model inference or 2.5-D support.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.tver_v54_full_section_infer import (
    MODEL_SHA256, THRESHOLDS, tiled_probability, _small_preview,
)
from trace_extraction.coverage import (
    annotated_trace_coverage, neighbor_consensus_support,
)

# Fixed before comparison, and not selected by the reference masks.
COHORTS=(("TRAIN1","train_seismic_float32.npy","train_fault_bool.npy",20),
         ("VAL1_ALREADY_SEEN","val_seismic_float32.npy","val_fault_bool.npy",43))
PIXEL_BUDGETS=(.02,.05,.10)
NEIGHBOR_POLICY=dict(spatial_radius=2,neighbor_threshold=.70,
                     center_floor=.12,max_boost=.35)


def _fixed_budget_profile(probability,reference,budgets=PIXEL_BUDGETS):
    result=[]
    for target in budgets:
        # Threshold derives from prediction distribution only, never labels.
        tau=float(np.quantile(probability,1.-target))
        tau=min(1.-1e-7,max(1e-7,tau))
        metric=annotated_trace_coverage(probability,reference,(tau,))[0]
        result.append({"target_density":target,"derived_threshold":tau,**metric})
    return result


def run(mmap_root,frontends_dir,out_dir):
    import torch
    from scripts.tver_v34_anza_p0_relation_cal import _state_report
    from scripts.tver_v33_thebe_relation_cal import _load_model
    from scripts.tver_v33_thebe_train_segmenters import read_memmap

    out=Path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    state=_state_report(frontends_dir)
    if (state["frontends"]["anza_v1"]["SHA256"]!=MODEL_SHA256
            or not state["heldout_THEBE_TEST7_NOT_OPENED"]):
        raise RuntimeError("Frozen Thebe source and TEST7 lock not reproduced")
    net=_load_model(frontends_dir,"anza_v1",state)
    torch.set_num_threads(2)
    scenes=[]
    for cohort,source_name,label_name,sid in COHORTS:
        # Only full-section predicted maps from observed neighboring slices.
        raw=read_memmap(mmap_root,source_name)
        predicted=[]
        receipts=[]
        for index in (sid-1,sid,sid+1):
            p,info=tiled_probability(
                raw[index],net,state["normalization"],
                tile_size=96,stride=64,batch_size=8)
            predicted.append(p)
            receipts.append(info)
        center=predicted[1]
        fused,context=neighbor_consensus_support(
            center,predicted[0],predicted[2],**NEIGHBOR_POLICY)
        del predicted,raw
        # Reference is read after the two predicted fields are frozen.
        expert=read_memmap(mmap_root,label_name)
        reference=np.array(expert[sid],dtype=bool,order="C",copy=True)
        del expert
        case=out/f"{cohort}_{sid}"
        case.mkdir(exist_ok=True)
        np.save(case/"V54_center_probability.npy",center,allow_pickle=False)
        np.save(case/"V54_neighbor_supported_probability.npy",fused,allow_pickle=False)
        _small_preview(center,case/"V54_center_preview.png")
        _small_preview(fused,case/"V54_neighbor_supported_preview.png")
        current=annotated_trace_coverage(center,reference,THRESHOLDS)
        enhanced=annotated_trace_coverage(fused,reference,THRESHOLDS)
        matched_a=_fixed_budget_profile(center,reference)
        matched_b=_fixed_budget_profile(fused,reference)
        report={
            "cohort":cohort,"section":sid,
            "contextual_neighbor_indices":[sid-1,sid+1],
            "model_sha256":MODEL_SHA256,
            "tiling_receipts":receipts,
            "support_policy":NEIGHBOR_POLICY,
            "support_summary":context,
            "center_threshold_profile":current,
            "neighbor_threshold_profile":enhanced,
            "center_equal_pixel_budget_profile":matched_a,
            "neighbor_equal_pixel_budget_profile":matched_b,
            "no_expert_truth_used_in_inference_or_fusion":True,
            "previously_inspected_field_not_independent_confirm":True,
            "no_natural_fault_instance_identity_ground_truth":True,
        }
        (case/"V54_2p5D_PROFILE.json").write_text(
            json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
        scenes.append(report)
        print("V54_CONTEXT_COMPARISON",cohort,sid,json.dumps({
            "boost":context,"center":matched_a,"fused":matched_b},
            ensure_ascii=False),flush=True)
    summary={
        "status":"V54_EXPLORATORY_LABEL_BLIND_THREE_SLICE_SUPPORT",
        "original_anaza_frozen_model_sha":MODEL_SHA256,
        "contexts":scenes,
        "no_TEST7_access":True,
        "no_hyperparameter_optimization_on_reference":True,
        "method_comparison_at_matching_foreground_pixel_budgets":True,
    }
    (out/"V54_2p5D_SUMMARY.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    return summary


def main():
    p=argparse.ArgumentParser()
    for name in ("mmap-root","frontends-dir","out-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    run(a.mmap_root,a.frontends_dir,a.out_dir)


if __name__=="__main__":
    main()
