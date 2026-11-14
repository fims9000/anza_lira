"""V28 posthoc TRAIN/CAL-only topology-aware segmentation threshold selection.

Never reads any source TEST section, uses oracle identity ONLY for TRAIN/CAL
threshold audit labels. True source/destination endpoints are NOT passed to
predicted SBPP candidate extraction. Evaluates V27 newly trained checkpoint
unchanged; any use of TEST sections 108/124 is after explicit freeze.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from scripts.cracks_real_oracle_gap_audit import sections
from scripts.tver_v26_predicted_cracks_e2e import (
    _img,_expert_gaps,_roi,_world,sbpp_proposals,is_true_relation,
)
from scripts.tver_v26_train_real_segmenters import (
    predict_patch,restore_model,
)

TOPOLOGY_THRESHOLDS=tuple(round(x/100,2)
                          for x in range(35,91,5))
CAL_POSITIVE_GAP_MIN=1
TRAIN_TRUE_PAIR_MIN=4
TRAIN_FALSE_PAIR_MIN=4


def _cached_gaps(zi,za,section_ids,model,norm):
    im_paths=sections(zi);lab_paths=sections(za,expert=True)
    cached=[]
    for sid in section_ids:
        image=_img(zi,im_paths[sid]).astype(np.float32)/255.
        expert=_img(za,lab_paths[sid])
        for gap in _expert_gaps(expert,sid):
            crop,origin,rect=_roi(image,gap["rectangle"])
            prob=predict_patch(model,crop,norm["mean"],norm["std"])
            cached.append({
                "id":gap["gap_id"],
                "origin":origin,"rect":rect,
                "ends":gap["true_ends"],
                "prob":prob,"gray":crop.mean(axis=2)})
    return cached


def _at_threshold(cases,threshold):
    receipts=[]
    for case in cases:
        source=_world(case["prob"],case["rect"],case["ends"],
                      case["origin"],"LINK")
        pred=sbpp_proposals(source,case["gray"],case["rect"],threshold)
        n_true=sum(is_true_relation(
            row,case["ends"],case["origin"])
            for row in pred["proposals"])
        n_cands=len(pred["proposals"])
        receipts.append({
            "case_id":case["id"],
            "status":pred["status"],
            "n_predicted_source_ports":pred["source_count"],
            "n_predicted_candidates":n_cands,
            "n_true_candidates":n_true,
            "n_false_candidates":n_cands-n_true,
            "correct_candidate_available":bool(n_true),
            "is_dense_prediction":pred["status"]=="DEGENERATE_DENSE_PREDICTED_MASK",
            "predicted_mask_fraction":float(np.mean(source>=threshold)),
        })
    return {
        "n_controlled_gaps":len(cases),
        "n_sources":sum(r["n_predicted_source_ports"]>0 for r in receipts),
        "n_with_any_candidate":sum(r["n_predicted_candidates"]>0 for r in receipts),
        "n_true_pair_gaps":sum(r["correct_candidate_available"] for r in receipts),
        "n_true_pairs":sum(r["n_true_candidates"] for r in receipts),
        "n_false_pairs":sum(r["n_false_candidates"] for r in receipts),
        "n_candidates":sum(r["n_predicted_candidates"] for r in receipts),
        "n_dense_rejected":sum(r["is_dense_prediction"] for r in receipts),
        "mean_predicted_mask_fraction":float(np.mean(
            [r["predicted_mask_fraction"] for r in receipts]))
            if receipts else None,
        "case_rows":receipts,
    }


def choose(train,cal):
    feasible=[]
    for key in TOPOLOGY_THRESHOLDS:
        tr=train[str(key)];v=cal[str(key)]
        if (tr["n_true_pairs"]>=TRAIN_TRUE_PAIR_MIN and
            tr["n_false_pairs"]>=TRAIN_FALSE_PAIR_MIN and
            v["n_true_pair_gaps"]>=CAL_POSITIVE_GAP_MIN):
            # CAL positive candidate coverage is the primary target,
            # fewer spurious candidates second. TEST scores unavailable.
            obj=(v["n_true_pair_gaps"],
                 -v["n_candidates"],
                 v["n_sources"],
                 -v["n_dense_rejected"],
                 key)
            feasible.append((obj,key))
    if not feasible:
        return None
    return float(max(feasible)[1])


def execute(images,annotations,trained_root,split_mapping,out):
    """Call only AFTER newly trained frontends but BEFORE any V28 TEST."""
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    trained_root=Path(trained_root)
    with (trained_root/"v26_segmenter_frozen.json").open() as f:
        segmentation=json.load(f)
    if segmentation.get("TEST_pixel_arrays_or_labels_loaded_before_freeze"):
        raise RuntimeError("STOP V28 segmentation TEST already opened")
    ids_train=sorted(s for s in split_mapping if split_mapping[s]=="TRAIN")
    ids_cal=sorted(s for s in split_mapping if split_mapping[s]=="CAL")
    ids_test=sorted(s for s in split_mapping if split_mapping[s]=="TEST")
    if (len(ids_train)<5 or len(ids_cal)!=2 or
        ids_test!=[108,124]):
        raise RuntimeError("STOP source cohort/heldout population changed")
    report={
        "status":"V28_POSTHOC_TRAIN_CAL_ONLY_TOPOLOGY_SELECTION",
        "posthoc_after_V27_TEST_inspection":True,
        "V28_TEST_images_labels_not_decoded_for_selection":True,
        "reserved_TEST_section_ids":ids_test,
        "TRAIN_section_ids":ids_train,"CAL_section_ids":ids_cal,
        "source_selection_has_BLUE_availability_bias":True,
        "allowed_thresholds":TOPOLOGY_THRESHOLDS,
        "TRAIN_true_pair_min":TRAIN_TRUE_PAIR_MIN,
        "TRAIN_false_pair_min":TRAIN_FALSE_PAIR_MIN,
        "CAL_positive_gap_with_true_candidate_min":CAL_POSITIVE_GAP_MIN,
        "frontends":{},
    }
    with zipfile.ZipFile(images) as zi,zipfile.ZipFile(annotations) as za:
        for name in ("unet","anza_v1"):
            model=restore_model(trained_root,name,segmentation)
            train=_cached_gaps(zi,za,ids_train,model,
                               segmentation["train_normalization"])
            cal=_cached_gaps(zi,za,ids_cal,model,
                             segmentation["train_normalization"])
            t={};v={}
            for threshold in TOPOLOGY_THRESHOLDS:
                t[str(threshold)]=_at_threshold(train,threshold)
                v[str(threshold)]=_at_threshold(cal,threshold)
            chosen=choose(t,v)
            report["frontends"][name]={
                "selected_threshold":chosen,
                "status":("FROZEN_VALID_TRAIN_CAL_PREDICTED_PORT_TOPOLOGY"
                          if chosen is not None else
                          "STOP_NO_TRAIN_CAL_CORRECT_CANDIDATE_THRESHOLD"),
                "previous_pixel_F1_threshold":
                    segmentation["frontends"][name]["selected_CAL_threshold"],
                "unchanged_actual_model_SHA256":
                    segmentation["frontends"][name]["checkpoint_SHA256"],
                "n_cached_original_TRAIN_gap_ROIs":len(train),
                "n_cached_original_CAL_gap_ROIs":len(cal),
                "TRAIN_metrics_by_threshold":t,
                "CAL_metrics_by_threshold":v,
            }
    # This is the FROZEN, outcome-blind V28 topology selection receipt.
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "status":report["status"],
        "frontends":{
            n:{"selected":x["selected_threshold"],"status":x["status"],
               "old_CAL_threshold":x["previous_pixel_F1_threshold"],
               "CAL_at_selected":(
                 x["CAL_metrics_by_threshold"].get(
                     str(x["selected_threshold"]),{}))}
            for n,x in report["frontends"].items()}
    },indent=2,ensure_ascii=False))
    return report
