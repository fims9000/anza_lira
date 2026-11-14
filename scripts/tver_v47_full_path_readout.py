"""V46: original Thebe consistent dense mask recovery + trained pair ranker + OOF scene gate.

Scientific scope: posthoc original Thebe VAL1, synthetic gap/NO_LINK worlds.
No untouched TEST7 data and no natural fault instance-level safety claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts.tver_v43_dense_mask_fallback import adaptive_threshold, DENSE_STOP
from scripts.tver_v45_oof_scene_gate import (
    build_oof_gate,score_scene_methods,evaluate,feature_row,CAL_THRESHOLDS)
from scripts.tver_v40_pairwise_ranker import (
    FROZEN_ANZA_SHA,fit_same_budget_pair_ranker,pair_rank_scores)
from scripts.tver_v30_p0_context import paired_training_examples

TARGET_Q=.70
MAX_PAIRS=48


def apply_frozen_dense_fallback(scenes,propose,is_true_relation):
    """Only densest original scenes are changed, with truth attached afterwards."""
    observed=[]
    rescues=0
    losses=0
    dense=0
    fallback_used=0
    for scene in scenes:
        orig=scene["proposals"]
        orig_truth=any(int(p["y"]) for p in orig)
        if scene["status"]!=DENSE_STOP:
            # Absolutely no source/target/label changes outside dense failures.
            observed.append(scene)
            continue
        dense+=1
        if orig:
            raise RuntimeError("Original dense refusal has non-empty candidates")
        tau=adaptive_threshold(scene["visible_prob"],TARGET_Q)
        raw=propose(scene["visible_prob"],scene["gray"],scene["rectangle"],tau)
        proposals=raw["proposals"]
        if len(proposals)>MAX_PAIRS:
            raise RuntimeError("Changed candidate budget")
        if any("y" in p or "true" in p for p in proposals):
            raise ValueError("Hidden expert identity leaked into proposals")
        # Labels are read exclusively AFTER the full candidate list was frozen.
        updated=[dict(p) for p in proposals]
        for p in updated:
            p["y"]=int(scene["kind"]=="LINK" and is_true_relation(
                p,scene["true_ends"],scene["origin"]))
        new=dict(scene)
        new.update({
            "proposals":updated,"source_count":raw["source_count"],
            "branch_count":raw["hard_branches"],"status":raw["status"],
            "has_true_proposal":any(int(p["y"]) for p in updated),
            "adaptive_tau":tau,
        })
        fallback_used+=1
        if new["kind"]=="LINK" and new["has_true_proposal"] and not orig_truth:
            rescues+=1
        if new["kind"]=="LINK" and not new["has_true_proposal"] and orig_truth:
            losses+=1
        observed.append(new)
    if len(observed)!=len(scenes) or losses:
        raise RuntimeError("Lost original candidate or scene identity")
    return observed,{
        "dense_worlds_including_synthetic_no_link":dense,
        "fallback_worlds":fallback_used,
        "new_true_candidate_LINK_rescues":rescues,
        "original_non_dense_scenes_unchanged":True,
        "original_48_pair_budget":MAX_PAIRS,
        "original_label_blind_quantile_q":TARGET_Q,
    }


def evidence_counts(scenes):
    pos=[s for s in scenes if s["kind"]=="LINK"]
    neg=[s for s in scenes if s["kind"]=="NO_LINK"]
    if len(pos)!=len(neg):
        raise RuntimeError("Controlled positive/negative world identity mismatch")
    if len({(s["section"],s["id"]) for s in pos})!=len(pos):
        raise RuntimeError("Repeated source section/gap pair")
    if {(s["section"],s["id"]) for s in pos}!={(s["section"],s["id"]) for s in neg}:
        raise RuntimeError("Paired LINK and NO_LINK source identities changed")
    return {
        "LINK_worlds":len(pos),
        "synthetic_NOLINK_worlds":len(neg),
        "candidate_available_LINK":sum(bool(s["has_true_proposal"]) for s in pos),
        "geometric_top1_LINK":sum(
            bool(s["proposals"] and s["proposals"][int(np.argmax(
                [p["geometry"] for p in s["proposals"]]))]["y"]) for s in pos),
        "LINK_worlds_without_proposals":sum(not s["proposals"] for s in pos),
        "NOLINK_worlds_with_any_proposals":sum(bool(s["proposals"]) for s in neg),
        "max_candidate_count":max([len(s["proposals"]) for s in scenes],default=0),
        "source_sections":len({s["section"] for s in pos}),
    }


def restore_controls(dir_v43,dir_v45):
    """Immutable completed earlier source receipts are read, not modified."""
    v43files=list(Path(dir_v43).rglob("V43_RESULT.json"))
    v45files=list(Path(dir_v45).rglob("V45_RESULT.json"))
    if len(v43files)!=1 or len(v45files)!=1:
        raise RuntimeError("Missing immutable completed V43 and V45 source receipts")
    v43=json.loads(v43files[0].read_text(encoding="utf8"))
    v45=json.loads(v45files[0].read_text(encoding="utf8"))
    if (v43["status"]!="V43_DENSE_MASK_ADAPTIVE_QUANTILE_POSTHOC_EVALUATED_TEST7_UNOPENED"
        or v43["summary"]["REL_CAL_ALREADY_VIEWED/dense_q70"]["correct_candidate_available"]!=75
        or v43["summary"]["TRAIN/dense_q70"]["correct_candidate_available"]!=125
        or not v43["never_read_Thebe_TEST7"]):
        raise RuntimeError("V43 immutable candidate gain receipt mismatch")
    if (v45["status"]!="V45_REAL_THEBE_SECTION_OOF_TWO_STAGE_GATE_POSTHOC_COMPLETE_TEST7_UNOPENED"
        or v45["summary"]["oof_scene_gate"]["rank_top1_correct_before_acceptance"]!=41
        or v45["summary"]["oof_scene_gate"]["best_correct_with_conditional_simultaneous_CP_gate"]!=0
        or v45["summary"]["oof_scene_gate"]["best_correct_at_descriptive_empirical_10pct_posthoc"]!=3
        or not v45["Thebe_external_TEST7_unopened"]):
        raise RuntimeError("V45 original two-stage receipt mismatch")
    return {
        "V43_source_status":v43["status"],
        "V43_candidate_availability_LINK_CAL":75,
        "V43_candidate_availability_LINK_TRAIN":125,
        "V45_source_status":v45["status"],
        "V45_baseline_rank_top1_CAL":41,
        "V45_baseline_empirical_risk_posthoc_accepted_correct_CAL":3,
        "V45_baseline_CP_conditional_accepted_correct_CAL":0,
        "control_receipts_immutable":True,
    }




def audited_path_readout(augmented_cal,rank_scaler,ranker,gate,score_rows,risk_summary):
    """Frozen repository straight/maximin path readout, expert truth only for scoring.

    This is a controlled-gap geometric metric, NOT autonomous scene recovery.
    Gate thresholds and their conditional source-risk decisions come from
    the unchanged V46 evaluation, and are merely JOINED to scored paths.
    """
    from scripts.tver_v26_predicted_cracks_e2e import _path_success
    result=[]
    for scene in augmented_cal:
        if scene["kind"]!="LINK":
            continue
        scores=pair_rank_scores(rank_scaler,ranker,scene)
        ix=int(np.argmax(scores)) if len(scores) else None
        correct=bool(ix is not None and scene["proposals"][ix]["y"]==1)
        if correct:
            pair=scene["proposals"][ix]
            straight=bool(_path_success(scene,pair,method="straight"))
            maximin=bool(_path_success(scene,pair,method="maximin"))
        else:
            straight=maximin=False
        gate_score=float(gate.predict_proba(feature_row(scene,scores)[None,:])[0,1]) if ix is not None else None
        result.append({
            "section":int(scene["section"]),"gap_id":str(scene["id"]),
            "n_candidates":len(scene["proposals"]),
            "correct_top1_pair":int(correct),
            "score_scene_gate":gate_score,
            "successful_straight_gap_path":int(straight),
            "successful_maximin_gap_path":int(maximin),
        })
    if len(result)!=120 or len({r["section"] for r in result})!=60:
        raise RuntimeError("Original real Thebe 120 LINK gap paths not reproduced")
    scoped_rows=[r for r in score_rows if r["method"]=="oof_scene_gate"]
    if len(scoped_rows)!=240:
        raise RuntimeError("V46 unchanged gate cohort missing")
    linked={(r["section"],r["gap_id"]):r for r in result}
    for row in scoped_rows:
        if row["world"]=="LINK":
            p=linked[(int(row["section"]),str(row["gap_id"]))]
            if bool(p["correct_top1_pair"])!=bool(row["top1_correct"]):
                raise RuntimeError("Gate and path evaluation scored different top1 candidate")
            if row["accept_score"]!=p["score_scene_gate"]:
                raise RuntimeError("Gate score mismatch at same source gap")
    def audit_t(t):
        accepted=[r for r in scoped_rows
                  if r["accept_score"] is not None and r["accept_score"]>=t]
        bad_link={r["section"] for r in accepted
                  if r["world"]=="LINK" and not r["top1_correct"]}
        bad_no={r["section"] for r in accepted if r["world"]=="NO_LINK"}
        positive=[linked[(r["section"],r["gap_id"])] for r in accepted
                  if r["world"]=="LINK" and r["top1_correct"]]
        return {
            "threshold":float(t),
            "correct_accepted_pairs":len(positive),
            "straight_correct_gap_paths":sum(r["successful_straight_gap_path"] for r in positive),
            "maximin_correct_gap_paths":sum(r["successful_maximin_gap_path"] for r in positive),
            "wrong_LINK_source_sections":len(bad_link),
            "false_synthetic_NOLINK_source_sections":len(bad_no),
        }

    fixed=[audit_t(t) for t in CAL_THRESHOLDS]
    gate_sweep=risk_summary["oof_scene_gate"]["predeclared_sweep"]
    if len(gate_sweep)!=len(fixed):
        raise RuntimeError("Frozen V46 risk thresholds drifted")
    for a,b in zip(fixed,gate_sweep):
        if (a["threshold"]!=b["threshold"]
            or a["correct_accepted_pairs"]!=b["correctly_accepted_link_gaps"]
            or a["wrong_LINK_source_sections"]!=b["n_wrong_link_source_sections"]
            or a["false_synthetic_NOLINK_source_sections"]!=b["n_false_synthetic_nolink_source_sections"]):
            raise RuntimeError("Actual path and frozen V46 source-risk gate inconsistent")
        a["conditional_joint_95pct_10pct_gate"]=b["conditional_joint_95pct_10pct_gate"]

    unique_scores=sorted({r["accept_score"] for r in scoped_rows
                          if r["accept_score"] is not None},reverse=True)
    empirical=[]
    for t in unique_scores:
        a=audit_t(t)
        if a["wrong_LINK_source_sections"]<=6 and a["false_synthetic_NOLINK_source_sections"]<=6:
            empirical.append(a)
    under_risk=[r for r in fixed if r["conditional_joint_95pct_10pct_gate"]]
    return {
        "status":"V47_REAL_THEBE_ACTUAL_PATH_READOUT_POSTHOC_COMPLETE_TEST7_UNOPENED",
        "controlled_LINK_gaps":120,"source_sections":60,
        "correct_top1_predicted_pairs":sum(r["correct_top1_pair"] for r in result),
        "successful_straight_path_given_top1_correct":sum(r["successful_straight_gap_path"] for r in result),
        "successful_maximin_path_given_top1_correct":sum(r["successful_maximin_gap_path"] for r in result),
        "best_correct_straight_paths_under_conditional_simultaneous_CP":max(
            [r["straight_correct_gap_paths"] for r in under_risk],default=0),
        "best_correct_maximin_paths_under_conditional_simultaneous_CP":max(
            [r["maximin_correct_gap_paths"] for r in under_risk],default=0),
        "best_correct_straight_paths_empirical_10pct_posthoc":max(
            [r["straight_correct_gap_paths"] for r in empirical],default=0),
        "best_correct_maximin_paths_empirical_10pct_posthoc":max(
            [r["maximin_correct_gap_paths"] for r in empirical],default=0),
        "scoring_precision_recall_threshold":.80,
        "scoring_dilation_tolerance_pixels":2,
        "expert_centerline_only_for_path_metric":True,
        "frozen_repository_path_modes":["straight","maximin"],
        "no_TEST7":True,
        "no_natural_geological_instance_truth":True,
        "predeclared_threshold_sweep":fixed,
        "case_rows":result,
    }

def execute(memmap_root,frontends_dir,v43_receipt,v45_receipt,out):
    import torch
    from scripts.tver_v34_anza_p0_relation_cal import _state_report
    from scripts.tver_v33_thebe_train_segmenters import read_memmap
    from scripts.tver_v33_thebe_relation_cal import (
        _candidate_scenes,_load_model,TRAIN_SOURCE_IDS,RELATION_CAL_SOURCE_IDS)
    from scripts.tver_v26_predicted_cracks_e2e import sbpp_proposals,is_true_relation

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    controls=restore_controls(v43_receipt,v45_receipt)
    model_meta=_state_report(frontends_dir)
    if model_meta["frontends"]["anza_v1"]["SHA256"]!=FROZEN_ANZA_SHA:
        raise RuntimeError("Frozen original V33 ANZA checkpoint checksum changed")
    segmenter=_load_model(frontends_dir,"anza_v1",model_meta)

    original_train,_=_candidate_scenes(
        read_memmap(memmap_root,"train_seismic_float32.npy"),
        read_memmap(memmap_root,"train_fault_bool.npy"),TRAIN_SOURCE_IDS,
        model_meta["normalization"],segmenter,.65)
    orig_train_counts=evidence_counts(original_train)
    if (orig_train_counts["candidate_available_LINK"]!=116 or
        orig_train_counts["LINK_worlds"]!=200):
        raise RuntimeError("Original V34 TRAIN source not reproduced")
    augmented_train,train_change=apply_frozen_dense_fallback(
        original_train,sbpp_proposals,is_true_relation)
    train_counts=evidence_counts(augmented_train)
    if train_counts["candidate_available_LINK"]!=125:
        raise RuntimeError("Original V43 q.70 TRAIN candidates 125/200 not reproduced")
    print("V46_ANNOTATION_SAFE_TRAIN",train_counts,train_change,flush=True)

    # Source-section OOF gate training on *augmented* same-policy candidates.
    gate,gate_passport=build_oof_gate(augmented_train)
    _,train_labels,train_features,paired_passport=paired_training_examples(
        augmented_train,.65)
    rank_scaler,ranker=fit_same_budget_pair_ranker(train_features,train_labels)

    original_cal,_=_candidate_scenes(
        read_memmap(memmap_root,"val_seismic_float32.npy"),
        read_memmap(memmap_root,"val_fault_bool.npy"),RELATION_CAL_SOURCE_IDS,
        model_meta["normalization"],segmenter,.65)
    orig_cal_counts=evidence_counts(original_cal)
    if (orig_cal_counts["candidate_available_LINK"]!=65 or
        orig_cal_counts["geometric_top1_LINK"]!=1 or
        orig_cal_counts["LINK_worlds_without_proposals"]!=18 or
        orig_cal_counts["LINK_worlds"]!=120):
        raise RuntimeError("Original V34 real Thebe VAL CAL source drift")
    augmented_cal,cal_change=apply_frozen_dense_fallback(
        original_cal,sbpp_proposals,is_true_relation)
    cal_counts=evidence_counts(augmented_cal)
    if (cal_counts["candidate_available_LINK"]!=75 or
        cal_counts["LINK_worlds_without_proposals"]!=0 or
        cal_counts["NOLINK_worlds_with_any_proposals"]!=120):
        raise RuntimeError("Original V43 q.70 VAL CAL candidate audit not reproduced")

    rows=score_scene_methods(augmented_cal,rank_scaler,ranker,gate)
    summary=evaluate(rows)
    path_evidence=audited_path_readout(augmented_cal,rank_scaler,ranker,gate,rows,summary)
    score_before=summary["oof_scene_gate"]["rank_top1_correct_before_acceptance"]
    report={
        "status":"V46_REAL_THEBE_DENSE_RECOVERY_OOF_GATE_INTEGRATED_POSTHOC_TEST7_UNOPENED",
        "original_ANZA_SHA256":FROZEN_ANZA_SHA,
        "original_checkpoint_receipt_SHA256":hashlib.sha256(
            (Path(frontends_dir)/"v33_THEBE_frontends_topology_TRAIN_CAL_frozen.json").read_bytes()).hexdigest(),
        "immutable_V43_V45_controls":controls,
        "original_TRAIN":orig_train_counts,"q70_TRAIN":train_counts,
        "original_CAL":orig_cal_counts,"q70_CAL":cal_counts,
        "TRAIN_fallback_details":train_change,"CAL_fallback_details":cal_change,
        "q70_TRAIN_source_fold_OOF_gate_receipt":gate_passport,
        "q70_TRAIN_matched_negative_ranker_receipt":paired_passport,
        "new_ranker_label_budget_different_from_V40":True,
        "no_TEST7_access":True,
        "previously_viewed_REL_CAL_posthoc":True,
        "source_sections_one_geological_field_independent_units_unverified":True,
        "natural_geological_fault_instance_truth_unavailable":True,
        "actual_full_path_reconstruction_evaluated":False,
        "summary":summary,
    }
    report["full_controlled_gap_path_readout_V47"]={
        key:value for key,value in path_evidence.items() if key not in ("case_rows","predeclared_threshold_sweep")}
    (out/"V47_PATH_RESULT.json").write_text(
        json.dumps({key:value for key,value in path_evidence.items() if key!="case_rows"},
                   indent=2)+"\\n",encoding="utf8")
    with (out/"V47_PATH_CASE_ROWS.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(path_evidence["case_rows"][0]))
        writer.writeheader();writer.writerows(path_evidence["case_rows"])
    with (out/"V46_CASE_ROWS.csv").open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (out/"V46_RESULT.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps({
        "status":report["status"],"original":orig_cal_counts,"q70":cal_counts,
        "q70_train_OOF_correct":gate_passport["positive_correct_top1_oof_scenes"],
        "q70_rank_top1_correct":score_before,
        "full_path_result":{key:value for key,value in path_evidence.items()
                            if key not in ("case_rows","predeclared_threshold_sweep")},
        "score_acceptance":{method:{k:v for k,v in values.items() if k!="predeclared_sweep"}
                            for method,values in summary.items()},
    },indent=2),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mmap-root",type=Path,required=True)
    p.add_argument("--frontends-dir",type=Path,required=True)
    p.add_argument("--v43-receipt-dir",type=Path,required=True)
    p.add_argument("--v45-receipt-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    execute(a.mmap_root,a.frontends_dir,a.v43_receipt_dir,a.v45_receipt_dir,a.out_dir)
