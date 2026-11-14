"""V45: source-section-OOF scene gate over unchanged V40 pairwise ranker.

Exploratory previously-viewed Thebe VAL1 REL_CAL; original TEST7 unopened.
Two distinct supervision targets: best candidate rank vs whether to link.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.tver_v40_pairwise_ranker import (
    FROZEN_ANZA_SHA, fit_same_budget_pair_ranker, pair_rank_scores,
)
from scripts.tver_v30_p0_context import paired_training_examples

METHODS=("raw_pair_rank_score","oof_scene_gate")
FOLDS=5
GATE_C=.3
CAL_THRESHOLDS=(.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,1.01)
CONFIDENCE=.95
RISK=.10
FEATURE_NAMES=(
    "max_rank_score","top1_minus_top2","top1_minus_median",
    "mean_rank_score","std_rank_score","log1p_n_candidates",
    "n_source_ports_over4","n_branches_over110",
    "fraction_visible_probability_gte_065","mean_visible_probability",
    "best_candidate_distance_over68","best_candidate_mean_probability",
    "best_candidate_mean_predicted_corridor_probability",
)


def feature_row(scene,scores):
    """Features are predicted image/SBPP/score evidence, not expert truth."""
    s=np.asarray(scores,np.float64)
    if len(s)!=len(scene["proposals"]) or not np.isfinite(s).all():
        raise ValueError("Rank scores do not match predicted pair records")
    p=np.asarray(scene["visible_prob"],np.float32)
    if p.shape!=(96,96) or not np.isfinite(p).all():
        raise ValueError("Invalid observed post-intervention prediction")
    if not len(s):
        return np.zeros(len(FEATURE_NAMES),np.float64)
    order=np.argsort(-s,kind="stable")
    top=float(s[order[0]])
    second=float(s[order[1]]) if len(order)>1 else top
    details=np.asarray(scene["proposals"][int(order[0])]["features"],np.float64)
    if details.shape!=(12,) or not np.isfinite(details).all():
        raise ValueError("Invalid prediction-derived pair features")
    vector=np.array([
        top,top-second,top-float(np.median(s)),float(np.mean(s)),
        float(np.std(s)),math.log1p(len(s)),float(scene["source_count"])/4.,
        float(scene["branch_count"])/110.,float(np.mean(p>=.65)),
        float(np.mean(p)),float(details[0]),float(details[5]),float(details[6]),
    ],np.float64)
    if not np.isfinite(vector).all():raise ValueError("Invalid scene summary")
    return vector


def source_fold_id(section):
    s=int(section)
    if not 0<=s<100:
        raise ValueError("Only TRAIN1 sections are allowed for OOF training")
    return s%FOLDS


def build_oof_gate(train_scenes):
    """Generate strictly source-held-out ranker scores for gate training."""
    train_ids={int(s["section"]) for s in train_scenes}
    if train_ids!=set(range(100)):
        raise ValueError("Original 100 TRAIN1 sections required for OOF gate")
    observations=[]
    for fold in range(FOLDS):
        fit_ids={s for s in train_ids if source_fold_id(s)!=fold}
        held_ids={s for s in train_ids if source_fold_id(s)==fold}
        if fit_ids & held_ids or len(fit_ids)!=80 or len(held_ids)!=20:
            raise RuntimeError("Source OOF split violation")
        fit_scenes=[s for s in train_scenes if int(s["section"]) in fit_ids]
        held_scenes=[s for s in train_scenes if int(s["section"]) in held_ids]
        _,label,feat,_=paired_training_examples(fit_scenes,.65)
        scaler,ranker=fit_same_budget_pair_ranker(feat,label)
        for scene in held_scenes:
            score=pair_rank_scores(scaler,ranker,scene)
            vector=feature_row(scene,score)
            ix=int(np.argmax(score)) if len(score) else None
            y=int(ix is not None and scene["kind"]=="LINK"
                  and scene["proposals"][ix]["y"]==1)
            observations.append((int(scene["section"]),fold,vector,y,scene["kind"]))
        print("V45_TRAIN_OOF_FOLD",fold,"heldout_sections",len(held_ids),
              "scenes",len(held_scenes),flush=True)
    if len(observations)!=400 or len({o[0] for o in observations})!=100:
        raise RuntimeError("OOF gate missing original 400 controlled TRAIN scenes")
    for sid,fold,*_ in observations:
        if source_fold_id(sid)!=fold:
            raise RuntimeError("Leak: OOF section scored by trained ranker")
    feats=np.stack([o[2] for o in observations])
    labels=np.array([o[3] for o in observations],int)
    if labels.sum()<8 or labels.sum()>len(labels)-8:
        raise RuntimeError("Insufficient OOF correctly ranked LINK examples")
    gate=make_pipeline(
        StandardScaler(),
        LogisticRegression(C=GATE_C,class_weight="balanced",
                           max_iter=1400,random_state=20261114))
    gate.fit(feats,labels)
    passport={
        "oof_splits_by_source_section":True,"n_source_sections":100,
        "n_oof_folds":FOLDS,"fit_sections_per_fold":80,
        "heldout_sections_per_fold":20,"n_scenes":len(observations),
        "positive_correct_top1_oof_scenes":int(labels.sum()),
        "negative_wrong_top1_or_nolink_oof_scenes":int(len(labels)-labels.sum()),
        "training_LINK_scenes":sum(o[4]=="LINK" for o in observations),
        "training_synthetic_NOLINK_scenes":sum(o[4]=="NO_LINK" for o in observations),
        "n_gate_features":len(FEATURE_NAMES),"gate_features":FEATURE_NAMES,
        "gate_C":GATE_C,"ranker_fitted_on_heldout_source_section":False,
    }
    return gate,passport


def score_scene_methods(cal_scenes,rank_scaler,ranker,gate):
    rows=[]
    for scene in cal_scenes:
        raw=pair_rank_scores(rank_scaler,ranker,scene)
        f=feature_row(scene,raw)
        idx=int(np.argmax(raw)) if len(raw) else None
        truth=int(idx is not None and scene["kind"]=="LINK"
                  and scene["proposals"][idx]["y"]==1)
        gate_score=float(gate.predict_proba(f[None,:])[0,1])
        m={
            "raw_pair_rank_score":float(raw[idx]) if idx is not None else None,
            "oof_scene_gate":gate_score if idx is not None else None,
        }
        for name,score in m.items():
            rows.append({
                "method":name,"section":int(scene["section"]),
                "gap_id":str(scene["id"]),"world":scene["kind"],
                "n_candidates":len(scene["proposals"]),
                "true_candidate_available":int(scene["has_true_proposal"]),
                "top1_correct":truth,
                "accept_score":score,
                "no_link_is_synthetic":True,
            })
    return rows


def cp_upper(count,n,alpha):
    if n<1 or not 0<=count<=n or not 0<alpha<1:
        raise ValueError("Binomial confidence parameters invalid")
    return float(1. if count==n else beta.ppf(1-alpha,count+1,n-count))


def evaluate(rows,*,expected_candidate_available=65):
    if not isinstance(expected_candidate_available,int) or not 0<=expected_candidate_available<=120:
        raise ValueError('Expected candidate count must be between 0 and 120')
    summary={}
    alpha=(1-CONFIDENCE)/(len(CAL_THRESHOLDS)*2)
    for name in METHODS:
        data=[r for r in rows if r["method"]==name]
        positive=[r for r in data if r["world"]=="LINK"]
        negative=[r for r in data if r["world"]=="NO_LINK"]
        if len(positive)!=120 or len(negative)!=120 or len({r["section"] for r in positive})!=60:
            raise RuntimeError("Original 60 sections, 120+120 scenes not reproduced")
        if sum(r["true_candidate_available"] for r in positive)!=expected_candidate_available:
            raise RuntimeError("Original predicted candidate availability changed, expected "+str(expected_candidate_available))
        scores=[r["accept_score"] for r in data if r["accept_score"] is not None]
        if any(not math.isfinite(s) or s<0 or s>1 for s in scores):
            raise ValueError("Bad gate score range")
        sweep=[]
        for t in CAL_THRESHOLDS:
            accepted=[r for r in data if r["accept_score"] is not None
                      and r["accept_score"]>=t]
            correct=sum(r["world"]=="LINK" and r["top1_correct"] for r in accepted)
            wrong_link={r["section"] for r in accepted
                        if r["world"]=="LINK" and not r["top1_correct"]}
            wrong_no={r["section"] for r in accepted if r["world"]=="NO_LINK"}
            upper_link=cp_upper(len(wrong_link),60,alpha)
            upper_no=cp_upper(len(wrong_no),60,alpha)
            sweep.append({
                "threshold":t,"correctly_accepted_link_gaps":int(correct),
                "n_accepted_scenes":len(accepted),
                "n_wrong_link_source_sections":len(wrong_link),
                "n_false_synthetic_nolink_source_sections":len(wrong_no),
                "joint_CP_upper_wrong_LINK":upper_link,
                "joint_CP_upper_synthetic_NOLINK":upper_no,
                "conditional_joint_95pct_10pct_gate":max(upper_link,upper_no)<=RISK,
            })
        unique=sorted(set(scores),reverse=True)
        best_zero=best_emp=0
        for t in unique:
            admitted=[r for r in data if r["accept_score"] is not None
                      and r["accept_score"]>=t]
            correct=sum(r["world"]=="LINK" and r["top1_correct"] for r in admitted)
            wrong_link={r["section"] for r in admitted
                        if r["world"]=="LINK" and not r["top1_correct"]}
            wrong_no={r["section"] for r in admitted if r["world"]=="NO_LINK"}
            if not wrong_link and not wrong_no:best_zero=max(best_zero,correct)
            if len(wrong_link)<=6 and len(wrong_no)<=6:best_emp=max(best_emp,correct)
        admitted=[r for r in sweep if r["conditional_joint_95pct_10pct_gate"]]
        summary[name]={
            "rank_top1_correct_before_acceptance":sum(r["top1_correct"] for r in positive),
            "n_scored_TOP1":len(scores),
            "n_predeclared_thresholds":len(CAL_THRESHOLDS),
            "simultaneous_CP_error_type_tests":2*len(CAL_THRESHOLDS),
            "simultaneous_CP_zero_error_section_risk_bound":cp_upper(0,60,alpha),
            "best_correct_with_conditional_simultaneous_CP_gate":max(
                [r["correctly_accepted_link_gaps"] for r in admitted],default=0),
            "best_correct_with_zero_observed_wrong_sections_posthoc":best_zero,
            "best_correct_at_descriptive_empirical_10pct_posthoc":best_emp,
            "predeclared_sweep":sweep,
            "CAL_already_viewed":True,
        }
    return summary


def execute(memmap_root,frontends_dir,out):
    import torch
    from scripts.tver_v34_anza_p0_relation_cal import _state_report
    from scripts.tver_v33_thebe_train_segmenters import read_memmap
    from scripts.tver_v33_thebe_relation_cal import (
        _candidate_scenes,_load_model,TRAIN_SOURCE_IDS,RELATION_CAL_SOURCE_IDS)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    receipt=_state_report(frontends_dir)
    if receipt["frontends"]["anza_v1"]["SHA256"]!=FROZEN_ANZA_SHA:
        raise RuntimeError("Original V33 trained ANZA source checkpoint changed")
    seg=_load_model(frontends_dir,"anza_v1",receipt)
    train,_=_candidate_scenes(read_memmap(memmap_root,"train_seismic_float32.npy"),
                             read_memmap(memmap_root,"train_fault_bool.npy"),
                             TRAIN_SOURCE_IDS,receipt["normalization"],seg,.65)
    if sum(s["has_true_proposal"] for s in train if s["kind"]=="LINK")!=116:
        raise RuntimeError("V34 TRAIN 116/200 candidate availability not reproduced")
    # OOF gate cannot see in-source ranker predictions during gate fit.
    gate,gate_passport=build_oof_gate(train)
    _,labels,features,matched_passport=paired_training_examples(train,.65)
    rank_scaler,ranker=fit_same_budget_pair_ranker(features,labels)
    cal,_=_candidate_scenes(read_memmap(memmap_root,"val_seismic_float32.npy"),
                           read_memmap(memmap_root,"val_fault_bool.npy"),
                           RELATION_CAL_SOURCE_IDS,receipt["normalization"],seg,.65)
    rows=score_scene_methods(cal,rank_scaler,ranker,gate)
    summary=evaluate(rows)
    for name in METHODS:
        if summary[name]["rank_top1_correct_before_acceptance"]!=41:
            raise RuntimeError("Original V40 41/120 pair-ranker top1 not reproduced")
    report={
        "status":"V45_REAL_THEBE_SECTION_OOF_TWO_STAGE_GATE_POSTHOC_COMPLETE_TEST7_UNOPENED",
        "frozen_ANZA_SHA256":FROZEN_ANZA_SHA,
        "source_receipt_SHA256":hashlib.sha256(
            (Path(frontends_dir)/"v33_THEBE_frontends_topology_TRAIN_CAL_frozen.json").read_bytes()).hexdigest(),
        "original_SOURCE_TRAIN_pairs":matched_passport,
        "scene_gate_TRAIN_5fold_oof":gate_passport,
        "original_VAL_REL_CAL_previously_viewed":True,
        "Thebe_external_TEST7_unopened":True,
        "natural_geological_NO_LINK_labels":False,
        "field_independence_claimed":False,
        "conditional_CP_is_not_geological_safety_guarantee":True,
        "summary":summary,
    }
    with (out/"V45_CASE_ROWS.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/"V45_RESULT.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"status":report["status"],"oof_gate":gate_passport,
                      "summary":{k:{m:v for m,v in row.items() if m!="predeclared_sweep"}
                                 for k,row in summary.items()}},
                      indent=2),flush=True)
    return report


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mmap-root",type=Path,required=True)
    ap.add_argument("--frontends-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    execute(a.mmap_root,a.frontends_dir,a.out_dir)
