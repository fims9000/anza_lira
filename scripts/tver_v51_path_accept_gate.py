"""V51 path-aware abstention: source-block OOF ranker AND CNN path evidence.

Targets are actual controlled gap-path success, not merely correct top1 pair.
Synthetic NO_LINK is not real geological fault-instance negative supervision.
Only TRAIN1 expert centerlines enter training labels; CAL truths are metrics.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.tver_v40_pairwise_ranker import fit_same_budget_pair_ranker,pair_rank_scores
from scripts.tver_v30_p0_context import paired_training_examples
from scripts.tver_v45_oof_scene_gate import (
    feature_row as old_gate_features,CAL_THRESHOLDS,CONFIDENCE,RISK)
from scripts.tver_v50_conditional_path import (
    fit_path_head,decode_candidate,path_heatmap)
from scripts.tver_v49_boundary_path import hidden_path_success

PATH_FEATURE_NAMES=(
    "decoded_path_inside_mean_model_probability",
    "decoded_path_inside_q20_model_probability",
    "decoded_path_mean_minus_erase_rectangle_mean_probability",
    "decoded_path_length_over_endpoint_chord",
    "decoded_path_mean_perpendicular_offset_to_endpoint_chord",
    "decoded_path_fraction_inside_erase_rectangle",
)
FOLDS=5
SOURCE_TRAIN_SECTIONS=100
MIN_POSITIVES_FOR_GATE=8
C=.3


def source_block(section):
    sid=int(section)
    if sid<0 or sid>=100:
        raise ValueError("OOF must use Thebe TRAIN1 source section ID 0..99")
    return sid//20


def path_only_features(path_net,scene,proposal):
    """Strictly observed path quantities; no expert labels or section IDs."""
    path,meta=decode_candidate(path_net,scene,proposal,mode="learned_ribbon")
    heat=path_heatmap(path_net,scene,proposal)
    a=np.asarray(proposal["source"],float)
    b=np.asarray(proposal["target"],float)
    d=b-a
    norm=max(float(np.linalg.norm(d)),1e-6)
    rect=scene["rectangle"]
    y0,y1,x0,x1=map(int,rect)
    points=np.asarray(path,int).reshape(-1,2)
    if len(points)==0:
        return np.zeros(6,np.float64),path,meta
    yi=np.clip(points[:,0],0,95)
    xi=np.clip(points[:,1],0,95)
    inside=((yi>=y0)&(yi<y1)&(xi>=x0)&(xi<x1))
    scores=np.asarray(heat[yi[inside],xi[inside]],float)
    mean=float(np.mean(scores)) if len(scores) else 0.
    quantile=float(np.quantile(scores,.2)) if len(scores) else 0.
    box_mean=float(np.mean(heat[y0:y1,x0:x1]))
    real_pts=points.astype(float)
    t=np.clip(((real_pts-a[None,:])@d)/(norm**2),0,1)
    foot=a[None,:]+t[:,None]*d
    distance=float(np.mean(np.linalg.norm(real_pts-foot,axis=1)))
    features=np.array([
        mean,quantile,mean-box_mean,
        len(points)/norm,distance/10.,float(np.mean(inside))
    ],np.float64)
    if not np.isfinite(features).all():
        raise RuntimeError("Nonfinite observed path-only evidence")
    return features,path,meta


def full_scene_features(path_net,scene,proposal,rank_scores):
    original=old_gate_features(scene,rank_scores)
    if proposal is None:
        return np.r_[original,np.zeros(6)],tuple(),{"anchor_status":"NO_CANDIDATES"}
    features,path,meta=path_only_features(path_net,scene,proposal)
    full=np.r_[original,features]
    if full.shape!=(19,) or not np.isfinite(full).all():
        raise ValueError("Invalid source-conditioned gate feature vector")
    return full,path,meta


def fit_path_success_gate(train,output_dir):
    """5 contiguous source blocks held out from BOTH ranker and path head."""
    src={int(s["section"]) for s in train}
    if src!=set(range(SOURCE_TRAIN_SECTIONS)):
        raise RuntimeError("Original 100 source TRAIN sections required")
    root=Path(output_dir)
    root.mkdir(parents=True,exist_ok=True)
    feats=[];labels=[];worlds=[];counts=[]
    for fold in range(FOLDS):
        fit_sections={s for s in src if source_block(s)!=fold}
        hold_sections={s for s in src if source_block(s)==fold}
        if fit_sections & hold_sections or len(fit_sections)!=80 or len(hold_sections)!=20:
            raise RuntimeError("Source-block OOF leak or missing 20-section block")
        fit=[s for s in train if int(s["section"]) in fit_sections]
        hold=[s for s in train if int(s["section"]) in hold_sections]
        _,lab,scalars,_=paired_training_examples(fit,.65)
        scale,ranker=fit_same_budget_pair_ranker(scalars,lab)
        pathnet,receipt=fit_path_head(fit,root/f"V51_OOF_FOLD_{fold}")
        npath=0
        for scene in hold:
            rank=pair_rank_scores(scale,ranker,scene)
            idx=int(np.argmax(rank)) if len(rank) else None
            proposal=scene["proposals"][idx] if idx is not None else None
            observed,decoded,_=full_scene_features(pathnet,scene,proposal,rank)
            # Truth consulted ONLY after both path and all 19 model features exist.
            path_good=int(
                idx is not None and scene["kind"]=="LINK" and
                int(proposal["y"])==1 and hidden_path_success(scene,decoded))
            feats.append(observed)
            labels.append(path_good)
            worlds.append((int(scene["section"]),source_block(scene["section"]),scene["kind"]))
            npath+=path_good
        if len(hold)!=80:
            raise RuntimeError("Original heldout fold must yield 20 sections x4 worlds")
        counts.append({
            "fold":fold,"fit_source_sections":80,"held_source_sections":20,
            "source_section_indices":sorted(hold_sections),
            "correct_OOF_path_labels":int(npath),
            "fitted_path_checkpoint_sha256":receipt["checkpoint_SHA256"],
            "TRAIN_only_path_head":True,
        })
        print("V51_BLOCK_OOF_PATH_FOLD",fold,"n_scored",len(hold),
              "successful_paths",npath,flush=True)
    data=np.asarray(feats,np.float64)
    targets=np.asarray(labels,int)
    if data.shape!=(400,19) or len(worlds)!=400 or len(set(t[0] for t in worlds))!=100:
        raise RuntimeError("Original 100 sections/400 scene block-OOF cohort drift")
    if any(source_block(sid)!=fold for sid,fold,_ in worlds):
        raise RuntimeError("OOF source section was scored in fitted block")
    positives=int(targets.sum())
    passport={
        "source_block_not_modulo_folds":True,
        "n_slices":100,"n_folds":FOLDS,
        "worlds_count":len(worlds),
        "positive_successful_OOF_paths":positives,
        "negative_wrong_path_or_synthetic_NOLINK":len(worlds)-positives,
        "synthetic_NOLINK_count":sum(w=="NO_LINK" for _,_,w in worlds),
        "feature_names":list(old_gate_features.__globals__["FEATURE_NAMES"])+list(PATH_FEATURE_NAMES),
        "n_gate_features":19,"fold_receipts":counts,
        "all_gate_observations_from_ranker_AND_path_net_not_fitted_on_source_block":True,
        "TRAIN_expert_line_only_used_as_target":True,
    }
    if positives<MIN_POSITIVES_FOR_GATE:
        # Do not train a nearly constant classifier or lower this criterion
        # merely because original CAL has already been viewed.
        raise RuntimeError(
            "STOP insufficient genuinely out-of-fold successful TRAIN paths "
            f"({positives}/400); V51 path-aware gate is not identifiable")
    gate=make_pipeline(StandardScaler(),
        LogisticRegression(C=C,class_weight="balanced",max_iter=1400,
                           random_state=20261114))
    gate.fit(data,targets)
    return gate,passport


def score_source_path_worlds(scenes,scale,ranker,pathnet,gate):
    """Full-train ranker/path net + source-block OOF-trained gate on CAL scenes."""
    out=[]
    for scene in scenes:
        scores=pair_rank_scores(scale,ranker,scene)
        idx=int(np.argmax(scores)) if len(scores) else None
        pair=scene["proposals"][idx] if idx is not None else None
        vector,path,meta=full_scene_features(pathnet,scene,pair,scores)
        accept=float(gate.predict_proba(vector[None,:])[0,1]) if idx is not None else None
        correct_pair=int(pair is not None and scene["kind"]=="LINK" and pair["y"]==1)
        path_correct=int(bool(correct_pair) and hidden_path_success(scene,path))
        out.append({
            "method":"path_OOF_gate",
            "section":int(scene["section"]),"gap_id":str(scene["id"]),
            "world":scene["kind"],"candidate_available":int(scene["has_true_proposal"]),
            "candidate_count":len(scene["proposals"]),
            "correct_top1_pair":correct_pair,"correct_decoded_path":path_correct,
            "accept_score":accept,"path_anchor_status":meta["anchor_status"],
        })
    return out


def cp_upper(k,n,alpha):
    if n<1 or not 0<=k<=n or not 0<alpha<1:
        raise ValueError("Invalid source-risk observation")
    return float(1 if k==n else beta.ppf(1-alpha,k+1,n-k))


def audit_path_gate(rows):
    pos=[r for r in rows if r["world"]=="LINK"]
    neg=[r for r in rows if r["world"]=="NO_LINK"]
    if len(pos)!=120 or len(neg)!=120 or len({r["section"] for r in pos})!=60:
        raise RuntimeError("Original real Thebe 60 section/120+120 case mismatch")
    if sum(r["candidate_available"] for r in pos)!=75 or sum(r["correct_top1_pair"] for r in pos)!=45:
        raise RuntimeError("Original fixed source/candidate/ranker V46 not reproduced")
    alpha=(1-CONFIDENCE)/(2*len(CAL_THRESHOLDS))
    sweep=[]
    for tau in CAL_THRESHOLDS:
        accepted=[r for r in rows if r["accept_score"] is not None and r["accept_score"]>=tau]
        accepted_good=sum(r["world"]=="LINK" and r["correct_decoded_path"] for r in accepted)
        bad_link={r["section"] for r in accepted
                  if r["world"]=="LINK" and not r["correct_decoded_path"]}
        bad_no={r["section"] for r in accepted if r["world"]=="NO_LINK"}
        upper_link=cp_upper(len(bad_link),60,alpha)
        upper_no=cp_upper(len(bad_no),60,alpha)
        sweep.append({
            "threshold":tau,"accepted_successful_paths":int(accepted_good),
            "accepted_wrong_LINK_source_sections":len(bad_link),
            "accepted_NO_LINK_source_sections":len(bad_no),
            "simultaneous_CP_upper_wrong_final_path":upper_link,
            "simultaneous_CP_upper_synthetic_NOLINK":upper_no,
            "conditional_joint_source_95pct_CP10_admissible":max(upper_link,upper_no)<=RISK,
        })
    unique=sorted({r["accept_score"] for r in rows if r["accept_score"] is not None},reverse=True)
    best_zero=best_emp=0
    for tau in unique:
        accepted=[r for r in rows if r["accept_score"] is not None and r["accept_score"]>=tau]
        correct=sum(r["world"]=="LINK" and r["correct_decoded_path"] for r in accepted)
        bad_link={r["section"] for r in accepted if r["world"]=="LINK" and not r["correct_decoded_path"]}
        bad_no={r["section"] for r in accepted if r["world"]=="NO_LINK"}
        if not bad_link and not bad_no:
            best_zero=max(best_zero,correct)
        if len(bad_link)<=6 and len(bad_no)<=6:
            best_emp=max(best_emp,correct)
    eligible=[x for x in sweep if x["conditional_joint_source_95pct_CP10_admissible"]]
    return {
        "status":"V51_BLOCK_OOF_PATH_SUCCESS_GATE_POSTHOC_THEBE_CAL_TEST7_UNOPENED",
        "correct_top1_pairs":sum(r["correct_top1_pair"] for r in pos),
        "correct_hidden_paths_from_top1":sum(r["correct_decoded_path"] for r in pos),
        "best_accepted_correct_paths_conditional_CP95_10pct":max(
            [x["accepted_successful_paths"] for x in eligible],default=0),
        "best_accepted_correct_paths_at_zero_observed_wrong_sections_posthoc":best_zero,
        "best_accepted_correct_paths_both_empirical_errors_le10pct_posthoc":best_emp,
        "fixed_thresholds":list(CAL_THRESHOLDS),
        "risk_error_wrong_LINK_includes_wrong_pair_OR_wrong_decoded_path":True,
        "source_sections_not_independent_geological_volumes":True,
        "no_natural_fault_identity":True,"TEST7_unopened":True,
        "predeclared_threshold_sweep":sweep,
    }
