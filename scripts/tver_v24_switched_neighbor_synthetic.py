"""Tver V24: challenge V23 with single-neighbor-matched NO_LINK counterfactuals.

A NO_LINK scene is a hybrid: z=-1 is EXACTLY one link world's neighbor,
z=+1 is EXACTLY the opposite link world's neighbor. Each individual near
slice is positive-looking; only consistent *both-slice source identity*
can reject. Frozen V23 TRAIN/CAL, no new task labels or TEST retuning.

Synthetic controlled graph-link validation, NOT historic ANZA-LIRA E2E.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.tver_v23_25d_counterfactual_synthetic import (
    H,W,BOOT,NO_LINK,
    _rng,_hashflip,_background,_render,_bridge_features,
    visible_features,_materialize,_fit_learned,_score_table,_calibrate,
    _predict,_metric,_paired_base_bootstrap,_semantic_pair_audit,
)

BASE=202673010
N=160
PRIMARY_V23_CAL_LOCK={
    "continuity_25d":(0.3902535167944451,3.03955731083e-06),
    "learned_25d":(0.3735086895653693,3.66766451828e-05),
}
METHODS=("continuity_25d","learned_25d","consensus_v24",
         "single_neighbor_minus","single_neighbor_plus",
         "central","learned_2d")


def render_hard_v24(index, *, base_seed=BASE, total_groups=N):
    if not 0<=index<total_groups:raise ValueError("Unknown independent hard-negative base")
    base=base_seed+index
    rng=_rng(base)
    A=float(rng.uniform(8.5,12.))
    d=float(rng.uniform(2.8,4.8))
    width=float(rng.uniform(.65,.95))
    amps=rng.uniform(.62,1.05,size=4)
    amps*=float(rng.uniform(.6,1.4))
    noise=_background(base,ood=True)
    flip=_hashflip(base)
    center=_render(noise[1].copy(),A=A,
                   shift_left=(0.,0.),shift_right=(0.,0.),
                   width=width,amplitudes=amps,clutter=.13,
                   seed=base,z=0,ood=True)
    views=[]
    for w in (0,1):
        adjacent=[]
        for z,n in ((-1,noise[0]),(1,noise[2])):
            left=(z*d,-z*d)
            right=(z*d,-z*d) if w==0 else (-z*d,z*d)
            # Frozen nuisance motion is shared ACROSS three worlds;
            # opposite hybrid cases therefore have exact individual
            # slice class-conditional matches, not merely close scores.
            reg=float((_rng(base+30000+z).uniform(-2.4,2.4)))
            image=_render(n.copy(),A=A,
                          shift_left=(left[0]+reg,left[1]+reg),
                          shift_right=(right[0]+reg,right[1]+reg),
                          width=width,amplitudes=amps,clutter=.28,
                          seed=base,z=z,ood=True)
            adjacent.append(image)
        views.append(np.asarray(adjacent))
    swapped=bool(base&1)
    negative=np.stack((
        views[1 if swapped else 0][0],
        views[0 if swapped else 1][1]))
    all_views=(views[0],views[1],negative)
    far=np.stack((
        noise[0]+.08*_rng(base+606).normal(size=(H,W)),
        noise[2]+.08*_rng(base+607).normal(size=(H,W))))
    candidates=((56.,32.+A),(56.,32.-A))
    if flip:candidates=candidates[::-1]
    rows=[]
    for world in (0,1,2):
        rows.append({
            "base":base,"family":"hybrid_adjacent_switch",
            "world":world,"label":(world^flip if world!=2 else 2),
            "flip":flip,"A":A,"source":(8.,32.+A),
            "candidates":candidates,
            "central":center.copy(),
            "adjacent":all_views[world].copy(),
            "far":far.copy(),
        })
    # This is a *mathematically stronger* NO_LINK than V23:
    # one-slice marginals have an EXACT positive-world counterpart.
    assert np.array_equal(rows[0]["central"],rows[2]["central"])
    assert np.array_equal(rows[1]["central"],rows[2]["central"])
    assert (np.array_equal(rows[2]["adjacent"][0],
                           rows[int(swapped)]["adjacent"][0]))
    assert (np.array_equal(rows[2]["adjacent"][1],
                           rows[1-int(swapped)]["adjacent"][1]))
    assert {r["label"] for r in rows}=={0,1,2}
    return rows


def _visible_scores(scene):
    observed=visible_features(scene)
    # Features 3..6 are right-candidate support from each actual near
    # image; 7..8 are left-source support. Both are computed without Y.
    _,feature=_bridge_features(scene["adjacent"],
                              scene["A"],scene["flip"])
    right=np.asarray([[feature[3],feature[4]],
                      [feature[5],feature[6]]],float)
    source=np.asarray([feature[7],feature[8]],float)
    common=np.minimum(right,source[:,None])
    consensus=np.min(common,axis=0)
    return {
        "continuity_25d":observed["continuity_25d"],
        "consensus_v24":consensus,
        "single_neighbor_minus":common[0].copy(),
        "single_neighbor_plus":common[1].copy(),
        "center_features":observed["center_features"],
        "all_features":observed["all_features"],
        "central":observed["central"],
    }


def _make_cal(rows,models):
    scored=[]
    for row in rows:
        # Original CAL worlds/observations, no V24 labels.
        old={
            "base":row["base"],"label":row["label"],
            "A":None,"flip":None,
            "scores":row["scores"],
            "center_features":row["center_features"],
            "all_features":row["all_features"],
        }
        # New consensus must use the actual original V23 CAL rendered
        # observed pixels. Recreate it deterministically from the
        # unchanged V23 CAL base seed and within-pair world assignment.
        from scripts.tver_v23_25d_counterfactual_synthetic import (
            render_triplet,
        )
        base_index=int(row["base"])-202633010
        selected=next(w for w in render_triplet("cal",base_index)
                      if w["label"]==row["label"])
        v=_visible_scores(selected)
        scored.append({
            "label":row["label"],"scores":{
                "continuity_25d":row["scores"]["continuity_25d"],
                "consensus_v24":v["consensus_v24"],
                "single_neighbor_minus":v["single_neighbor_minus"],
                "single_neighbor_plus":v["single_neighbor_plus"],
                "central":row["scores"]["central"],
            },
            "center_features":row["center_features"],
            "all_features":row["all_features"],
        })
    return scored


def _model_scores(rows,models):
    d={m:[] for m in METHODS}
    for row in rows:
        v=row["visible"]
        for name in ("continuity_25d","consensus_v24",
                     "single_neighbor_minus","single_neighbor_plus",
                     "central"):
            d[name].append(np.asarray(v[name],float))
        for name,key in (("learned_25d","all_features"),
                         ("learned_2d","center_features")):
            model=models[name][0]
            probabilities=model.predict_proba(v[key][None,:])[0]
            mapping=dict(zip(model.classes_,probabilities))
            d[name].append(np.asarray([
                mapping.get(0,0.)-mapping.get(2,0.),
                mapping.get(1,0.)-mapping.get(2,0.)]))
    return {k:np.asarray(v,float) for k,v in d.items()}


def execute(out):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    # Fixed V23 TRAIN/CAL labels, no V24 TEST data opened for fit.
    train=_materialize("train")
    cal=_materialize("cal")
    models=_fit_learned(train)
    orig=_score_table(cal,models)
    original_y=np.asarray([r["label"] for r in cal],int)
    locked={
        name:_calibrate(original_y,orig[name])
        for name in PRIMARY_V23_CAL_LOCK}
    for name,known in PRIMARY_V23_CAL_LOCK.items():
        got=locked[name]
        if (abs(got["accept_threshold"]-known[0])>1e-6 or
            abs(got["margin_threshold"]-known[1])>1e-6):
            raise RuntimeError("STOP: original V23 TRAIN/CAL risk contract drift")
    cal_visible=_make_cal(cal,models)
    new_scores={
        name:np.asarray([w["scores"][name] for w in cal_visible])
        for name in ("consensus_v24","single_neighbor_minus",
                     "single_neighbor_plus")}
    for name,values in new_scores.items():
        locked[name]=_calibrate(original_y,values)
    for name in ("central","learned_2d"):
        locked[name]=_calibrate(original_y,orig[name])
    original_lock={
        "status":"V24_ALL_THRESHOLDS_FROZEN_BEFORE_HYBRID_TEST_OPENED",
        "original_V23_base_scoring_unchanged":True,
        "already_seen_V23_test_not_used_to_fit_or_calibrate":True,
        "new_hybrid_base_seed":BASE,"n_hybrid_base_groups":N,
        "methods":METHODS,"thresholds":locked,
        "new_consensus_calibration_only_original_V23_CAL":True,
        "original_matched_25D_HGB_retrained_only_original_V23_TRAIN":True,
        "no_historical_ANZA_LIRA_weights":True,
        "no_hybrid_negative_labels_used_to_choose_thresholds":True,
    }
    freeze=out/"v24_before_test_freeze.json"
    freeze.write_text(json.dumps(original_lock,indent=2)+"\n")
    digest=hashlib.sha256(freeze.read_bytes()).hexdigest()
    # Only now generate V24 new hidden three-world TEST labels/images.
    test=[]
    for i in range(N):
        for scene in render_hard_v24(i):
            feature=_visible_scores(scene)
            test.append({
                "base":scene["base"],"label":scene["label"],
                "world":scene["world"],"family":scene["family"],
                "central_sha":hashlib.sha256(
                    scene["central"].tobytes()).hexdigest(),
                "near_minus_sha":hashlib.sha256(
                    scene["adjacent"][0].tobytes()).hexdigest(),
                "near_plus_sha":hashlib.sha256(
                    scene["adjacent"][1].tobytes()).hexdigest(),
                "visible":feature,
                "center_features":feature["center_features"],
                "all_features":feature["all_features"],
            })
    grouped=defaultdict(list)
    for r in test:grouped[r["base"]].append(r)
    for base,rows in grouped.items():
        assert len(rows)==3
        assert {r["label"] for r in rows}=={0,1,2}
        assert len({r["central_sha"] for r in rows})==1
        n=next(r for r in rows if r["label"]==2)
        positives=[r for r in rows if r["label"]!=2]
        assert any(n["near_minus_sha"]==p["near_minus_sha"] for p in positives)
        assert any(n["near_plus_sha"]==p["near_plus_sha"] for p in positives)
        # Crucially the left/right correspondence from the matched
        # minus and plus neighbor belongs to DIFFERENT true worlds.
        assert (next(p["label"] for p in positives if p["near_minus_sha"]==n["near_minus_sha"])
                != next(p["label"] for p in positives if p["near_plus_sha"]==n["near_plus_sha"]))
    scores=_model_scores(test,models)
    pred={name:_predict(scores[name],locked[name]) for name in METHODS}
    stats={name:_metric(test,pred[name]) for name in METHODS}
    contrast={
        candidate:_paired_base_bootstrap(
            test,pred["consensus_v24"],pred[candidate])
        for candidate in (
            "continuity_25d","learned_25d","central",
            "single_neighbor_minus","single_neighbor_plus")}
    admission=bool(
        stats["consensus_v24"]["false_bridge_rate"]<=.02 and
        stats["consensus_v24"]["wrong_branch_rate"]<=.01 and
        stats["consensus_v24"]["positive_correct_accepted_rate"]>=.15 and
        contrast["learned_25d"]["base_group_cluster_CI95"][0]>0)
    output_rows=[]
    for ix,r in enumerate(test):
        item={
            "base":r["base"],"world":r["world"],"label":r["label"],
            "central_sha256":r["central_sha"],
            "near_minus_sha256":r["near_minus_sha"],
            "near_plus_sha256":r["near_plus_sha"],
        }
        for name in METHODS:
            item[name+"_pred"]=int(pred[name][ix])
            item[name+"_score0"]=float(scores[name][ix,0])
            item[name+"_score1"]=float(scores[name][ix,1])
        output_rows.append(item)
    cases=out/"v24_hard_negative_case_scores.csv"
    with cases.open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(output_rows[0]))
        writer.writeheader();writer.writerows(output_rows)
    report={
        "protocol":"TV_V24_SWITCHING_NEIGHBOR_MARGINAL_MATCHED_NOLINK",
        "status":"EXECUTED_NEW_SYNTHETIC_HARD_NEGATIVE",
        "V24_TEST_opened_only_after_CAL_freeze":True,
        "source_2D_pixels_identical_for_all_worlds":True,
        "NO_LINK_each_neighbor_exact_positive_world_match":True,
        "NO_LINK_marginal_link_evidence_misleading_by_construction":True,
        "N_base_groups":N,"N_worlds":len(test),
        "V24_CAL_lock_SHA256":digest,
        "method_results":stats,
        "consensus_minus_baselines_group_bootstrap":contrast,
        "hard_negative_OOD_criterion_met":admission,
        "cases_sha256":hashlib.sha256(cases.read_bytes()).hexdigest(),
        "true_geological_fault_instancing_not_tested":True,
        "historical_anza_lira_checkpoint_not_loaded":True,
    }
    (out/"v24_results.json").write_text(
        json.dumps(report,indent=2)+"\n")
    print(json.dumps({
        "status":report["status"],
        "frozen_before_test":report["V24_TEST_opened_only_after_CAL_freeze"],
        "pass":admission,
        "methods":{k:{a:v[a] for a in (
            "correct_accepted","wrong_branch_accepted","false_accepted",
            "all_3way_accuracy")} for k,v in stats.items()},
        "consensus_minus_25d_HGB":contrast["learned_25d"],
    },indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    execute(args.out)
