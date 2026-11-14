"""V40: posthoc real Thebe candidate pair ranking instead of pair classification.

Same frozen V33 ANZA, same V34 predicted SBPP pairs, exact 1:1 matched
training label budget. No new TEST7, no probability calibration claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FROZEN_ANZA_SHA="ba0c3b8060fd0e3cb1c15f6c1598b94a977677c2b075663b5414207df382c25e"
FROZEN_THRESHOLD=.65
EXPECTED_TRAIN_AVAILABLE=116
EXPECTED_CAL_AVAILABLE=65
C=.3


def fit_same_budget_pair_ranker(features,labels):
    """Use exactly the original consecutive (true, hard-negative) training pairs."""
    f=np.asarray(features,np.float64)
    y=np.asarray(labels,int)
    if f.ndim!=2 or f.shape[1]!=12 or len(f)<16 or len(f)%2 or not np.isfinite(f).all():
        raise ValueError("Requires finite >=8 matched 12-feature pairs")
    if not (np.array_equal(y[::2],np.ones(len(y)//2,int)) and
            np.array_equal(y[1::2],np.zeros(len(y)//2,int))):
        raise ValueError("Expected consecutive positive/negative original matched pairs")
    scale=StandardScaler().fit(f)
    z=scale.transform(f)
    delta=z[::2]-z[1::2]
    symmetric=np.concatenate([delta,-delta],axis=0)
    targets=np.r_[np.ones(len(delta),int),np.zeros(len(delta),int)]
    model=LogisticRegression(fit_intercept=False,C=C,max_iter=1400,
                             solver="lbfgs",random_state=20263000)
    model.fit(symmetric,targets)
    if model.coef_.shape!=(1,12):
        raise AssertionError("Unexpected linear pair ranker")
    return scale,model


def pair_rank_scores(scale,ranker,scene):
    rows=scene["proposals"]
    if not rows:return np.empty(0,float)
    x=np.asarray([p["features"] for p in rows],np.float64)
    if x.shape!=(len(rows),12) or not np.isfinite(x).all():
        raise ValueError("Malformed prediction-derived pair features")
    # These are ranking scores; sigmoid output is NOT calibrated probability.
    return expit(scale.transform(x)@ranker.coef_.ravel())


def scored_cases(scenes,logistic,scale,ranker):
    from scripts.tver_v30_p0_context import score_logistic
    rows=[]
    for scene in scenes:
        scores={
            "matched_classification":score_logistic(logistic,scene),
            "paired_difference_ranking":pair_rank_scores(scale,ranker,scene),
        }
        for name,z in scores.items():
            if len(z)!=len(scene["proposals"]):
                raise RuntimeError("Scoring changed candidate length")
            i=int(np.argmax(z)) if len(z) else None
            pair=scene["proposals"][i] if i is not None else None
            rows.append({
                "section":int(scene["section"]),"gap_id":str(scene["id"]),
                "world":scene["kind"],"method":name,
                "candidate_count":len(scene["proposals"]),
                "true_candidate_available":int(scene["has_true_proposal"]),
                "top1_correct":int(scene["kind"]=="LINK" and pair is not None and pair["y"]==1),
                "top1_score":float(z[i]) if i is not None else None,
            })
    return rows


def evaluate_paired(rows,replicates=10000,seed=20261114):
    pos={m:{} for m in ("matched_classification","paired_difference_ranking")}
    for r in rows:
        if r["world"]!="LINK":continue
        key=(int(r["section"]),r["gap_id"])
        if key in pos[r["method"]]:raise ValueError("Duplicate source gap")
        pos[r["method"]][key]=int(r["top1_correct"])
    a=pos["matched_classification"]
    b=pos["paired_difference_ranking"]
    if len(a)!=120 or set(a)!=set(b) or len({s for s,_ in a})!=60:
        raise ValueError("Methods must cover identical 60 sections / 120 positive gaps")
    available=sum(int(r["true_candidate_available"]) for r in rows
                  if r["method"]=="matched_classification" and r["world"]=="LINK")
    if available!=EXPECTED_CAL_AVAILABLE:
        raise ValueError("Original V34 65/120 candidate recall not reproduced")
    sections=sorted({s for s,_ in a})
    diffs={s:sum(b[k]-a[k] for k in a if k[0]==s) for s in sections}
    rng=random.Random(seed)
    deltas=[]
    for _ in range(replicates):
        g=sum(diffs[sections[rng.randrange(len(sections))]] for __ in sections)
        deltas.append(g/120)
    deltas.sort()
    only_old=sum(a[k] and not b[k] for k in a)
    only_new=sum(b[k] and not a[k] for k in a)
    return {
        "n_controlled_positive_gaps":120,
        "source_section_count":60,
        "true_candidate_available":available,
        "matched_logistic_top1":sum(a.values()),
        "matched_pairwise_ranker_top1":sum(b.values()),
        "only_classifier_correct":only_old,
        "only_pair_ranker_correct":only_new,
        "paired_ranker_minus_classifier_top1_fraction":(only_new-only_old)/120,
        "section_bootstrap_95_descriptive_interval":[
            deltas[int(.025*replicates)],
            deltas[min(replicates-1,int(.975*replicates))]],
        "bootstrap_source_sections_not_independent_geological_volumes":True,
        "bootstrap_replicates":replicates,"bootstrap_seed":seed,
    }


def execute(memmap_root,frontends_dir,out):
    import torch
    from scripts.tver_v34_anza_p0_relation_cal import _state_report
    from scripts.tver_v33_thebe_train_segmenters import read_memmap
    from scripts.tver_v33_thebe_relation_cal import (
        _candidate_scenes,_load_model,TRAIN_SOURCE_IDS,RELATION_CAL_SOURCE_IDS)
    from scripts.tver_v30_p0_context import paired_training_examples,fit_matched_logistic

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    receipt=_state_report(frontends_dir)
    if receipt["frontends"]["anza_v1"]["SHA256"]!=FROZEN_ANZA_SHA:
        raise RuntimeError("Original frozen ANZA weights changed")
    segmenter=_load_model(frontends_dir,"anza_v1",receipt)
    tx=read_memmap(memmap_root,"train_seismic_float32.npy")
    ty=read_memmap(memmap_root,"train_fault_bool.npy")
    vx=read_memmap(memmap_root,"val_seismic_float32.npy")
    vy=read_memmap(memmap_root,"val_fault_bool.npy")
    train,_=_candidate_scenes(tx,ty,TRAIN_SOURCE_IDS,
                              receipt["normalization"],segmenter,FROZEN_THRESHOLD)
    available=sum(s["has_true_proposal"] for s in train if s["kind"]=="LINK")
    if available!=EXPECTED_TRAIN_AVAILABLE:
        raise RuntimeError("V34 TRAIN candidate count not reproduced: "+str(available))
    _,labels,features,passport=paired_training_examples(train,FROZEN_THRESHOLD)
    baseline=fit_matched_logistic(features,labels)
    scaler,ranker=fit_same_budget_pair_ranker(features,labels)
    # REL-CAL was previously viewed in V34; do NOT open TEST7 here.
    cal,_=_candidate_scenes(vx,vy,RELATION_CAL_SOURCE_IDS,
                            receipt["normalization"],segmenter,FROZEN_THRESHOLD)
    rows=scored_cases(cal,baseline,scaler,ranker)
    outcome=evaluate_paired(rows)
    if outcome["matched_logistic_top1"]!=36:
        raise RuntimeError("Original V34 matched Logistic 36/120 not reproduced")
    report={
        "status":"V40_REAL_THEBE_PAIRED_RANKING_EXPLORATORY_COMPLETE_TEST7_UNOPENED",
        "original_ANZA_SHA256":FROZEN_ANZA_SHA,
        "V33_original_receipt_SHA256":hashlib.sha256(
          (Path(frontends_dir)/"v33_THEBE_frontends_topology_TRAIN_CAL_frozen.json").read_bytes()).hexdigest(),
        "threshold":FROZEN_THRESHOLD,
        "original_train_true_candidate_gaps":available,
        "same_original_matched_TRAIN_pairs_and_labels":True,
        "training_pair_receipt":passport,
        "new_ranking_objective_C":C,
        "new_ranking_score_not_calibrated_probability":True,
        "VAL1_40_99_already_inspected":True,
        "Thebe_TEST7_unopened":True,
        "natural_geological_negative_relation_truth_unavailable":True,
        "no_changed_P0_weights_or_architecture":True,
        "metrics":outcome,
    }
    with (out/"V40_CASE_ROWS.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/"V40_RESULT.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"status":report["status"],"metrics":outcome},indent=2),flush=True)
    return report


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mmap-root",type=Path,required=True)
    ap.add_argument("--frontends-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    execute(a.mmap_root,a.frontends_dir,a.out_dir)
