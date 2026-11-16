"""Predeclared out-of-renderer-family geometry stress test for V50 vs V52.

TRAIN uses the historical V50 sine-ridge generator (seeds 0..99).
Evaluation uses separately constructed sigmoid and multi-bend path families
(seeds 5000..5079) with increased curvature and distractor ridges.  This
is NOT natural geology or external seismic-field validation. The same frozen
TRAIN checkpoint and candidate endpoints are passed to both decoders.
No parameter search or threshold fitting against the evaluation outcomes.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from scripts.tver_v52_joint_path import decode_candidate_v52

REGIMES = ("sigmoid_turn", "multibend_interference")
OOD_SEEDS = tuple(range(5000, 5080))
TRAIN_SEEDS = tuple(range(100))


def ood_scene(seed, regime):
    if regime not in REGIMES:
        raise ValueError("Unknown frozen OOD renderer")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:96, :96]
    amplitude = float(rng.uniform(7., 12.)) * (-1 if rng.random() < .5 else 1)
    shift = float(rng.uniform(-2., 2.))
    tilt = float(rng.uniform(-.10, .10))
    if regime == "sigmoid_turn":
        center = float(rng.uniform(45., 51.))
        steepness = float(rng.uniform(2.1, 4.2))
        line = lambda y: 48 + shift + tilt*(y-48) + amplitude*np.tanh((y-center)/steepness)
        roi = (39, 57, 25, 72)
    else:
        phase = float(rng.uniform(-.8, .8))
        period = float(rng.uniform(9., 13.))
        line = lambda y: 48 + shift + tilt*(y-48) + amplitude*np.sin((y-40)/period*np.pi + phase)
        roi = (35, 61, 25, 72)
    ridge = np.exp(-.5*((xx-line(yy))/1.4)**2)
    # Competing structure crosses the gap near the real line and has weaker support.
    rival = np.exp(-.5*((xx-(line(yy)+6*np.cos(yy/11+seed)))/1.7)**2)
    img_noise = rng.normal(0., .065, (96,96))
    gray = np.clip(.17+.53*ridge+.22*rival+img_noise, 0, 1).astype(np.float32)
    p = np.clip(.02+.84*ridge+.23*rival+rng.normal(0., .06,(96,96)),0,1).astype(np.float32)
    y0,y1,x0,x1=roi
    p[y0:y1,x0:x1]=0
    pred_start=(30.,float(line(30.)+rng.normal(0,2.0)))
    pred_end=(66.,float(line(66.)+rng.normal(0,2.0)))
    # Truth is only a training/evaluation label; decoder never receives it.
    hidden = [(int(y),int(round(line(y)))) for y in range(y0+2,y1-2)]
    return {
        "kind":"LINK", "id":str(seed), "section":int(seed),
        "gray":gray,"visible_prob":p,"rectangle":roi,"origin":(0,0),
        "hidden_centerline":hidden,
        "proposals":[{"source":pred_start,"target":pred_end,"geometry":-.5,"y":1}],
    }


def metrics(scene, route):
    y0,y1,x0,x1=scene["rectangle"]
    true=np.zeros((96,96),bool)
    predicted=np.zeros((96,96),bool)
    for y,x in scene["hidden_centerline"]:
        if y0 <= y < y1 and x0 <= x < x1:
            true[y,x]=True
    if route is not None:
        for y,x in route:
            if y0 <= y < y1 and x0 <= x < x1:
                predicted[y,x]=True
    if not true.any() or not predicted.any():
        return {"precision_2px":0.,"recall_2px":0.,"successful":False,
                "symmetric_mean_distance_px":None}
    near_true=ndi.binary_dilation(true,iterations=2)
    near_pred=ndi.binary_dilation(predicted,iterations=2)
    precision=float(np.mean(near_true[predicted]))
    recall=float(np.mean(near_pred[true]))
    dist_true=ndi.distance_transform_edt(~true)
    dist_pred=ndi.distance_transform_edt(~predicted)
    symmetric=(float(np.mean(dist_true[predicted]))+float(np.mean(dist_pred[true])))/2
    return {"precision_2px":precision,"recall_2px":recall,
            "successful":precision >= .8 and recall >= .8,
            "symmetric_mean_distance_px":symmetric}


def run(out_dir):
    from scripts.tver_v50_synthetic_sanity import scene as train_scene
    from scripts.tver_v50_conditional_path import fit_path_head, decode_candidate
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    model,train_receipt=fit_path_head([train_scene(i) for i in TRAIN_SEEDS],out/"train")
    rows=[]
    for regime in REGIMES:
        for seed in OOD_SEEDS:
            sc=ood_scene(seed,regime)
            pair=sc["proposals"][0]
            v50,_=decode_candidate(model,sc,pair,mode="learned_ribbon")
            v52,meta=decode_candidate_v52(model,sc,pair)
            m50=metrics(sc,v50)
            m52=metrics(sc,v52)
            rows.append({
                "regime":regime,"seed":seed,
                "v50_success":int(m50["successful"]),
                "v52_success":int(m52["successful"]),
                "v52_abstained":int(v52 is None),
                "v50_symmetric_distance_px":m50["symmetric_mean_distance_px"],
                "v52_symmetric_distance_px":m52["symmetric_mean_distance_px"],
                "v50_precision_2px":m50["precision_2px"],
                "v50_recall_2px":m50["recall_2px"],
                "v52_precision_2px":m52["precision_2px"],
                "v52_recall_2px":m52["recall_2px"],
                "v52_decoder_status":meta["status"],
            })
    by_regime={}
    for regime in REGIMES:
        cohort=[r for r in rows if r["regime"]==regime]
        by_regime[regime]={
            "cases":len(cohort),
            "v50_success":sum(r["v50_success"] for r in cohort),
            "v52_success":sum(r["v52_success"] for r in cohort),
            "v52_abstained":sum(r["v52_abstained"] for r in cohort),
            "v52_only_wins":sum(r["v52_success"] and not r["v50_success"] for r in cohort),
            "v50_only_wins":sum(r["v50_success"] and not r["v52_success"] for r in cohort),
        }
    report={
        "status":"V52_SYNTHETIC_OUT_OF_RENDERER_FAMILY_STRESS_NOT_GEOLOGY",
        "train_generator":"original_V50_one_sine_ridge_renderer",
        "evaluation_generators":list(REGIMES),
        "train_seeds":[TRAIN_SEEDS[0],TRAIN_SEEDS[-1]],
        "ood_seeds":[OOD_SEEDS[0],OOD_SEEDS[-1]],
        "n_ood":len(rows),"same_frozen_training_checkpoint":True,
        "candidate_endpoints_synthetically_generated_from_ground_truth":True,
        "no_Thebe_TEST7_access":True,
        "model_training_receipt":train_receipt,
        "by_regime":by_regime,
    }
    with (out/"V52_OOD_SYNTHETIC_CASES.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/"V52_OOD_SYNTHETIC_RESULT.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("V52_OOD_SYNTHETIC_SUMMARY",json.dumps({k:v for k,v in report.items() if k!="model_training_receipt"},indent=2),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    run(a.out_dir)