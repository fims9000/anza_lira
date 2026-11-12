#!/usr/bin/env python3
"""V21. Real CRACKS F3 backgrounds, geometry-matched synthetic endpoint decoys.

Source expert annotation provides a positive true endpoint and connected-component
decoy target; its hidden path does NOT enter the policy scorer. The same real
seismic pixels remain available. Synthetic annotation-identity labels only.
No original ANZA/LIRA checkpoints, fault-instance labels or full E2E claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from scripts.cracks_real_oracle_gap_audit import COLORS, paths_from_mask, sections
from scripts.cracks_real_neighbor_path_v17 import gray, gradient, rgb

V19_CSV_SHA256="c9a0b0a31f93c9731c57dca6536af87227bc22fc96c0192afd80c0c576198482"
GAPS=(36,48,24)
SHIFTS=(0,-6,6,-12,12)
MIN_VISIBLE=15
MAX_LOG_RATIO=math.log(1.15)
MAX_COS_DELTA=.06
MAX_TAN_DELTA=.10
MIN_DECOY_SEP=5.
LAMBDA_IMAGE=.10
N_BOOT=10000
BOOT_SEED=20261112
CAL_MAX_NULL_FAR=.10
METHODS=("nearest","heading","geometry","central","adjacent","far")

def unit(v):
    v=np.asarray(v,float)
    return v/max(float(np.linalg.norm(v)),1e-12)

def cos_heading(v,w):
    return float(np.dot(unit(v),unit(w)))

def source_gap(source,ix,gap,shift):
    n=len(source)
    if n<gap+2*MIN_VISIBLE:
        return None
    start=int(np.clip(ix-gap//2+shift,MIN_VISIBLE,n-gap-MIN_VISIBLE))
    stop=start+gap
    if start<MIN_VISIBLE or n-stop<MIN_VISIBLE:
        return None
    e0=np.asarray(source[start-1],float)
    t0=e0-np.asarray(source[start-7],float)
    real=np.asarray(source[stop],float)
    tr=np.asarray(source[stop+6],float)-real
    if np.linalg.norm(t0)<1e-8 or np.linalg.norm(real-e0)<1e-8:
        return None
    return e0,t0,real,tr,(start,stop)

def rank_decoys(other,e0,t0,real,tr):
    """Construct a difficult scenario using true known annotation *only here*."""
    d_true=float(np.linalg.norm(real-e0))
    c_true=cos_heading(t0,real-e0)
    k_true=abs(cos_heading(t0,tr))
    matched=[]
    for j in range(MIN_VISIBLE,len(other)-MIN_VISIBLE):
        target=np.asarray(other[j],float)
        td=np.asarray(other[j+6],float)-np.asarray(other[j-6],float)
        d_fake=float(np.linalg.norm(target-e0))
        if d_fake<1e-7 or np.linalg.norm(target-real)<MIN_DECOY_SEP:
            continue
        logratio=abs(math.log(d_fake/d_true))
        dc=abs(cos_heading(t0,target-e0)-c_true)
        dk=abs(abs(cos_heading(t0,td))-k_true)
        if logratio>MAX_LOG_RATIO or dc>MAX_COS_DELTA or dk>MAX_TAN_DELTA:
            continue
        loss=logratio/MAX_LOG_RATIO+dc/MAX_COS_DELTA+dk/MAX_TAN_DELTA
        matched.append((loss,j,target,td,{
            "log_distance_ratio":float(logratio),
            "cosine_difference":float(dc),
            "tangent_difference":float(dk)
        }))
    return sorted(matched,key=lambda rec:(rec[0],rec[1]))

def construct_one(source,other,source_ix):
    for gap in GAPS:
        for shift in SHIFTS:
            g=source_gap(source,source_ix,gap,shift)
            if g is None:continue
            e0,t0,real,tr,span=g
            matching=rank_decoys(other,e0,t0,real,tr)
            if not matching:continue
            first=matching[0]
            second=next((v for v in matching[1:]
                if np.linalg.norm(v[2]-first[2])>=MIN_DECOY_SEP),None)
            return {
                "e0":e0,"t0":t0,"true":real,"true_tan":tr,
                "false_1":first[2],"false_tan_1":first[3],
                "false_match":first[4],"false_j":first[1],
                "false_2":second[2] if second else None,
                "false_tan_2":second[3] if second else None,
                "false_2_j":second[1] if second else None,
                "gap":gap,"shift":shift,"span":span
            }
    return None

def scores_for_candidates(e0,t0,targets,tangents,images,gap):
    """Pure scoring accepts observable features, never labels or a true path."""
    results=[]
    for target,tgt_tangent in zip(targets,tangents):
        dvec=target-e0
        d=float(np.linalg.norm(dvec))
        heading=cos_heading(t0,dvec)
        k=abs(cos_heading(t0,tgt_tangent))
        geom=(d/(gap+1)-1.)**2 + .6*(1-heading)+.2*(1-k)
        t=np.linspace(.08,.92,36)
        line=e0[:,None]*(1-t)[None,:]+target[:,None]*t[None,:]
        image_support=[float(ndi.map_coordinates(im,line,order=1,mode="nearest").mean())
                       for im in images]
        results.append({
            "length":d,"heading":heading,"tangent_compat":k,
            "geometry":float(geom),"image_support":image_support
        })
    return results

def costs(features,method):
    if method=="nearest":return [f["length"] for f in features]
    if method=="heading":return [-f["heading"] for f in features]
    if method=="geometry":return [f["geometry"] for f in features]
    if method=="central":
        return [f["geometry"]-LAMBDA_IMAGE*f["image_support"][0] for f in features]
    if method=="adjacent":
        return [f["geometry"]-LAMBDA_IMAGE*float(
            np.mean(f["image_support"][1:3])) for f in features]
    if method=="far":
        return [f["geometry"]-LAMBDA_IMAGE*float(
            np.mean(f["image_support"][3:5])) for f in features]
    raise ValueError(method)

def flip_hash(sid,a,b,direction,gap,kind):
    return hashlib.sha256(
        f"{sid}|{a}|{b}|{direction}|{gap}|{kind}".encode()).digest()[0]&1

def build_case(sid,a,b,direction,source_idx,decoy_idx,scene,imgs,kind):
    """Build positive or abstention scenario; class identity absent from scorer."""
    e0=scene["e0"];t0=scene["t0"]
    if kind=="positive":
        targets=[scene["true"],scene["false_1"]]
        tangents=[scene["true_tan"],scene["false_tan_1"]]
        correct=0
    elif kind=="no_link":
        if scene["false_2"] is None:
            return None
        targets=[scene["false_1"],scene["false_2"]]
        tangents=[scene["false_tan_1"],scene["false_tan_2"]]
        correct=-1
    else:raise ValueError(kind)
    flip=flip_hash(sid,a,b,direction,scene["gap"],kind)
    if flip:
        targets.reverse();tangents.reverse()
        if correct>=0:correct=1-correct
    feat=scores_for_candidates(e0,t0,targets,tangents,imgs,scene["gap"])
    out={
        "section":sid,"expert_pair_a":a,"expert_pair_b":b,
        "direction":direction,"kind":kind,"source_component":source_idx,
        "decoy_component":decoy_idx,"gap":scene["gap"],"shift":scene["shift"],
        "source_endpoint":json.dumps(e0.tolist()),
        "positive_true_endpoint":json.dumps(scene["true"].tolist()),
        "candidate_0":json.dumps(targets[0].tolist()),
        "candidate_1":json.dumps(targets[1].tolist()),
        "correct_index":correct,
        "matched_log_distance_ratio":scene["false_match"]["log_distance_ratio"],
        "matched_cos_difference":scene["false_match"]["cosine_difference"],
        "matched_tan_difference":scene["false_match"]["tangent_difference"],
        "candidate_features_json":json.dumps(feat,separators=(",",":")),
    }
    for method in METHODS:
        c=costs(feat,method)
        best=int(np.argmin(c))
        out[f"{method}_chosen"]=best
        out[f"{method}_best_cost"]=float(c[best])
        out[f"{method}_correct"]=int(best==correct) if correct>=0 else -1
    return out

def fit_cal_threshold(rows,method):
    calpos=[r for r in rows if r["section"]%2==0 and r["kind"]=="positive"]
    calneg=[r for r in rows if r["section"]%2==0 and r["kind"]=="no_link"]
    if not calneg:
        return {"threshold":None,"N_pos":len(calpos),"N_null":0,"status":"NO_CAL_NULL"}
    costkey=method+"_best_cost"
    values=sorted({float(r[costkey]) for r in calpos+calneg})
    thresholds=[min(values)-1e-9]+values
    choices=[]
    for th in thresholds:
        negaccept=sum(float(r[costkey])<=th for r in calneg)
        if negaccept/len(calneg)>CAL_MAX_NULL_FAR+1e-12:
            continue
        correct=sum(float(r[costkey])<=th and int(r[method+"_correct"])==1
                    for r in calpos)
        wrong=sum(float(r[costkey])<=th and int(r[method+"_correct"])==0
                  for r in calpos)
        choices.append((correct,-wrong,-negaccept,-float(th),th,wrong,negaccept))
    if not choices:raise AssertionError("No valid CAL threshold")
    _,_,_,_,th,wrong,negaccept=max(choices)
    return {"threshold":float(th),"N_pos":len(calpos),"N_null":len(calneg),
            "CAL_correct_accept":int(max(choices)[0]),
            "CAL_wrong_accept":int(wrong),
            "CAL_null_accept":int(negaccept),
            "status":"CALIBRATED" if len(calneg)>=10 else "SMALL_CAL_NULL"}

def section_bootstrap(rows,a,b,reps=N_BOOT):
    groups={}
    for r in rows:
        groups.setdefault(r["section"],[]).append(
            int(r[a+"_correct"])-int(r[b+"_correct"]))
    if len(groups)<2:
        return None
    ids=sorted(groups)
    sums=np.array([sum(groups[s]) for s in ids],float)
    sizes=np.array([len(groups[s]) for s in ids],float)
    rng=np.random.default_rng(BOOT_SEED)
    ix=rng.integers(0,len(ids),size=(reps,len(ids)))
    boot=sums[ix].sum(axis=1)/sizes[ix].sum(axis=1)
    return {"delta_accuracy":float(sums.sum()/sizes.sum()),
            "ci95":list(map(float,np.quantile(boot,[.025,.975]))),
            "N_sections":len(ids)}

def experiment(images:Path,annotations:Path,v19:Path,out:Path,cases_csv:Path):
    source_sha=hashlib.sha256(v19.read_bytes()).hexdigest()
    if source_sha!=V19_CSV_SHA256:
        raise RuntimeError("STOP: frozen V19 cases SHA256 mismatch: "+source_sha)
    with v19.open(newline="",encoding="utf8") as f:
        candidates=[r for r in csv.DictReader(f)
                    if float(r["closest_interior_distance_px"])<=20.]
    rows=[];cnt=Counter();source_counter=Counter()
    with zipfile.ZipFile(images) as zi,zipfile.ZipFile(annotations) as za:
        expert=sections(za,expert=True)
        ims=sections(zi)
        all_ids=sorted(ims)
        imgcache={};gradcache={};pathcache={}
        def getimg(sid):
            if sid not in imgcache:imgcache[sid]=gray(zi,ims[sid])
            return imgcache[sid]
        def getgrad(sid):
            if sid not in gradcache:gradcache[sid]=gradient(getimg(sid))
            return gradcache[sid]
        def getpaths(sid):
            if sid not in pathcache:
                lab=rgb(za,expert[sid])
                blue=np.all(lab==np.asarray(COLORS["blue"],np.uint8),axis=2)
                green=np.all(lab==np.asarray(COLORS["green"],np.uint8),axis=2)
                clean=blue&~ndi.binary_dilation(green,iterations=5)
                pathcache[sid]=sorted(
                    (p for p in paths_from_mask(clean) if len(p)>=36),
                    key=lambda p:(-len(p),p[0][0],p[0][1]))
            return pathcache[sid]
        for r in candidates:
            sid=int(r["section"])
            if sid-1 not in ims or sid+1 not in ims:
                cnt["missing_adjacent"]+=1;continue
            far1=all_ids[(all_ids.index(sid)+107)%len(all_ids)]
            far2=all_ids[(all_ids.index(sid)+211)%len(all_ids)]
            if min(abs(far1-sid),abs(far2-sid))<20:
                cnt["far_not_distant"]+=1;continue
            grads=[getgrad(k) for k in (sid,sid-1,sid+1,far1,far2)]
            if any(g.shape!=grads[0].shape for g in grads[1:]):
                cnt["image_dimension_mismatch"]+=1;continue
            paths=getpaths(sid)
            a=int(r["path_id_a"]);b=int(r["path_id_b"])
            if max(a,b)>=len(paths):raise RuntimeError("V19 path IDs mismatch")
            ia=int(r["interior_index_a"]);ib=int(r["interior_index_b"])
            for direction,(si,di,sidx) in enumerate(((a,b,ia),(b,a,ib))):
                cnt["attempted_directed_pairs"]+=1
                sc=construct_one(paths[si],paths[di],sidx)
                if sc is None:
                    cnt["geometry_match_unavailable"]+=1;continue
                cnt["matched_positive"]+=1
                source_counter[sid]+=1
                pos=build_case(sid,a,b,direction,si,di,sc,grads,"positive")
                rows.append(pos)
                if sc["false_2"] is not None:
                    rows.append(build_case(sid,a,b,direction,si,di,sc,grads,"no_link"))
                    cnt["constructed_no_link"]+=1
                else:
                    cnt["no_second_matched_decoy"]+=1
    cases_csv.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with cases_csv.open("w",newline="",encoding="utf8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    else:
        cases_csv.write_text("",encoding="utf8")
    positives=[r for r in rows if r["kind"]=="positive"]
    nulls=[r for r in rows if r["kind"]=="no_link"]
    testpos=[r for r in positives if int(r["section"])%2==1]
    testnull=[r for r in nulls if int(r["section"])%2==1]
    thresholds={m:fit_cal_threshold(rows,m) for m in METHODS}
    def branch_summary(sub):
        return {"N":len(sub),"sections":len({r["section"] for r in sub}),
                "method_accuracy":{m:{
                    "correct":sum(int(r[m+"_correct"]) for r in sub),
                    "N":len(sub),
                    "rate":float(np.mean([int(r[m+"_correct"]) for r in sub]))
                         if sub else None} for m in METHODS},
                "paired_section_bootstrap":{
                    "adjacent_minus_geometry":section_bootstrap(sub,"adjacent","geometry"),
                    "adjacent_minus_far":section_bootstrap(sub,"adjacent","far"),
                    "central_minus_geometry":section_bootstrap(sub,"central","geometry")
                } if sub else {}}
    def risk_summary(method):
        t=thresholds[method]["threshold"]
        if t is None:return {"status":"NO_CAL_THRESHOLD"}
        key=method+"_best_cost"
        accepted=[r for r in testpos if float(r[key])<=t]
        rejected=[r for r in testpos if float(r[key])>t]
        negaccepted=[r for r in testnull if float(r[key])<=t]
        return {
            "threshold":t,
            "N_test_pos":len(testpos),"N_test_null":len(testnull),
            "pos_correct_accept":sum(int(r[method+"_correct"]) for r in accepted),
            "pos_wrong_accept":sum(1-int(r[method+"_correct"]) for r in accepted),
            "pos_abstained":len(rejected),
            "null_false_accept":len(negaccepted),
            "null_abstained":len(testnull)-len(negaccepted),
            "status":"DESCRIPTIVE" if
                len(testpos)>=10 and len(testnull)>=10 else "LOW_TEST_SAMPLE",
        }
    report={
        "protocol":"CRACKS_V21_REAL_GEOMETRY_MATCHED_SYNTHETIC_ENDPOINTS",
        "source":"CRACKS v2 DOI 10.5281/zenodo.13926822",
        "source_v19_csv_sha256":source_sha,
        "status":"EXECUTED_EXPLORATORY" if positives else "NO_MATCHED_CANDIDATES",
        "geometry_thresholds":{"ratio_lower":.85,"ratio_upper":1.15,
            "cosine_delta":MAX_COS_DELTA,"tangent_delta":MAX_TAN_DELTA,
            "target_separation_min_px":MIN_DECOY_SEP},
        "candidate_construction":"source positive same expert component; matched decoy(s) other expert component, using source true endpoint in dataset construction but not model scoring",
        "gap_order":list(GAPS),"anchor_shift_order":list(SHIFTS),
        "n_source_v19_pairs":len(candidates),
        "counts":dict(cnt),
        "matched_sections":len(source_counter),
        "all_positive":branch_summary(positives),
        "even_cal_positive":branch_summary([r for r in positives if int(r["section"])%2==0]),
        "odd_test_positive":branch_summary(testpos),
        "no_link_positive_labels":"none of provided candidates is from same annotated connected source trace",
        "no_link_N":len(nulls),
        "no_link_even_CAL":sum(int(r["section"])%2==0 for r in nulls),
        "no_link_odd_TEST":len(testnull),
        "calibration_empirical_null_fp_ceiling":CAL_MAX_NULL_FAR,
        "calibration":thresholds,
        "odd_test_reject_results":{m:risk_summary(m) for m in METHODS},
        "original_seismic_images_unchanged":True,
        "full_ANZA_LIRA_E2E":False,
        "valid_natural_geological_no_link_labels":False,
        "out_of_volume_test":False,
        "claim":"Only exploratory synthetic annotation-identity pairing on real F3 seismic pixels. Matched cases adversarially selected from expert masks.",
        "cases_csv_sha256":hashlib.sha256(cases_csv.read_bytes()).hexdigest()
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"status":report["status"],
        "n_source_v19_pairs":len(candidates),"counts":dict(cnt),
        "matched_sections":len(source_counter),
        "no_link_even_CAL":report["no_link_even_CAL"],
        "no_link_odd_TEST":report["no_link_odd_TEST"]},ensure_ascii=False,indent=2))
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--images",required=True)
    parser.add_argument("--annotations",required=True)
    parser.add_argument("--v19-cases",required=True)
    parser.add_argument("--out",default="results/cracks_real_controlled_v1/geommatch_v21.json")
    parser.add_argument("--cases",default="results/cracks_real_controlled_v1/geommatch_v21_cases.csv")
    a=parser.parse_args()
    experiment(Path(a.images),Path(a.annotations),Path(a.v19_cases),
               Path(a.out),Path(a.cases))

if __name__=="__main__":
    main()
