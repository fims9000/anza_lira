#!/usr/bin/env python3
"""V18 CRACKS real F3: longer expert-label gaps, frozen geometry and 2.5D controls.

Scientific boundary: known expert surviving endpoints and visible expert tangents.
Not a learned segmentation or end-to-end ANZA/LIRA evaluation. Hidden expert mask
is accessed only AFTER generating every candidate and its image-based score.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from scripts.cracks_real_oracle_gap_audit import paths_from_mask, sections, COLORS, split
from scripts.cracks_real_neighbor_path_v17 import (
    gray, rgb, gradient, normalized_corr, candidate_scores, choose
)

GAPS = (24, 36, 48)
OFFSETS = (-4., -2., 0., 2., 4.)
METHODS = ("straight", "hermite", "central", "adjacent",
           "central_adjacent", "central_shuffled")
BOOT_PAIRS = (("adjacent","straight"),("adjacent","hermite"),
              ("central_adjacent","hermite"),
              ("central_adjacent","central_shuffled"),
              ("adjacent","central"))
RNG_SEED = 20261112

def make_curves(path: list[tuple[int,int]], gap: int):
    n = len(path)
    if n < gap + 30:
        return None
    start = (n-gap)//2
    stop = start+gap
    if start < 15 or n-stop < 15:
        return None
    e0 = np.asarray(path[start-1], dtype=float)
    e1 = np.asarray(path[stop], dtype=float)
    v0 = (e0 - np.asarray(path[start-7],float))/6.
    v1 = (np.asarray(path[stop+6],float) - e1)/6.
    t = np.linspace(0.,1.,max(40,3*(gap+1)))
    scale = gap+1
    c = (2*t**3-3*t**2+1)[None,:]*e0[:,None]
    c += (t**3-2*t**2+t)[None,:]*(scale*v0)[:,None]
    c += (-2*t**3+3*t**2)[None,:]*e1[:,None]
    c += (t**3-t**2)[None,:]*(scale*v1)[:,None]
    straight = e0[:,None]*(1-t)[None,:] + e1[:,None]*t[None,:]
    chord=e1-e0
    size=max(float(np.linalg.norm(chord)),1.e-6)
    normal=np.asarray([-chord[1],chord[0]])/size
    curves=[c + (amp*np.sin(np.pi*t))[None,:]*normal[:,None]
            for amp in OFFSETS]
    # No hidden points enter candidates; only e0/e1 and visible tangents.
    hidden=np.asarray(path[start:stop],dtype=int)
    return straight, curves, hidden, (start,stop), (e0,e1)

def score_geometry(p:np.ndarray, distance:np.ndarray):
    vals=ndi.map_coordinates(distance,p,order=1,mode="nearest")
    return float(vals.mean()),float((vals <= 2.).mean())

def extract_cases(sid:int,label:np.ndarray):
    blue=np.all(label==np.asarray(COLORS["blue"],np.uint8),axis=2)
    green=np.all(label==np.asarray(COLORS["green"],np.uint8),axis=2)
    clean=blue & ~ndi.binary_dilation(green,iterations=5)
    pathlist=sorted(paths_from_mask(clean),key=lambda p:-len(p))
    for trace_id,path in enumerate(pathlist[:3]):
        for gap in GAPS:
            data=make_curves(path,gap)
            if data is None:
                continue
            straight,curves,hidden,span,ends=data
            # The line was selected solely by observed length and a fixed mid-gap.
            # Expert pixels in the gap are retained only for post-selection scoring.
            yield trace_id,gap,straight,curves,hidden,span,ends

def clustered_interval(rows:list[dict],method_a:str,method_b:str,reps=10000):
    if not rows:
        return None
    groups={}
    for r in rows:
        groups.setdefault(int(r["section"]),[]).append(
            float(r[method_a+"_mean_distance"]) -
            float(r[method_b+"_mean_distance"]))
    sections=sorted(groups)
    if len(sections)<2:
        return None
    counts=np.array([len(groups[s]) for s in sections],int)
    sums=np.array([sum(groups[s]) for s in sections],float)
    rng=np.random.default_rng(RNG_SEED)
    ix=rng.integers(0,len(sections),size=(reps,len(sections)))
    means=sums[ix].sum(axis=1)/counts[ix].sum(axis=1)
    return {"delta":float(sums.sum()/counts.sum()),
            "ci95":list(map(float,np.quantile(means,[0.025,0.975]))),
            "N_sections":len(sections)}

def evaluate(images:Path,annotations:Path,out:Path,cases:Path,max_sections:int):
    rows=[]
    alignment=[]
    ctr=Counter()
    with zipfile.ZipFile(images) as zi, zipfile.ZipFile(annotations) as za:
        ims=sections(zi)
        expert=sections(za,expert=True)
        ids=sorted(set(ims)&set(expert))[:max_sections]
        all_ids=sorted(ims)
        imgcache={}
        gradcache={}
        def get_image(sid):
            if sid not in imgcache:
                imgcache[sid]=gray(zi,ims[sid])
            return imgcache[sid]
        def get_grad(sid):
            if sid not in gradcache:
                gradcache[sid]=gradient(get_image(sid))
            return gradcache[sid]
        for sid in ids:
            label=rgb(za,expert[sid])
            if sid not in ims or sid-1 not in ims or sid+1 not in ims:
                ctr["no_immediate_neighbors"]+=1
                continue
            im=get_image(sid)
            if label.shape[:2]!=im.shape or any(
                    get_image(k).shape!=im.shape for k in (sid-1,sid+1)):
                ctr["shape_mismatch"]+=1
                continue
            far1=all_ids[(all_ids.index(sid)+107)%len(all_ids)]
            far2=all_ids[(all_ids.index(sid)+211)%len(all_ids)]
            if min(abs(far1-sid),abs(far2-sid))<20:
                ctr["sham_not_far"]+=1
                continue
            alignment.append((sid,
                .5*(normalized_corr(im,get_image(sid-1)) +
                    normalized_corr(im,get_image(sid+1))),
                .5*(normalized_corr(im,get_image(far1)) +
                    normalized_corr(im,get_image(far2)))))
            grads=[get_grad(k) for k in (sid,sid-1,sid+1,far1,far2)]
            for trace_id,gap,straight,curves,hidden,span,ends in extract_cases(sid,label):
                scores=candidate_scores(curves,grads)
                # Expert hidden path ONLY used after all model inputs are built.
                gt=np.zeros(im.shape,dtype=bool)
                gt[hidden[:,0],hidden[:,1]]=True
                dmap=ndi.distance_transform_edt(~gt)
                selected={
                    "straight":straight,
                    "hermite":curves[OFFSETS.index(0.)]
                }
                for method in METHODS[2:]:
                    selected[method]=curves[choose(scores,method)]
                baseline_distance,baseline_precision=score_geometry(straight,dmap)
                split_name,block=split(sid)
                row={"section":sid,"block":block,"split":split_name,
                     "trace_id":trace_id,"requested_gap":gap,
                     "hidden_samples":len(hidden),
                     "start_stop":json.dumps(span),
                     "visible_endpoint_0":json.dumps(ends[0].tolist()),
                     "visible_endpoint_1":json.dumps(ends[1].tolist()),
                     "straight_is_nontrivial":int(baseline_precision<0.90),
                     "gradient_scores_sha256":hashlib.sha256(
                         np.round(scores,8).tobytes()).hexdigest()}
                for method,p in selected.items():
                    dist,pr=score_geometry(p,dmap)
                    row[method+"_mean_distance"]=dist
                    row[method+"_precision_tol2"]=pr
                rows.append(row)
            ctr["sections_scanned"]+=1
    cases.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with cases.open("w",newline="",encoding="utf8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        cases.write_text("",encoding="utf8")
    def summarize(rr):
        return {
            "N":len(rr),
            "N_sections":len({r["section"] for r in rr}),
            "N_nontrivial_straight":sum(r["straight_is_nontrivial"] for r in rr),
            "by_method":{m:{
                "mean_distance":float(np.mean([r[m+"_mean_distance"] for r in rr])),
                "mean_precision_tol2":float(np.mean(
                     [r[m+"_precision_tol2"] for r in rr]))
            } for m in METHODS} if rr else {},
            "paired_section_bootstrap":{
                a+"_minus_"+b:clustered_interval(rr,a,b)
                for a,b in BOOT_PAIRS} if rr else {}
        }
    hard=[r for r in rows if r["straight_is_nontrivial"]]
    aligned=np.asarray(alignment,dtype=float)
    result={
        "protocol":"CRACKS_V18_LONG_GAP_CURVATURE_REAL_SEISMIC_EXPLORATORY",
        "source":"CRACKS v2 doi:10.5281/zenodo.13926822",
        "source_commit_protocol":"docs/research/tver2026/V18_REAL_CURVED_GAPS_FROZEN_PROTOCOL_20261112.md",
        "status":"EXECUTED_DESCRIPTIVE" if rows else "NO_ELIGIBLE_CASES",
        "full_ANZA_LIRA_e2e":False,
        "known_positive_endpoint_pair":True,
        "valid_negative_instance_pairs":False,
        "original_seismic_pixels_not_masked":True,
        "read_hidden_expert_mask_for_inference":False,
        "predefined_gap_samples":list(GAPS),
        "predefined_normal_offsets":list(OFFSETS),
        "predefined_nontrivial_criterion":"straight_precision_tol2 < 0.90",
        "data_sections_with_expert_masks":len(ids),
        "counts":dict(ctr),
        "image_alignment":{
            "N":len(alignment),
            "mean_near_corr":float(np.nanmean(aligned[:,1])) if len(aligned) else None,
            "mean_far_corr":float(np.nanmean(aligned[:,2])) if len(aligned) else None
        },
        "all":summarize(rows),
        "nontrivial_straight":summarize(hard),
        "by_gap":{str(g):summarize([r for r in rows if r["requested_gap"]==g])
                  for g in GAPS},
        "claim_boundary":"Oracle-visible endpoints, expert masks select cases and score results, same F3 cube; no real E2E or negative link risk.",
        "cases_sha256":hashlib.sha256(cases.read_bytes()).hexdigest()
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({k:v for k,v in result.items()
                      if k not in ("all","by_gap","nontrivial_straight")},
                     ensure_ascii=False,indent=2))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--images",required=True)
    p.add_argument("--annotations",required=True)
    p.add_argument("--out",default="results/cracks_real_controlled_v1/longgap_v18.json")
    p.add_argument("--cases",default="results/cracks_real_controlled_v1/longgap_v18_cases.csv")
    p.add_argument("--max-sections",type=int,default=400)
    a=p.parse_args()
    evaluate(Path(a.images),Path(a.annotations),Path(a.out),Path(a.cases),a.max_sections)

if __name__=="__main__":
    main()
