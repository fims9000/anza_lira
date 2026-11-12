#!/usr/bin/env python3
"""V20 CRACKS: controlled annotation-level endpoint competition on REAL F3 images.

The source and two candidate coordinates (plus local *visible* tangents) are
provided from expert annotations. True/decoy labels are hidden from scorers.
No physical-fault identity and no full ANZA/LIRA E2E are asserted.
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

from scripts.cracks_real_oracle_gap_audit import COLORS, paths_from_mask, sections
from scripts.cracks_real_neighbor_path_v17 import gray, gradient, rgb

EXPECTED_V19_CASES_SHA256 = "c9a0b0a31f93c9731c57dca6536af87227bc22fc96c0192afd80c0c576198482"
MIN_VISIBLE = 15
GAP = 12
LAMBDA_IMAGE = .10
METHODS=("nearest","heading","geometry","geometry_central",
         "geometry_adjacent","geometry_far")

def unit(v):
    v=np.asarray(v,float)
    return v/max(float(np.linalg.norm(v)),1.e-9)

def candidate_features(source,src_tan,targets,tgt_tans,gradient_images):
    """Inference features: only visible coordinates, visible tangents, seismic."""
    source=np.asarray(source,float)
    forward=unit(src_tan)
    out=[]
    for target,other_tan in zip(targets,tgt_tans):
        target=np.asarray(target,float)
        delta=target-source
        d=float(np.linalg.norm(delta))
        chord=unit(delta)
        c=float(np.dot(forward,chord))
        k=float(abs(np.dot(forward,unit(other_tan))))
        geom=float((d/(GAP+1)-1.)**2 + .6*(1.-c) + .2*(1.-k))
        t=np.linspace(.08,.92,36)
        points=source[:,None]*(1-t)[None,:] + target[:,None]*t[None,:]
        support=np.asarray([
            float(ndi.map_coordinates(g,points,order=1,mode="nearest").mean())
            for g in gradient_images],float)
        out.append({"distance":d,"heading":c,"tangent_compat":k,
                    "geometry":geom,"gradient_support":support})
    return out

def decide(features,method):
    """Neither label nor original expert centerline is accepted by this API."""
    if method=="nearest":
        scores=[f["distance"] for f in features]
    elif method=="heading":
        scores=[-f["heading"] for f in features]
    elif method=="geometry":
        scores=[f["geometry"] for f in features]
    elif method=="geometry_central":
        scores=[f["geometry"] - LAMBDA_IMAGE*f["gradient_support"][0]
                for f in features]
    elif method=="geometry_adjacent":
        scores=[f["geometry"]- LAMBDA_IMAGE*float(
            np.mean(f["gradient_support"][1:3])) for f in features]
    elif method=="geometry_far":
        scores=[f["geometry"]- LAMBDA_IMAGE*float(
            np.mean(f["gradient_support"][3:5])) for f in features]
    else:
        raise ValueError(method)
    return int(np.argmin(scores))

def conditional_case(path_source,path_decoy,index_source,index_decoy):
    """Return true+decoy candidates, visible tangents; never the hidden path."""
    n=len(path_source)
    start=int(np.clip(index_source - GAP//2,
                      MIN_VISIBLE,n-GAP-MIN_VISIBLE))
    stop=start+GAP
    if (start<MIN_VISIBLE or n-stop<MIN_VISIBLE or
            index_decoy<MIN_VISIBLE or
            len(path_decoy)-index_decoy<=MIN_VISIBLE):
        return None
    e0=np.asarray(path_source[start-1],float)
    true=np.asarray(path_source[stop],float)
    false=np.asarray(path_decoy[index_decoy],float)
    if np.linalg.norm(true-false)<1.e-6:
        return None
    src_tan=e0-np.asarray(path_source[start-7],float)
    true_tan=(np.asarray(path_source[stop+6],float) -
              np.asarray(path_source[stop],float))
    decoy_tan=(np.asarray(path_decoy[index_decoy+6],float) -
               np.asarray(path_decoy[index_decoy-6],float))
    return e0,src_tan,(true,false),(true_tan,decoy_tan),(
        start,stop)

def score_correct(rows,method):
    return sum(r[method+"_correct"] for r in rows)

def grouped_ci(rows,a,b,reps=10000):
    if not rows:
        return None
    groups={}
    for r in rows:
        groups.setdefault(int(r["section"]),[]).append(
            float(r[a+"_correct"]-r[b+"_correct"]))
    ids=sorted(groups)
    if len(ids)<2:
        return None
    counts=np.asarray([len(groups[s]) for s in ids],int)
    sums=np.asarray([sum(groups[s]) for s in ids],float)
    rng=np.random.default_rng(20261112)
    ix=rng.integers(0,len(ids),size=(reps,len(ids)))
    v=sums[ix].sum(axis=1)/counts[ix].sum(axis=1)
    return {
        "difference_accuracy":float(sums.sum()/counts.sum()),
        "ci95":list(map(float,np.quantile(v,[.025,.975]))),
        "N_sections":len(ids)
    }

def experiment(images:Path,annotations:Path,v19_cases:Path,out:Path,cases:Path):
    original=v19_cases.read_bytes()
    digest=hashlib.sha256(original).hexdigest()
    if digest!=EXPECTED_V19_CASES_SHA256:
        raise RuntimeError("STOP: V19 preselected candidate source hash changed: "+digest)
    with v19_cases.open(encoding="utf8",newline="") as f:
        selected=[r for r in csv.DictReader(f)
                  if float(r["closest_interior_distance_px"])<=20.]
    rows=[]
    counters=Counter()
    with zipfile.ZipFile(images) as zi,zipfile.ZipFile(annotations) as za:
        ims=sections(zi)
        expert=sections(za,expert=True)
        imgcache={}
        gradcache={}
        pathcache={}
        all_ids=sorted(ims)
        def getimg(s):
            if s not in imgcache:imgcache[s]=gray(zi,ims[s])
            return imgcache[s]
        def getgrad(s):
            if s not in gradcache:gradcache[s]=gradient(getimg(s))
            return gradcache[s]
        def getpaths(sid):
            if sid not in pathcache:
                lab=rgb(za,expert[sid])
                blue=np.all(lab==np.asarray(COLORS["blue"],np.uint8),axis=2)
                green=np.all(lab==np.asarray(COLORS["green"],np.uint8),axis=2)
                clean=blue & ~ndi.binary_dilation(green,iterations=5)
                pathcache[sid]=sorted(
                    (p for p in paths_from_mask(clean) if len(p)>=36),
                    key=lambda p:(-len(p),p[0][0],p[0][1]))
            return pathcache[sid]
        for r in selected:
            sid=int(r["section"])
            if sid-1 not in ims or sid+1 not in ims:
                counters["missing_adjacent"]+=1
                continue
            near_id=all_ids.index(sid)
            far1=all_ids[(near_id+107)%len(all_ids)]
            far2=all_ids[(near_id+211)%len(all_ids)]
            if min(abs(far1-sid),abs(far2-sid))<20:
                counters["sham_not_far"]+=1
                continue
            paths=getpaths(sid)
            a=int(r["path_id_a"]);b=int(r["path_id_b"])
            ia=int(r["interior_index_a"]);ib=int(r["interior_index_b"])
            if a>=len(paths) or b>=len(paths):
                raise RuntimeError("V19 path IDs no longer align")
            grads=[getgrad(x) for x in (sid,sid-1,sid+1,far1,far2)]
            if any(g.shape!=grads[0].shape for g in grads[1:]):
                counters["shape_mismatch"]+=1
                continue
            for direction,(si,di,ix,jx) in enumerate(
                    ((a,b,ia,ib),(b,a,ib,ia))):
                candidate=conditional_case(paths[si],paths[di],ix,jx)
                if candidate is None:
                    counters["insufficient_visible_support"]+=1
                    continue
                e0,tangent,targets,tans,span=candidate
                digest_choice=hashlib.sha256(
                    f"{sid}|{a}|{b}|{direction}".encode()).digest()
                flip=digest_choice[0] & 1
                if flip:
                    targets=(targets[1],targets[0])
                    tans=(tans[1],tans[0])
                correct_index=int(flip)
                feats=candidate_features(e0,tangent,targets,tans,grads)
                # Class labels deliberately kept outside scoring API.
                true_distance=feats[correct_index]["distance"]
                false_distance=feats[1-correct_index]["distance"]
                group=int(sid)//5
                row={
                    "section":sid,
                    "block":group,
                    "annotated_component_a":a,
                    "annotated_component_b":b,
                    "direction":direction,
                    "source_component":si,
                    "correct_target_component":si,
                    "decoy_component":di,
                    "visible_source_endpoint":json.dumps(e0.tolist()),
                    "gap_span_in_expert_trace":json.dumps(span),
                    "candidate_0_endpoint":json.dumps(
                        np.asarray(targets[0]).tolist()),
                    "candidate_1_endpoint":json.dumps(
                        np.asarray(targets[1]).tolist()),
                    "correct_candidate_index":correct_index,
                    "true_dist_px":true_distance,
                    "decoy_dist_px":false_distance,
                    "distance_ambiguous":int(
                        false_distance<=1.2*true_distance),
                    "feature_sha256":hashlib.sha256(
                        np.round(np.array([
                        [f["distance"],f["heading"],f["tangent_compat"],
                         f["geometry"],*f["gradient_support"]] for f in feats
                        ]),8).tobytes()).hexdigest()
                }
                for method in METHODS:
                    chosen=decide(feats,method)
                    row[method+"_selected"]=chosen
                    row[method+"_correct"]=int(chosen==correct_index)
                rows.append(row)
                counters["directed_cases"]+=1
    if not rows:
        raise RuntimeError("No eligible cases from V19 frozen source")
    cases.parent.mkdir(parents=True,exist_ok=True)
    with cases.open("w",encoding="utf8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    hard=[r for r in rows if r["distance_ambiguous"]]
    def summarize(sub):
        return {
            "N":len(sub),"N_sections":len(set(r["section"] for r in sub)),
            "accuracy":{m:{
                "correct":score_correct(sub,m),
                "N":len(sub),
                "rate":score_correct(sub,m)/len(sub) if sub else None}
                for m in METHODS},
            "paired_section_bootstrap":{
                f"{a}_minus_{b}":grouped_ci(sub,a,b)
                for a,b in (("geometry_adjacent","geometry"),
                            ("geometry_adjacent","geometry_far"),
                            ("geometry_central","geometry"),
                            ("geometry","nearest"))}
        }
    report={
        "protocol":"CRACKS_V20_CONTROLLED_ANNOTATION_PAIRING_REAL_BACKGROUND",
        "source":"CRACKS v2, doi:10.5281/zenodo.13926822",
        "status":"EXECUTED_EXPLORATORY",
        "pinned_v19_csv_sha256":digest,
        "eligible_annotated_pairs_v19":len(selected),
        "N_directed_cases":len(rows),
        "N_unique_original_annotated_pairs":len(
            {(r["section"],r["annotated_component_a"],
              r["annotated_component_b"]) for r in rows}),
        "N_sections":len(set(r["section"] for r in rows)),
        "counts":dict(counters),
        "all":summarize(rows),
        "distance_ambiguous_subset":summarize(hard),
        "sham_control":"two far sections of same F3 cube",
        "labels":"synthetic annotation-component identity only, not geology",
        "uses_expert_endpoint_oracle":True,
        "uses_hidden_expert_centerline_in_scoring":False,
        "valid_geological_negative_pairs":False,
        "permits_NO_LINK":False,
        "full_ANZA_LIRA_e2e":False,
        "case_csv_sha256":hashlib.sha256(cases.read_bytes()).hexdigest(),
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({
        k:v for k,v in report.items()
        if k not in ("all","distance_ambiguous_subset")
    },ensure_ascii=False,indent=2))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--images",required=True)
    p.add_argument("--annotations",required=True)
    p.add_argument("--v19-cases",required=True)
    p.add_argument("--out",default="results/cracks_real_controlled_v1/pairing_v20.json")
    p.add_argument("--cases",default="results/cracks_real_controlled_v1/pairing_v20_cases.csv")
    x=p.parse_args()
    experiment(Path(x.images),Path(x.annotations),
               Path(x.v19_cases),Path(x.out),Path(x.cases))

if __name__=="__main__":
    main()
