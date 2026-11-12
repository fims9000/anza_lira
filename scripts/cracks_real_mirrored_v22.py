#!/usr/bin/env python3
"""V22 CRACKS: exact mirror-geometry counterfactual on real F3 image pixels.

The artificial reflected target is NOT an observed candidate fault; this is a
diagnostic of available real-image information after geometric ties are forced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from scripts.cracks_real_oracle_gap_audit import COLORS, paths_from_mask, sections
from scripts.cracks_real_neighbor_path_v17 import gray,gradient,rgb
from scripts.cracks_real_matched_nolink_v21 import source_gap,unit

V21_SOURCE_SHA256="c30b1279e8e114d8c11c656f7f20ebfeacc237ca163a304f19b41a9389b7d237"
V19_SOURCE_SHA256="c9a0b0a31f93c9731c57dca6536af87227bc22fc96c0192afd80c0c576198482"
METHODS=("geometry_tie","central","adjacent","central_adjacent","far","source_similarity")
RNG_SEED=20261112
N_BOOT=10000

def reflect_about_axis(point:np.ndarray,axis_origin:np.ndarray,tangent:np.ndarray):
    """Isometric reflection about the line through origin in tangent direction."""
    v=np.asarray(point,float)-np.asarray(axis_origin,float)
    u=unit(tangent)
    return np.asarray(axis_origin,float)+2*np.dot(v,u)*u-v

def reflect_vector(vector:np.ndarray,tangent:np.ndarray):
    v=np.asarray(vector,float);u=unit(tangent)
    return 2*np.dot(v,u)*u-v

def tied_geometry(e0,tan,target,target_tan,fake,fake_tan):
    """Compare d, heading, abs(tangent compatibility) without knowing labels."""
    a=np.asarray(target,float)-e0
    b=np.asarray(fake,float)-e0
    u=unit(tan)
    return np.asarray([
        abs(np.linalg.norm(a)-np.linalg.norm(b)),
        abs(np.dot(unit(a),u)-np.dot(unit(b),u)),
        abs(abs(np.dot(unit(target_tan),u)) -
            abs(np.dot(unit(fake_tan),u))),
    ],dtype=float)

def image_scores(paths:list[np.ndarray],images:list[np.ndarray],
                 visible_prior:np.ndarray):
    """Scorer only sees both candidate straight paths and seismic gradients."""
    signals=np.asarray([[float(ndi.map_coordinates(
        img,p,order=1,mode="nearest").mean()) for img in images]
        for p in paths],dtype=float)
    central=signals[:,0]
    near=signals[:,1:3].mean(axis=1)
    far=signals[:,3:5].mean(axis=1)
    past= float(ndi.map_coordinates(
        images[0],visible_prior,order=1,mode="nearest").mean())
    return {
        "geometry_tie":np.asarray([0.,0.]),
        "central":central,
        "adjacent":near,
        "central_adjacent":signals[:,:3].mean(axis=1),
        "far":far,
        "source_similarity":-np.abs(central-past),
    },signals,past

def choose(scores:np.ndarray):
    return int(np.argmax(scores))

def group_ci(rows:list[dict],name:str,contrast:str|None=None):
    if len(rows)<2:return None
    grouped=defaultdict(list)
    for r in rows:
        value=float(r[name+"_correct"])-(
            float(r[contrast+"_correct"]) if contrast else .5)
        grouped[int(r["section"])].append(value)
    sections=sorted(grouped)
    if len(sections)<2:return None
    sums=np.asarray([sum(grouped[s]) for s in sections],float)
    sizes=np.asarray([len(grouped[s]) for s in sections],float)
    rng=np.random.default_rng(RNG_SEED)
    ix=rng.integers(0,len(sections),size=(N_BOOT,len(sections)))
    delta=sums[ix].sum(axis=1)/sizes[ix].sum(axis=1)
    return {"delta":float(sums.sum()/sizes.sum()),
        "ci95":list(map(float,np.quantile(delta,[.025,.975]))),
        "N_sections":len(sections)}

def evaluate(images:Path,annotations:Path,v19:Path,v21:Path,
             out:Path,cases:Path):
    h21=hashlib.sha256(v21.read_bytes()).hexdigest()
    h19=hashlib.sha256(v19.read_bytes()).hexdigest()
    if h21!=V21_SOURCE_SHA256:
        raise RuntimeError("STOP: pinned V21 case CSV SHA256 changed: "+h21)
    if h19!=V19_SOURCE_SHA256:
        raise RuntimeError("STOP: pinned V19 case CSV SHA256 changed: "+h19)
    with v19.open(newline="",encoding="utf8") as f:
        v19map={(int(r["section"]),int(r["path_id_a"]),int(r["path_id_b"])):r
                for r in csv.DictReader(f)}
    with v21.open(newline="",encoding="utf8") as f:
        positives=[r for r in csv.DictReader(f) if r["kind"]=="positive"]
    counters=Counter()
    rows=[]
    with zipfile.ZipFile(images) as zi,zipfile.ZipFile(annotations) as za:
        ims=sections(zi);expert=sections(za,expert=True)
        all_ids=sorted(ims)
        imgcache={};gradcache={};pathcache={}
        def get_img(sid):
            if sid not in imgcache:imgcache[sid]=gray(zi,ims[sid])
            return imgcache[sid]
        def get_grad(sid):
            if sid not in gradcache:gradcache[sid]=gradient(get_img(sid))
            return gradcache[sid]
        def paths_for(sid):
            if sid not in pathcache:
                label=rgb(za,expert[sid])
                blue=np.all(label==np.asarray(COLORS["blue"],np.uint8),axis=2)
                green=np.all(label==np.asarray(COLORS["green"],np.uint8),axis=2)
                clean=blue&~ndi.binary_dilation(green,iterations=5)
                pathcache[sid]=sorted(
                    (p for p in paths_from_mask(clean) if len(p)>=36),
                    key=lambda p:(-len(p),p[0][0],p[0][1]))
            return pathcache[sid]
        for r in positives:
            sid=int(r["section"])
            a=int(r["expert_pair_a"]);b=int(r["expert_pair_b"])
            direction=int(r["direction"])
            gap=int(r["gap"]);shift=int(r["shift"])
            key=(sid,a,b)
            if key not in v19map:raise RuntimeError("V21/V19 pairing mismatch")
            source_id=a if direction==0 else b
            ix=int(v19map[key]["interior_index_a" if direction==0
                else "interior_index_b"])
            path=paths_for(sid)[source_id]
            case=source_gap(path,ix,gap,shift)
            if case is None:raise RuntimeError("V21 source gap cannot be reconstructed")
            e0,tan,target,target_tan,span=case
            if not (np.allclose(e0,np.asarray(json.loads(r["source_endpoint"]))) and
                    np.allclose(target,np.asarray(json.loads(r["positive_true_endpoint"])))):
                raise RuntimeError("V21 pinned source/true endpoint mismatch")
            fake=reflect_about_axis(target,e0,tan)
            fake_tan=reflect_vector(target_tan,tan)
            sep=float(np.linalg.norm(target-fake))
            counters["V21_positive_inputs"]+=1
            if sep<5.:
                counters["fake_too_close"]+=1;continue
            image=get_img(sid)
            if not (3<=fake[0]<image.shape[0]-3 and
                    3<=fake[1]<image.shape[1]-3):
                counters["fake_out_of_bounds"]+=1;continue
            if sid-1 not in ims or sid+1 not in ims:
                counters["no_neighbor"]+=1;continue
            far1=all_ids[(all_ids.index(sid)+107)%len(all_ids)]
            far2=all_ids[(all_ids.index(sid)+211)%len(all_ids)]
            imgs=[get_grad(x) for x in (sid,sid-1,sid+1,far1,far2)]
            if any(g.shape!=imgs[0].shape for g in imgs[1:]):
                counters["dimension_mismatch"]+=1;continue
            tie=tied_geometry(e0,tan,target,target_tan,fake,fake_tan)
            if tie.max()>1e-8:
                raise AssertionError("Geometric tie violated: "+str(tie))
            flip=hashlib.sha256(
                f"{sid}|{a}|{b}|{direction}|{gap}|V22".encode()).digest()[0]&1
            endpoints=[target,fake]
            if flip:endpoints.reverse()
            # Exclude 8% at each end; identical sampling on both paths.
            t=np.linspace(.08,.92,48)
            traces=[e0[:,None]*(1-t)[None,:]+p[:,None]*t[None,:]
                    for p in endpoints]
            visible_prior=np.asarray(path[span[0]-12:span[0]],float).T
            scores,signals,prior=image_scores(traces,imgs,visible_prior)
            correct_idx=int(flip)
            row={"section":sid,"expert_pair_a":a,"expert_pair_b":b,
                "direction":direction,"gap":gap,"shift":shift,
                "correct_candidate_index":correct_idx,
                "mirror_target_separation_px":sep,
                "max_geometry_tie_error":float(tie.max()),
                "candidate_0":json.dumps(endpoints[0].tolist()),
                "candidate_1":json.dumps(endpoints[1].tolist()),
                "image_gradients_json":json.dumps(signals.tolist()),
                "visible_prior_central_gradient":prior}
            for m in METHODS:
                pick=choose(scores[m])
                row[m+"_choice"]=pick
                row[m+"_correct"]=int(pick==correct_idx)
                row[m+"_score_difference"]=float(scores[m][0]-scores[m][1])
            rows.append(row)
            counters["eligible_mirrored_pairs"]+=1
    cases.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with cases.open("w",newline="",encoding="utf8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:cases.write_text("",encoding="utf8")
    summary={
        "protocol":"CRACKS_V22_REAL_F3_EXACT_GEOMETRIC_MIRROR_TEST",
        "status":"EXECUTED_EXPLORATORY" if rows else "NO_ELIGIBLE_CASES",
        "pinned_V19_sha256":h19,"pinned_V21_sha256":h21,
        "counts":dict(counters),
        "N_cases":len(rows),
        "N_sections":len({r["section"] for r in rows}),
        "true_fake_geometry_exact_tied":all(
            r["max_geometry_tie_error"]<=1e-8 for r in rows),
        "max_tie_error":max((r["max_geometry_tie_error"] for r in rows),default=None),
        "methods":{m:{
            "correct":sum(r[m+"_correct"] for r in rows),
            "N":len(rows),
            "accuracy":sum(r[m+"_correct"] for r in rows)/len(rows)
                if rows else None,
            "vs_chance_section_bootstrap":group_ci(rows,m),
        } for m in METHODS},
        "adjacent_minus_far_bootstrap":group_ci(rows,"adjacent","far"),
        "central_adjacent_minus_far_bootstrap":
            group_ci(rows,"central_adjacent","far"),
        "case_csv_sha256":hashlib.sha256(cases.read_bytes()).hexdigest(),
        "claim_boundary":"Artificial mirrored endpoint, not a real detected fault candidate; real image pixels but oracle source geometry and annotated true target. No natural LINK/REJECT, no ANZA/LIRA E2E.",
        "no_learning":True,"no_NOLINK_in_this_diagnostic":True,
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("methods",)},
                     indent=2,ensure_ascii=False))
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--images",required=True)
    p.add_argument("--annotations",required=True)
    p.add_argument("--v19-cases",required=True)
    p.add_argument("--v21-cases",required=True)
    p.add_argument("--out",default="results/cracks_real_controlled_v1/mirror_v22.json")
    p.add_argument("--cases",default="results/cracks_real_controlled_v1/mirror_v22_cases.csv")
    a=p.parse_args()
    evaluate(Path(a.images),Path(a.annotations),Path(a.v19_cases),
             Path(a.v21_cases),Path(a.out),Path(a.cases))

if __name__=="__main__":
    main()
