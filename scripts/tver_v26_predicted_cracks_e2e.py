"""V26 newly trained REAL CRACKS image→AZConv/U-Net→SBPP→LIRA pilot.

Not a claim about the original inaccessible ANZA/LIRA checkpoints.
All model input and SBPP/LIRA features come from observed source image and
*model-predicted probability*, with expert-derived rectangles used ONLY to
define controlled missing-segmentation interventions. No expert endpoint,
hidden centerline or link identity is passed into model, SBPP, or scorer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter,defaultdict
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from scripts.cracks_real_oracle_gap_audit import (
    COLORS,sections,paths_from_mask,
)
from scripts.tver_v26_cracks_inventory import split_map
from scripts.tver_v26_train_real_segmenters import (
    restore_model,predict_patch,
)
from anza_tracegraph.ports_v3.branches import extract_branches
from anza_tracegraph.ports_v3.candidates import source_ports
from anza_tracegraph.ports_v3_b.soft_branches import (
    SoftBranch,extract_soft_branches,
)
from anza_tracegraph.ports_v3_b.clustering import cluster_branches
from anza_tracegraph.ports_v3_b.candidates import propose_cluster_candidates


ROI=96
K=12
MAX_SOURCES=4
MAX_HARD=110
MAX_SOFT=70
THRESHOLDS=tuple(np.round(np.linspace(.2,.95,16),3))
REQUIRED_LIRA_POS=4
REQUIRED_LIRA_NEG=4
N_SELECT_PER_SECTION=2


def _img(archive,name):
    with Image.open(io.BytesIO(archive.read(name))) as im:
        return np.asarray(im.convert("RGB"),np.uint8)


def _expert_gaps(rgb,section):
    """A fixed, label-derived intervention SCHEDULE, not source ports.

    The expert path and exact endpoints are used ONLY at scene construction
    and for training/evaluation labels, NEVER inside sbpp_proposals/scorer.
    """
    blue=np.all(rgb==COLORS["blue"],axis=-1)
    green=np.all(rgb==COLORS["green"],axis=-1)
    clean=blue&~ndi.binary_dilation(green,iterations=5)
    paths=sorted(
        (np.asarray(p,dtype=np.int32) for p in paths_from_mask(clean)
         if len(p)>=80),
        key=lambda p:(-len(p),int(p[0,0]),int(p[0,1])))
    gaps=[]
    for trace_id,path in enumerate(paths[:N_SELECT_PER_SECTION]):
        j=len(path)//2
        hidden=path[j-6:j+6]
        if len(hidden)!=12:continue
        ymin=int(hidden[:,0].min()-3)
        ymax=int(hidden[:,0].max()+4)
        xmin=int(hidden[:,1].min()-3)
        xmax=int(hidden[:,1].max()+4)
        h,w=blue.shape
        if ymin<9 or xmin<9 or ymax>=h-9 or xmax>=w-9:continue
        if np.any(green[ymin:ymax,xmin:xmax]):continue
        inside=lambda p:ymin<=p[0]<ymax and xmin<=p[1]<xmax
        first=next((p for p in path[:j][::-1] if not inside(p)),None)
        last=next((p for p in path[j:] if not inside(p)),None)
        if first is None or last is None:continue
        if np.linalg.norm(first-last)<6:continue
        # Interventional scene uses expert to DEFINE a missing-segmentation
        # rectangle. No annotated endpoint or instance is accessible to
        # sbpp_proposals or LIRA feature extraction.
        gaps.append({
            "gap_id":f"{section}:{trace_id}",
            "section":int(section),
            "rectangle":(ymin,ymax,xmin,xmax),
            "true_ends":(tuple(map(float,first)),tuple(map(float,last))),
            "hidden_centerline":[tuple(map(int,x)) for x in hidden],
        })
    return gaps


def _roi(image,rectangle):
    h,w=image.shape[:2]
    y0,y1,x0,x1=rectangle
    cy=(y0+y1)//2;cx=(x0+x1)//2
    oy=int(np.clip(cy-ROI//2,0,h-ROI))
    ox=int(np.clip(cx-ROI//2,0,w-ROI))
    piece=image[oy:oy+ROI,ox:ox+ROI].copy()
    if piece.shape[:2]!=(ROI,ROI):
        raise RuntimeError("STOP ROI source image smaller than 96x96")
    return piece,(oy,ox),(y0-oy,y1-oy,x0-ox,x1-ox)


def _world(prob,rect,ends,origin,world):
    """No hidden labels enter actual scoring, only intervention map.

    Positive world erases missing rectangle from model segmentation.
    Negative world also erases the expected target neighborhood (based
    on intervention oracle) => synthetic candidate-absent NO_LINK.
    """
    if world not in ("LINK","NO_LINK"):
        raise ValueError("Unknown controlled intervention class")
    out=np.asarray(prob,np.float32).copy()
    y0,y1,x0,x1=rect
    out[y0:y1,x0:x1]=0.
    if world=="NO_LINK":
        # This is a controlled absence-of-candidate world, NOT a natural
        # ground-truth geological false link. Entire circle suppressed.
        gy,gx=ends[1]
        py=gy-origin[0];px=gx-origin[1]
        yy,xx=np.ogrid[:ROI,:ROI]
        out[(yy-py)**2+(xx-px)**2<=12**2]=0.
    return out


def _center(rect):
    return np.asarray([(rect[0]+rect[1])/2.,(rect[2]+rect[3])/2.])


def _sample(image,start,dest,n=32):
    ys=np.linspace(start[0],dest[0],n)
    xs=np.linspace(start[1],dest[1],n)
    return ndi.map_coordinates(image,[ys,xs],order=1,mode="nearest")


def _features(source,cand,prob,gray):
    """ONLY predicted source/candidate and observed image, never truth."""
    a=np.asarray(source.point_yx,float)
    b=np.asarray(cand.landing_point_yx,float)
    distance=max(float(np.linalg.norm(b-a)),1e-5)
    q=_sample(prob,a,b)
    gray=ndi.gaussian_filter(gray,1.)
    grad=np.abs(ndi.sobel(gray,axis=1))
    grad/=max(float(np.quantile(grad,.95)),1e-6)
    g=_sample(grad,a,b)
    src=np.asarray(source.tangent_yx,float)
    landing=np.asarray(cand.landing_tangent_yx,float)
    direction=(b-a)/distance
    return np.asarray([
        distance/68.,
        float(cand.geometric_score)/120.,
        float(cand.source_angle)/math.pi,
        float(cand.destination_angle)/math.pi,
        float(source.confidence),
        float(cand.mean_probability),
        float(q.mean()),
        float(np.quantile(q,.2)),
        float(g.mean()),
        float(np.quantile(g,.2)),
        float(np.dot(src,direction)),
        float(np.dot(landing,-direction)),
    ],np.float64)


def sbpp_proposals(visible_prob,gray,rect,hard_threshold):
    """Actual repository SBPP V3-B hard+soft clustering and proposals.

    Inputs have only model probability, original image ROI and
    INTERVENTION RECTANGLE. No source or destination oracle coordinates.
    """
    prob=np.asarray(visible_prob,np.float32)
    if (prob.shape!=(ROI,ROI) or gray.shape!=(ROI,ROI)
        or not np.isfinite(prob).all()
        or np.any((prob<0)|(prob>1))):
        raise ValueError("Wrong observed predicted probability")
    mask=prob>=float(hard_threshold)
    if mask.mean()>.38:
        return {"status":"DEGENERATE_DENSE_PREDICTED_MASK",
                "source_count":0,"hard_branches":0,"proposals":[]}
    if mask.sum()<12:
        return {"status":"NO_PREDICTED_FAULT_PIXELS",
                "source_count":0,"hard_branches":0,"proposals":[]}
    hard=extract_branches(
        mask,prob,gray,tau_micro=.4*hard_threshold)
    if len(hard)>MAX_HARD:
        return {"status":"TOO_MANY_PREDICTED_BRANCHES",
                "source_count":0,"hard_branches":len(hard),"proposals":[]}
    ports=source_ports(hard,prob)
    center=_center(rect)
    selected=[]
    for port in ports:
        point=np.asarray(port.point_yx,float)
        distance=float(np.linalg.norm(point-center))
        if not 3.<=distance<=29.:continue
        toward=(center-point)/max(distance,1e-8)
        if np.dot(toward,np.asarray(port.tangent_yx,float))<.10:continue
        selected.append((distance,-float(port.confidence),port))
    selected.sort(key=lambda row:(row[0],row[1],row[2].branch_id))
    unique={};candidates=[]
    for _,_,port in selected[:MAX_SOURCES]:
        sources=tuple(
            extract_soft_branches(
                prob,gray,mask,port,tau_s=ratio*hard_threshold)
            for ratio in (.4,.6,.8))
        soft=[]
        for bank in sources:
            for branch in bank:
                soft.append(SoftBranch(
                    branch_id=len(soft),points_yx=branch.points_yx,
                    mean_probability=branch.mean_probability,
                    mean_contrast=branch.mean_contrast,
                    orientation_coherence=branch.orientation_coherence,
                    hysteresis_rule=branch.hysteresis_rule))
        soft=sorted(soft,key=lambda b:(-b.length,-b.mean_probability))[:MAX_SOFT]
        clusters=cluster_branches(hard,tuple(soft))
        proposed=propose_cluster_candidates(port,clusters)[:K]
        for item in proposed:
            key=tuple(np.rint([*port.point_yx,*item.landing_point_yx]).astype(int))
            if key in unique:continue
            unique[key]=True
            candidates.append({
                "source":tuple(map(float,port.point_yx)),
                "target":tuple(map(float,item.landing_point_yx)),
                "features":_features(port,item,prob,gray).tolist(),
                "geometry":float(-item.geometric_score),
            })
    candidates=sorted(
        candidates,key=lambda row:(-row["geometry"],row["source"],row["target"]))
    return {"status":"OK" if candidates else "NO_SOURCES_OR_CANDIDATES",
            "source_count":min(MAX_SOURCES,len(selected)),
            "hard_branches":len(hard),
            "proposals":candidates[:48],
            "n_proposals_before_cap":len(candidates)}


def is_true_relation(proposal,ends,origin,tol=8.):
    """EXPERT truth solely for TRAIN labels or TEST metric, NOT scorer."""
    a=np.asarray(proposal["source"])+np.asarray(origin)
    b=np.asarray(proposal["target"])+np.asarray(origin)
    x=np.asarray(ends[0]);y=np.asarray(ends[1])
    return bool(
        (np.linalg.norm(a-x)<=tol and np.linalg.norm(b-y)<=tol)
        or (np.linalg.norm(a-y)<=tol and np.linalg.norm(b-x)<=tol))


def _case_worlds(image,gap,model,receipt):
    raw,origin,rectangle=_roi(image,gap["rectangle"])
    probability=predict_patch(
        model,raw,receipt["train_normalization"]["mean"],
        receipt["train_normalization"]["std"])
    gray=raw.mean(axis=2)
    worlds=[]
    for kind in ("LINK","NO_LINK"):
        candidate_mask=_world(
            probability,rectangle,gap["true_ends"],origin,kind)
        pred=sbpp_proposals(
            candidate_mask,gray,rectangle,
            receipt["selected_threshold"])
        # No oracle use before this line: only now attach scoring labels.
        proposals=pred["proposals"]
        for proposal in proposals:
            proposal["y"]=int(
                kind=="LINK" and is_true_relation(
                    proposal,gap["true_ends"],origin))
        worlds.append({
            "id":gap["gap_id"],"kind":kind,
            "section":gap["section"],
            "source_count":pred["source_count"],
            "branch_count":pred["hard_branches"],
            "status":pred["status"],
            "p_original_visible":float(np.mean(probability)),
            "mask_visible_fraction":float(
                np.mean(candidate_mask>=receipt["selected_threshold"])),
            "has_true_proposal":bool(any(x["y"] for x in proposals)),
            "proposals":proposals,
            "origin":origin,"rectangle":rectangle,
            "true_ends":gap["true_ends"],
            "hidden_centerline":gap["hidden_centerline"],
            "gray":gray,
            "visible_prob":candidate_mask,
        })
    return worlds


def _dataset(zi,za,model,ids,receipt):
    img_paths=sections(zi)
    lab_paths=sections(za,expert=True)
    scenes=[];source_count=0
    for sid in ids:
        source=_img(zi,img_paths[sid]).astype(np.float32)/255.
        expert=_img(za,lab_paths[sid])
        for gap in _expert_gaps(expert,sid):
            for scene in _case_worlds(source,gap,model,receipt):
                scenes.append(scene)
            source_count+=1
        print("phase=V26_REAL_CASES",sid,
              "controlled_positive_gaps",source_count,flush=True)
    return scenes


def _train_relation(train):
    feature=[];labels=[]
    for scene in train:
        for row in scene["proposals"]:
            feature.append(row["features"])
            labels.append(row["y"])
    labels=np.asarray(labels,int)
    count={str(c):int(np.sum(labels==c)) for c in (0,1)}
    if (count["1"]<REQUIRED_LIRA_POS
        or count["0"]<REQUIRED_LIRA_NEG):
        return None,{"status":"STOP_INSUFFICIENT_REAL_PREDICTED_PAIRS",
                     "counts":count,
                     "min_positive":REQUIRED_LIRA_POS,
                     "min_negative":REQUIRED_LIRA_NEG}
    X=np.asarray(feature,np.float64)
    head=make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1400,C=.3,class_weight="balanced",
                           random_state=20262691))
    head.fit(X,labels)
    return head,{"status":"TRAINED_PREDICTED_PAIR_RELATION",
                 "counts":count,"training_pairs":len(labels),
                 "trained_original_historical_LIRA":False}


def _rank(scene,method,head):
    pairs=scene["proposals"]
    if not pairs:return None,-np.inf
    if method=="geometry":
        scores=np.asarray([item["geometry"] for item in pairs],float)
        # Distances often in [-100,0], normalized for common CAL sweep.
        scores=1./(1.+np.exp(-np.clip(scores/18.,-40,40)))
    elif method=="new_lira":
        if head is None:return None,-np.inf
        features=np.asarray([p["features"] for p in pairs],float)
        scores=head.predict_proba(features)[:,1]
    else:raise ValueError("Unknown fixed policy")
    index=int(np.argmax(scores))
    return index,float(scores[index])


def _case_prediction(scene,method,head,threshold):
    index,score=_rank(scene,method,head)
    if index is None or score<threshold:
        return {"accepted":False,"correct":False,
                "score":None if not np.isfinite(score) else float(score),
                "proposal_index":None}
    p=scene["proposals"][index]
    return {"accepted":True,
            "correct":bool(scene["kind"]=="LINK" and p["y"]==1),
            "score":score,"proposal_index":index}


def _calibrate(scenes,method,head):
    eligible=[scene for scene in scenes if scene["kind"]=="LINK"]
    negative=[scene for scene in scenes if scene["kind"]=="NO_LINK"]
    if len(eligible)<2 or len(negative)<2:
        return {"status":"STOP_INSUFFICIENT_CAL_SCENES",
                "n_positive":len(eligible),"n_negative":len(negative)}
    # CAL-only finite preregistered grid. Never adjust after seeing TEST.
    best=(-1,-1.,-1.);choice=1.01
    for threshold in THRESHOLDS+(1.01,):
        predictions=[_case_prediction(s,method,head,threshold) for s in scenes]
        tp=sum(s["kind"]=="LINK" and p["correct"]
               for s,p in zip(scenes,predictions))
        wrong=sum(s["kind"]=="LINK" and p["accepted"] and not p["correct"]
                  for s,p in zip(scenes,predictions))
        fp=sum(s["kind"]=="NO_LINK" and p["accepted"]
               for s,p in zip(scenes,predictions))
        if (wrong/len(eligible)>.1 or fp/len(negative)>.1):
            continue
        criteria=(tp,-fp,-wrong)
        if criteria>best:best=criteria;choice=threshold
    return {"status":"CAL_FROZEN","accept_threshold":float(choice),
            "n_positive":len(eligible),"n_negative":len(negative),
            "n_correct_accept_CAL":int(max(best[0],0)),
            "n_wrong_accept_CAL":int(max(-best[2],0)),
            "n_false_accept_CAL":int(max(-best[1],0)),
            "CAL_empirical_risk_not_geological_guarantee":True}


def _simple_straight(a,b,n=40):
    line=np.rint(
        np.linspace(np.asarray(a,float),np.asarray(b,float),n)
    ).astype(int)
    return tuple(tuple(map(int,x)) for x in line)


def _path_success(scene,pair):
    """Path readout only after selected prediction, hidden truth for metric."""
    a=tuple(np.rint(pair["source"]).astype(int))
    b=tuple(np.rint(pair["target"]).astype(int))
    path=_simple_straight(a,b)
    oy,ox=scene["origin"]
    y0,y1,x0,x1=scene["rectangle"]
    gt=np.asarray(scene["hidden_centerline"],int)
    inside=np.zeros((ROI,ROI),bool)
    for yy,xx in path:
        if y0<=yy<y1 and x0<=xx<x1 and 0<=yy<ROI and 0<=xx<ROI:
            inside[yy,xx]=True
    true=np.zeros_like(inside)
    for y,x in gt:
        yy=y-oy;xx=x-ox
        if 0<=yy<ROI and 0<=xx<ROI and y0<=yy<y1 and x0<=xx<x1:
            true[yy,xx]=True
    if not inside.any() or not true.any():return False
    pred_near=ndi.binary_dilation(inside,iterations=2)
    true_near=ndi.binary_dilation(true,iterations=2)
    precision=true_near[inside].mean()
    recall=pred_near[true].mean()
    return bool(precision>=.8 and recall>=.8)


def _metrics(scenes,method,head,cal):
    thresholds=cal if cal["status"]=="CAL_FROZEN" else {
        "accept_threshold":1.01}
    report=[];po=ne=0
    for scene in scenes:
        pred=_case_prediction(
            scene,method,head,thresholds["accept_threshold"])
        path_ok=False
        if pred["accepted"] and pred["correct"]:
            path_ok=_path_success(
                scene,scene["proposals"][pred["proposal_index"]])
        row={
            "case_id":scene["id"],"section":scene["section"],
            "kind":scene["kind"],"status":scene["status"],
            "n_source_ports":scene["source_count"],
            "n_predicted_hard_branches":scene["branch_count"],
            "n_sbpp_proposals":len(scene["proposals"]),
            "has_correct_candidate":scene["has_true_proposal"],
            "accepted":pred["accepted"],"correct":pred["correct"],
            "path_tol2_success":path_ok,
            "lira_or_geometry_score":pred["score"],
            "n_predictions_leaking_expert_endpoints":0}
        report.append(row)
        po+=scene["kind"]=="LINK"
        ne+=scene["kind"]=="NO_LINK"
    pos=[r for r in report if r["kind"]=="LINK"]
    neg=[r for r in report if r["kind"]=="NO_LINK"]
    result={
        "n_positive_gaps":len(pos),"n_synthetic_candidate_absent_NO_LINK":len(neg),
        "n_source_available":sum(r["n_source_ports"]>0 for r in pos),
        "n_candidate_available":sum(r["n_sbpp_proposals"]>0 for r in pos),
        "n_true_candidate_available":sum(r["has_correct_candidate"] for r in pos),
        "n_correct_accepted":sum(r["correct"] for r in pos),
        "n_wrong_accepted":sum(r["accepted"] and not r["correct"] for r in pos),
        "n_false_accepted_synthetic_NO_LINK":sum(r["accepted"] for r in neg),
        "n_correctly_completed_straight_path":sum(r["path_tol2_success"] for r in pos),
        "n_real_TEST_source_sections":len({r["section"] for r in pos}),
        "end_to_end_true_path_fraction":float(
            sum(r["path_tol2_success"] for r in pos)/len(pos))
            if pos else None,
        "not_historical_model":True,"oracle_endpoints_used_as_scoring_input":False,
        "intervention_location_is_expert_selected":True,
    }
    return result,report


def execute(images,annotations,trained_root,out,*,split_mapping=None):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    receipt=json.loads((trained_root/"v26_segmenter_frozen.json").read_text())
    if receipt["TEST_pixel_arrays_or_labels_loaded_before_freeze"]:
        raise RuntimeError("STOP segmentation TEST leakage")
    statuses={};frozen={};models={};hist={};cal_summary={}
    with zipfile.ZipFile(images) as zi,zipfile.ZipFile(annotations) as za:
        im_paths=sections(zi);experts=sections(za,expert=True)
        ids=sorted(set(im_paths)&set(experts))
        mapping=split_map(ids) if split_mapping is None else dict(split_mapping)
        if set(mapping).difference(ids):raise RuntimeError('External cohort contains unknown expert IDs')
        ids_train=[sid for sid in ids if mapping.get(sid)=="TRAIN"]
        ids_cal=[sid for sid in ids if mapping.get(sid)=="CAL"]
        ids_test=[sid for sid in ids if mapping.get(sid)=="TEST"]
        # All *both* frontend candidate models trained and CAL decisions
        # frozen before even decoding TEST image or mask.
        for name in ("unet","anza_v1"):
            model=restore_model(trained_root,name,receipt)
            models[name]=model
            config={
                "train_normalization":receipt["train_normalization"],
                "selected_threshold":receipt["frontends"][name][
                    "selected_CAL_threshold"]}
            train=_dataset(zi,za,model,ids_train,config)
            cal=_dataset(zi,za,model,ids_cal,config)
            lira,fit_receipt=_train_relation(train)
            if lira is not None:
                path=out/(name+"_new_lira.joblib")
                joblib.dump(lira,path)
                fit_receipt["checkpoint_SHA256"]=hashlib.sha256(
                    path.read_bytes()).hexdigest()
                fit_receipt["checkpoint_filename"]=path.name
            hist[name]=lira
            gates={
                m:_calibrate(cal,m,lira)
                if m=="geometry" or lira is not None else
                {"status":"STOP_UNTRAINED_RELATION","accept_threshold":1.01}
                for m in ("geometry","new_lira")}
            frozen[name]={
                "segmentation_checkpoint_SHA256":receipt["frontends"][
                    name]["checkpoint_SHA256"],
                "real_TRAIN_pairs":fit_receipt,
                "CAL_thresholds":gates,
                "n_TRAIN_gap_worlds":len(train),
                "n_CAL_gap_worlds":len(cal),
                "no_expert_endpoints_used_for_predicted_SBPP":True,
                "original_historical_ANZA_or_LIRA":False,
            }
            statuses[name]=fit_receipt["status"]
            del train,cal
        lock={
            "status":"V26_NEW_REAL_MODEL_AND_LIRA_FROZEN_BEFORE_TEST",
            "protocol":"TV_V26_PREDICTED_CRACKS_REAL_IMAGE_CONTROLLED_GAPS",
            "real_TR_sections":ids_train,"real_CAL_sections":ids_cal,
            "buffer_sections":[s for s in ids if mapping[s]=="BUFFER"],
            "reserved_unseen_TEST_sections":ids_test,
            "no_TEST_image_mask_decoded_for_supervised_learning":True,
            "frontends":frozen,
            "segmentation_frozen_receipt_sha256":hashlib.sha256(
                (trained_root/"v26_segmenter_frozen.json").read_bytes()
            ).hexdigest(),
            "historical_model_weights_absent":True}
        lock_path=out/"v26_all_model_CAL_frozen.json"
        lock_path.write_text(json.dumps(lock,indent=2,ensure_ascii=False)+"\n")
        lock_sha=hashlib.sha256(lock_path.read_bytes()).hexdigest()
        # NOW AND ONLY NOW open official five TEST expert section images/masks.
        results={};all_rows=[]
        for name in ("unet","anza_v1"):
            model=models[name]
            config={
                "train_normalization":receipt["train_normalization"],
                "selected_threshold":receipt["frontends"][name][
                    "selected_CAL_threshold"]}
            test=_dataset(zi,za,model,ids_test,config)
            model_stats={}
            for method in ("geometry","new_lira"):
                metrics,rows=_metrics(
                    test,method,hist[name],
                    frozen[name]["CAL_thresholds"][method])
                model_stats[method]=metrics
                all_rows.extend({
                    "frontend":name,"method":method,**r
                } for r in rows)
            results[name]={
                "methods":model_stats,
                "n_REAL_TEST_intervention_gaps":len(test)//2,
                "source_is_real_CRACKS_seismic_pixels":True,
                "segmentation_is_true_trained_AZConv_or_U_NET":True,
                "gap_rectangle_expert_intervention_but_NOT_oracle_input":True,
                "predicted_source_port_and_candidate_only":True,
                "original_ANZA_SBPP_LIRA_checkpoints_used":False,
                "new_LIRA_may_STOP_if_insufficient_proposals":True}
        case_file=out/"v26_real_predicted_case_decisions.csv"
        with case_file.open("w",newline="",encoding="utf8") as f:
            writer=csv.DictWriter(f,fieldnames=sorted({
                k for row in all_rows for k in row}))
            writer.writeheader();writer.writerows(all_rows)
        summary={
            "status":"V26_REAL_IMAGE_NEW_MODELS_PREDICTED_PORT_E2E_EXECUTED",
            "seismic_dataset":"official CRACKS v2 DOI 10.5281/zenodo.13926822",
            "real_TEST_image_accessed_after_ALL_fitted_models_frozen":True,
            "n_official_real_expert_TEST_sections":len(ids_test),
            "n_buffer_sections":len(lock["buffer_sections"]),
            "n_train_sections":len(ids_train),
            "n_cal_sections":len(ids_cal),
            "frozen_SHA256":lock_sha,
            "case_csv_SHA256":hashlib.sha256(
                case_file.read_bytes()).hexdigest(),
            "models":results,
            "historical_pretrained_weights_recovered":False,
            "V27_blue_eligible_source_selection_after_V26_source_audit":bool(split_mapping is not None),
            "no_claim_ANZA_SOTA":True,
            "synthetic_candidate_absent_NO_LINK_is_not_geological_truth":True,
        }
        (out/"v26_real_e2e_results.json").write_text(
            json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
        print(json.dumps(summary,indent=2))
        return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--images",type=Path,required=True)
    p.add_argument("--annotations",type=Path,required=True)
    p.add_argument("--trained-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    execute(a.images,a.annotations,a.trained_root,a.out)
