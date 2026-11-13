#!/usr/bin/env python3
"""V27: crowd-CAL risk-limited selective linking on real CRACKS images.

Uses existing crowd-trained V23 HGB + image-only V25 region hybrid.
The selection threshold is fitted on nonexpert crowd-CAL pixels only.
Expert reference is opened strictly AFTER all predictions and link choices.
No natural geologic fault-instance labels are available; this is exploratory.
"""
from __future__ import annotations
import argparse,csv,json,zipfile
from pathlib import Path
import numpy as np
from scripts.cracks_real_auto_run_v23 import (
    _train,TRAIN_IDS,CAL_IDS,zip_index,load_image,load_mask,
    consensus_training_mask,evaluate_mask,candidate_reachability,
    sha_file,_bootstrap,
)
from scripts.cracks_real_auto_core_v23 import anisotropic_features
from scripts.cracks_real_region_v25 import run_regions

THRESHOLDS=(.75,.50,.30,.10,-.10,-.30)
MAX_CROWD_ADDED_ORANGE_FRACTION=.0005

def subset_mask(prediction,threshold):
    original=np.asarray(prediction["raw_mask"],bool)
    completed=original.copy()
    count=0
    for row in prediction["selected"]:
        if float(row["score"])<threshold:continue
        points=np.asarray(row["path"],dtype=int)
        if len(points)<2:continue
        completed[points[:,0],points[:,1]]=True
        count+=1
    return completed,count

def calibrate_region_link_threshold(calibration_cases):
    """Choose link acceptance solely from nonexpert crowd labels.

    Maximize accepted links with fixed bound on incremental ORANGE.
    Never choose threshold using expert mask or test connectivity.
    """
    scored=[]
    for threshold in THRESHOLDS:
        novel_blue=novel_orange=blue_total=orange_total=links=0
        for pred,pos,neg in calibration_cases:
            before=np.asarray(pred["raw_mask"],bool)
            after,n=subset_mask(pred,threshold)
            added=after&~before
            novel_blue+=int((added&pos).sum())
            novel_orange+=int((added&neg).sum())
            blue_total+=int(pos.sum());orange_total+=int(neg.sum())
            links+=n
        rate=novel_orange/max(orange_total,1)
        scored.append({"threshold":threshold,"links":links,
                       "new_blue":novel_blue,"new_orange":novel_orange,
                       "denominator_crowd_blue":blue_total,
                       "denominator_crowd_orange":orange_total,
                       "added_orange_fraction":rate,
                       "within_fixed_budget":rate<=MAX_CROWD_ADDED_ORANGE_FRACTION})
    admissible=[r for r in scored if r["within_fixed_budget"]]
    if not admissible:
        return None,scored
    choice=max(admissible,key=lambda r:(r["links"],r["new_blue"],-r["new_orange"],r["threshold"]))
    return choice,scored

def evaluate(images_path,annotations_path,out,max_expert=40):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(images_path) as zi,zipfile.ZipFile(annotations_path) as za:
        images,annotations=zip_index(zi,za)
        all_expert=sorted(k for k in images if k in annotations and
                          "expert" in annotations[k])[:max_expert]
        if len(all_expert)<10:raise RuntimeError("Need at least ten expert sections")
        if set(all_expert)&(set(TRAIN_IDS)|set(CAL_IDS)):
            raise RuntimeError("Expert sections overlap crowd TRAIN/CAL")
        model,segmentation_threshold,model_receipt=_train(
            zi,za,images,annotations,out=out)
        crowd_cal=[]
        for sid in CAL_IDS:
            if sid not in images or sid not in annotations:continue
            if "expert" in annotations[sid]:raise RuntimeError("Expert in calibration")
            target=consensus_training_mask(za,annotations[sid])
            if target is None:continue
            pos,neg=target
            image=load_image(zi,images[sid])
            feats=anisotropic_features(image)
            probability=model.predict_proba(feats.reshape(-1,feats.shape[-1]))[:,1].reshape(image.shape)
            result=run_regions(probability,segmentation_threshold,mode="region_hybrid")
            crowd_cal.append((result,pos,neg))
        if len(crowd_cal)<4:raise RuntimeError("Too few crowd CAL sections")
        link_choice,link_curves=calibrate_region_link_threshold(crowd_cal)
        if link_choice is None:raise RuntimeError("No admissible fixed link threshold")
        link_threshold=link_choice["threshold"]
        sections=[]
        for idx,sid in enumerate(all_expert):
            image=load_image(zi,images[sid])
            f=anisotropic_features(image)
            prob=model.predict_proba(f.reshape(-1,f.shape[-1]))[:,1].reshape(image.shape)
            result=run_regions(prob,segmentation_threshold,mode="region_hybrid")
            raw=np.asarray(result["raw_mask"],bool)
            free,full_links=subset_mask(result,-.30)
            controlled,controlled_links=subset_mask(result,link_threshold)
            # Constructed without gold; expert mask only now opened for metrics.
            expert=load_mask(za,annotations[sid]["expert"])
            row={"section":sid,"links_free":full_links,"links_controlled":controlled_links}
            for name,mask in (("raw",raw),("free",free),("controlled",controlled)):
                scores=evaluate_mask(mask,expert)
                for k,v in scores.items():row[f"{name}_{k}"]=v
            sections.append(row)
            print("V27",sid,"links",full_links,"risk-limited",controlled_links,flush=True)
        if not sections:raise RuntimeError("No expert sections evaluated")
    output=out/"cases_v27.csv"
    with output.open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(sections[0]))
        writer.writeheader();writer.writerows(sections)
    results={}
    for name in ("raw","free","controlled"):
        tp=sum(r[f"{name}_expert_blue_recalled_tol2"] for r in sections)
        pos=sum(r[f"{name}_N_expert_blue"] for r in sections)
        fp=sum(r[f"{name}_explicit_orange_false_pixels"] for r in sections)
        neg=sum(r[f"{name}_N_explicit_orange"] for r in sections)
        connected=sum(r[f"{name}_connected_gold_anchor_pairs"] for r in sections)
        eligible=sum(r[f"{name}_eligible_paths"] for r in sections)
        results[name]={
            "blue_recall_tol2":tp/max(pos,1),
            "orange_fp_fraction":fp/max(neg,1),
            "connected_expert_anchor_pairs":connected,
            "N_eligible_expert_paths":eligible,
            "connectivity_rate":connected/max(eligible,1),
            "total_added_links":sum(r[f"links_{name}"] for r in sections) if name!="raw" else 0
        }
    report={
        "status":"EXECUTED_EXPLORATORY",
        "protocol":"CRACKS_V27_NONEXPERT_CAL_RISK_LIMITED_LINKS",
        "segmentation":"V23 crowd only HGB features, not original ANZA",
        "linker":"V25 image-only region hybrid, restricted by nonexpert CAL added-orange budget",
        "trained_on_expert":False,
        "expert_used_for_inference":False,
        "natural_geological_links_ground_truth":False,
        "section_ids":[r["section"] for r in sections],
        "model":model_receipt,
        "crowd_cal_sections":len(crowd_cal),
        "fixed_max_added_crowd_orange_fraction":MAX_CROWD_ADDED_ORANGE_FRACTION,
        "selected_link_threshold":link_threshold,
        "link_calibration_curve":link_curves,
        "metrics":results,
        "source_images_sha256":sha_file(images_path),
        "source_annotations_sha256":sha_file(annotations_path),
        "per_section_sha256":sha_file(output),
        "interpretation":"Same already-opened F3 expert volume; only conditional connectivity of annotated traces, not source identities. Retrospective exploratory method."
    }
    (out/"report_v27.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"threshold":link_threshold,"metrics":results},indent=2))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--images",required=True,type=Path)
    p.add_argument("--annotations",required=True,type=Path)
    p.add_argument("--out",default=Path("results/cracks_real_v27"),type=Path)
    p.add_argument("--max-expert",default=40,type=int)
    a=p.parse_args()
    evaluate(a.images,a.annotations,a.out,a.max_expert)

if __name__=="__main__":main()
