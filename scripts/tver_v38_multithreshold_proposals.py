"""V38: posthoc exploratory candidate recall audit on frozen Thebe ANZA.

No neural retraining, no threshold tuning on REL-CAL, no Thebe TEST7 access.
The four fixed strategies compare the same 48-pair budget and the original
predicted endpoint pipeline. They are NOT an independently confirmatory test:
VAL1 40..99 already appeared in V34 result analysis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CAP = 48
DUPLICATE_TOLERANCE_PX = 4.0
THRESHOLDS = (0.45, 0.55, 0.65, 0.75)
METHODS = {
    "original_065": (0.65,),
    "union_055_065": (0.65, 0.55),
    "union_055_065_075": (0.65, 0.55, 0.75),
    "union_045_055_065_075": (0.65, 0.55, 0.75, 0.45),
}


def same_pair(a, b, tolerance=DUPLICATE_TOLERANCE_PX):
    """Duplicate physical relation, including reversed endpoint order."""
    a0, a1 = np.asarray(a["source"], float), np.asarray(a["target"], float)
    b0, b1 = np.asarray(b["source"], float), np.asarray(b["target"], float)
    return bool((np.linalg.norm(a0-b0)<=tolerance and
                 np.linalg.norm(a1-b1)<=tolerance) or
                (np.linalg.norm(a0-b1)<=tolerance and
                 np.linalg.norm(a1-b0)<=tolerance))


def proposal_banks(visible_prob, gray, rect, propose_fn):
    """Only image/model-derived inputs, no labels, hidden centerline or oracle ends."""
    banks = {}
    for t in THRESHOLDS:
        raw = propose_fn(visible_prob, gray, rect, t)
        records = []
        for pair in raw["proposals"]:
            if "y" in pair or "true" in pair:
                raise ValueError("STOP proposal carries ground-truth label before matching")
            record = dict(pair)
            record["producer_threshold"] = t
            records.append(record)
        banks[t] = {"proposals": records, "status": raw["status"],
                    "hard_branches": int(raw["hard_branches"]),
                    "source_count": int(raw["source_count"])}
    return banks


def merge_fixed_budget(banks, thresholds, cap=CAP):
    """Deterministic round-robin by each bank's original geometric ranking.

    Includes the original .65 bank first.  Geometry is NOT a truth score.
    Extra thresholds do not increase the total pair budget above 48.
    """
    if not thresholds or cap<=0:
        raise ValueError("At least one threshold and positive cap required")
    for t in thresholds:
        if t not in banks:
            raise KeyError(f"No predictions for threshold {t}")
    if thresholds==(0.65,):
        # Exact V34 SBPP baseline: never deduplicate or reorder its 48 pairs.
        return [dict(p) for p in banks[0.65]["proposals"][:cap]]
    indices = {t: 0 for t in thresholds}
    unique = []
    pool = {t: banks[t]["proposals"] for t in thresholds}
    while len(unique)<cap:
        progressed = False
        for t in thresholds:
            rows = pool[t]
            while indices[t]<len(rows):
                row=rows[indices[t]]
                indices[t]+=1
                progressed=True
                if not any(same_pair(row,prior) for prior in unique):
                    unique.append(dict(row))
                    break
            if len(unique)>=cap:
                break
        if not progressed:
            break
    return unique[:cap]


def summarize(rows):
    """Positive controlled gaps only; NO_LINK has no expert instance identity."""
    result={}
    cohorts=sorted({r["cohort"] for r in rows})
    for cohort in cohorts:
        subset=[r for r in rows if r["cohort"]==cohort]
        base={r["case_id"]:r for r in subset if r["method"]=="original_065"
              and r["world"]=="LINK"}
        for method in METHODS:
            v=[r for r in subset if r["method"]==method]
            pos=[r for r in v if r["world"]=="LINK"]
            neg=[r for r in v if r["world"]=="NO_LINK"]
            if len({(r['section'],r['case_id']) for r in pos})!=len(pos):
                raise ValueError("Duplicate positive case keys")
            if len(pos)!=len(base):
                raise ValueError("Methods have different source positive cases")
            by={r["case_id"]:r for r in pos}
            rescued=sum(not b["true_available"] and by[k]["true_available"]
                        for k,b in base.items())
            lost=sum(b["true_available"] and not by[k]["true_available"]
                     for k,b in base.items())
            result[f"{cohort}/{method}"]={
                "n_link_gaps":len(pos),
                "source_section_count":len({r["section"] for r in pos}),
                "correct_candidate_available":sum(r["true_available"] for r in pos),
                "geometry_top1_correct":sum(r["geometry_top1_correct"] for r in pos),
                "n_synthetic_nolink":len(neg),
                "synthetic_nolink_with_any_candidate":sum(r["candidate_count"]>0 for r in neg),
                "mean_LINK_candidate_count":round(
                    sum(r["candidate_count"] for r in pos)/max(1,len(pos)),3),
                "max_LINK_candidate_count":max([r["candidate_count"] for r in pos],default=0),
                "rescue_vs_original":rescued,
                "lost_vs_original":lost,
                "candidate_budget":CAP,
            }
    return result


def _rows_for_scenes(cohort, scenes, propose_fn, is_true_relation):
    rows=[]
    for scene in scenes:
        banks=proposal_banks(scene["visible_prob"], scene["gray"],
                             scene["rectangle"], propose_fn)
        # Baseline identity check: exactly the same source and target order.
        exact=[(p["source"],p["target"])
               for p in banks[0.65]["proposals"]]
        expected=[(p["source"],p["target"])
                  for p in scene["proposals"]]
        if exact!=expected:
            raise RuntimeError("STOP frozen 0.65 SBPP baseline not reproduced")
        for name,thresholds in METHODS.items():
            proposals=merge_fixed_budget(banks,thresholds)
            # Expert labels are read ONLY after all proposal variants exist.
            truth=[bool(scene["kind"]=="LINK" and is_true_relation(
                p,scene["true_ends"],scene["origin"])) for p in proposals]
            best=(max(range(len(proposals)),
                      key=lambda k:float(proposals[k]["geometry"]))
                  if proposals else None)
            rows.append({
                "cohort":cohort,
                "section":int(scene["section"]),
                "case_id":str(scene["id"]),
                "world":str(scene["kind"]),
                "method":name,
                "candidate_count":len(proposals),
                "true_available":int(any(truth)),
                "geometry_top1_correct":int(best is not None and truth[best]),
                "geometry_top1_score":(
                    float(proposals[best]["geometry"]) if best is not None else None),
                "thresholds_used":",".join(map(str,thresholds)),
                "hard_branches_065":banks[0.65]["hard_branches"],
                "status_065":banks[0.65]["status"],
            })
    return rows


def execute(memmap_root,frontend_dir,out):
    # Deferred imports make the pure candidate policy unit-testable without
    # the original repository's large image/segmentation dependencies.
    import torch
    from scripts.tver_v34_anza_p0_relation_cal import (
        _state_report,FROZEN_ANZA_TOPOLOGY)
    from scripts.tver_v33_thebe_train_segmenters import read_memmap
    from scripts.tver_v33_thebe_relation_cal import (
        _candidate_scenes,_load_model,TRAIN_SOURCE_IDS,RELATION_CAL_SOURCE_IDS)
    from scripts.tver_v26_predicted_cracks_e2e import (
        sbpp_proposals,is_true_relation)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    receipt=_state_report(frontend_dir)
    assert FROZEN_ANZA_TOPOLOGY==0.65
    segmenter=_load_model(frontend_dir,"anza_v1",receipt)
    report={"status":"V38_POSTHOC_CANDIDATE_AUDIT_STARTED",
            "model_sha256":receipt["frontends"]["anza_v1"]["SHA256"],
            "source_receipt_SHA256":hashlib.sha256(
               (Path(frontend_dir)/"v33_THEBE_frontends_topology_TRAIN_CAL_frozen.json").read_bytes()).hexdigest(),
            "risk_gate_not_changed":True,
            "neural_P0_not_retrained":True,
            "existing_V34_CAL_already_viewed_posthoc":True,
            "original_Thebe_TEST7_NEVER_OPENED":True,
            "no_natural_geological_NO_LINK_truth":True,
            "same_total_candidate_budget":CAP,
            "method_thresholds":METHODS}
    rows=[]
    for cohort,src,mask,sections in (
        ("TRAIN",read_memmap(memmap_root,"train_seismic_float32.npy"),
         read_memmap(memmap_root,"train_fault_bool.npy"),TRAIN_SOURCE_IDS),
        ("REL_CAL_ALREADY_VIEWED",read_memmap(memmap_root,"val_seismic_float32.npy"),
         read_memmap(memmap_root,"val_fault_bool.npy"),RELATION_CAL_SOURCE_IDS)):
        scenes, _ = _candidate_scenes(
            src,mask,sections,receipt["normalization"],segmenter,0.65)
        rows.extend(_rows_for_scenes(cohort,scenes,sbpp_proposals,is_true_relation))
        print("V38_COUNTS",cohort,"n_intervention_worlds",len(scenes),flush=True)
    if not rows:
        raise RuntimeError("STOP no real predicted candidate cases")
    with (out/"V38_CASE_ROWS.csv").open("w",newline="",encoding="utf8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    report["summary"]=summarize(rows)
    report["status"]="V38_REAL_THEBE_FIXED_48_CANDIDATE_AUDIT_COMPLETE_TEST7_UNOPENED"
    (out/"V38_RESULT.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps({"status":report["status"],"summary":report["summary"]},
          ensure_ascii=False,indent=2),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mmap-root",required=True,type=Path)
    parser.add_argument("--frontends-dir",required=True,type=Path)
    parser.add_argument("--out-dir",required=True,type=Path)
    args=parser.parse_args()
    execute(args.mmap_root,args.frontends_dir,args.out_dir)