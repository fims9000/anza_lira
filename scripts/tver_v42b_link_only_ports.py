"""V42b bounded-compute real Thebe source-port study (LINK candidates only).

The pre-registered full V42 RUN 43619324978 was cancelled after timeout;
V42b reduces work to 120 original previously viewed REL_CAL positive gaps.
No new TEST7, TRAIN or synthetic NO_LINK outcome is claimed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ANZA_SHA="ba0c3b8060fd0e3cb1c15f6c1598b94a977677c2b075663b5414207df382c25e"
ALLOCATIONS={"original_4x12":(4,12),"ports6_k8":(6,8),
             "ports8_k6":(8,6),"ports12_k4":(12,4)}
MAX_PAIRS=48
THRESHOLD=.65


def compare_positive_scenes(scenes,propose,is_true_relation):
    rows=[]
    for scene in scenes:
        if scene["kind"]!="LINK":
            raise ValueError("V42b is candidate-only LINK audit, no NO_LINK")
        baseline=scene["proposals"]
        if not all("y" in item for item in baseline):
            raise ValueError("Baseline expert labels are required only for scoring")
        original_true=any(int(p["y"]) for p in baseline)
        for name,(sources,k) in ALLOCATIONS.items():
            if name=="original_4x12":
                raw={
                    "proposals":baseline,"status":scene["status"],
                    "source_count":scene["source_count"],
                    "hard_branches":scene["branch_count"],
                    "n_eligible_source_ports_before_cap":None,
                }
            else:
                raw=propose(scene["visible_prob"],scene["gray"],
                            scene["rectangle"],THRESHOLD,
                            max_sources=sources,per_source_k=k)
            candidates=raw["proposals"]
            if len(candidates)>MAX_PAIRS:
                raise RuntimeError("48 candidate budget violated")
            if name!="original_4x12" and any("y" in p or "true" in p
                    for p in candidates):
                raise ValueError("Expert truth leaked into proposed candidate")
            labels=[int(p["y"]) if name=="original_4x12" else
                    int(is_true_relation(p,scene["true_ends"],scene["origin"]))
                    for p in candidates]
            idx=int(np.argmax([p["geometry"] for p in candidates])) if candidates else None
            rows.append({
                "method":name,"section":int(scene["section"]),
                "gap_id":str(scene["id"]),"world":"LINK",
                "n_candidates":len(candidates),
                "n_source_ports_used":raw["source_count"],
                "n_source_ports_eligible_before_cap":raw.get("n_eligible_source_ports_before_cap"),
                "status":raw["status"],
                "original_dense_stop":int(scene["status"]=="DEGENERATE_DENSE_PREDICTED_MASK"),
                "original_correct_candidate":int(original_true),
                "has_true_candidate":int(any(labels)),
                "geometric_top1_correct":int(idx is not None and labels[idx]),
            })
    return rows


def summary(rows):
    base={r["gap_id"]:r for r in rows if r["method"]=="original_4x12"}
    if len(base)!=120 or len({r["section"] for r in base.values()})!=60:
        raise RuntimeError("Original Thebe 60-section 120-gap CAL cohort changed")
    if sum(r["has_true_candidate"] for r in base.values())!=65:
        raise RuntimeError("Original 65/120 true candidate source drift")
    if sum(r["geometric_top1_correct"] for r in base.values())!=1:
        raise RuntimeError("Original geometric 1/120 source drift")
    report={}
    for name in ALLOCATIONS:
        found={r["gap_id"]:r for r in rows if r["method"]==name}
        if set(found)!=set(base) or len(found)!=len(base):
            raise RuntimeError("Alternative lost source/gap or duplicated cases")
        vals=list(found.values())
        report[name]={
            "source_sections":60,"n_controlled_LINK_gaps":120,
            "true_candidate_available":sum(r["has_true_candidate"] for r in vals),
            "geometry_top1_correct":sum(r["geometric_top1_correct"] for r in vals),
            "rescued_vs_baseline":sum(not b["has_true_candidate"] and found[g]["has_true_candidate"] for g,b in base.items()),
            "lost_vs_baseline":sum(b["has_true_candidate"] and not found[g]["has_true_candidate"] for g,b in base.items()),
            "n_original_dense_stops":sum(r["original_dense_stop"] for r in vals),
            "n_eligible_source_ports_gt4":sum((r["n_source_ports_eligible_before_cap"] or 0)>4 for r in vals),
            "n_eligible_source_ports_gt8":sum((r["n_source_ports_eligible_before_cap"] or 0)>8 for r in vals),
            "mean_candidate_count":sum(r["n_candidates"] for r in vals)/120,
            "max_candidate_count":max(r["n_candidates"] for r in vals),
            "zero_candidate_cases":sum(not r["n_candidates"] for r in vals),
            "n_cap_48":sum(r["n_candidates"]==48 for r in vals),
        }
    return report


def execute(memmap_root,frontends_dir,out):
    import torch
    from scripts.tver_v34_anza_p0_relation_cal import _state_report
    from scripts.tver_v33_thebe_train_segmenters import read_memmap
    from scripts.tver_v33_thebe_relation_cal import _candidate_scenes,_load_model,RELATION_CAL_SOURCE_IDS
    from scripts.tver_v26_predicted_cracks_e2e import sbpp_proposals,is_true_relation
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    source=_state_report(frontends_dir)
    if source["frontends"]["anza_v1"]["SHA256"]!=ANZA_SHA:
        raise RuntimeError("Wrong ANZA weight SHA")
    model=_load_model(frontends_dir,"anza_v1",source)
    scenes,_=_candidate_scenes(read_memmap(memmap_root,"val_seismic_float32.npy"),
                              read_memmap(memmap_root,"val_fault_bool.npy"),
                              RELATION_CAL_SOURCE_IDS,source["normalization"],
                              model,THRESHOLD)
    positives=[s for s in scenes if s["kind"]=="LINK"]
    rows=compare_positive_scenes(positives,sbpp_proposals,is_true_relation)
    metrics=summary(rows)
    out_json={
        "status":"V42B_SCOPED_REAL_THEBE_PORT_ALLOCATION_CAL_LINK_ONLY_TEST7_UNOPENED",
        "original_V42_full_run_cancelled_after_timeout":43619324978,
        "original_ANZA_SHA256":ANZA_SHA,
        "original_checkpoint_receipt_SHA256":hashlib.sha256(
            (Path(frontends_dir)/"v33_THEBE_frontends_topology_TRAIN_CAL_frozen.json").read_bytes()).hexdigest(),
        "same_threshold":THRESHOLD,"same_total_candidate_budget":48,
        "evaluated_TRAIN_data":False,"evaluated_synthetic_NOLINK_data":False,
        "original_REL_CAL_previously_viewed":True,"original_TEST7_unopened":True,
        "no_link_acceptance_risk_evaluated":False,"summary":metrics,
    }
    (out/"V42B_RESULT.json").write_text(json.dumps(out_json,indent=2)+"\n",encoding="utf8")
    with (out/"V42B_LINK_CASE_ROWS.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    print(json.dumps({"status":out_json["status"],"summary":metrics},indent=2),flush=True)
    return out_json


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mmap-root",type=Path,required=True)
    ap.add_argument("--frontends-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    execute(a.mmap_root,a.frontends_dir,a.out_dir)
