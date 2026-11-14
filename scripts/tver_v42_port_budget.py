"""V42: real Thebe fixed-48 source-port budget study, no TEST7.

Preserves original ANZA weights/threshold/source and original candidate rules.
Extra source ports trade off against fewer landing candidates per source.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ALLOCATIONS={"original_4x12":(4,12),"ports6_k8":(6,8),
             "ports8_k6":(8,6),"ports12_k4":(12,4)}
MAX_TOTAL=48
V33_SHA="ba0c3b8060fd0e3cb1c15f6c1598b94a977677c2b075663b5414207df382c25e"
FROZEN_THRESHOLD=.65


def evaluate_scenes(cohort,scenes,propose,is_true_relation):
    rows=[]
    for scene in scenes:
        banks={}
        for name,(max_sources,k) in ALLOCATIONS.items():
            raw=propose(scene["visible_prob"],scene["gray"],scene["rectangle"],
                        FROZEN_THRESHOLD,max_sources=max_sources,per_source_k=k)
            if len(raw["proposals"])>MAX_TOTAL or raw.get("n_proposals_before_cap",0)>MAX_TOTAL:
                raise ValueError("Budget violation")
            if any("y" in p or "true" in p for p in raw["proposals"]):
                raise ValueError("Oracle leaks into predicted pair proposals")
            banks[name]=raw
        b=banks["original_4x12"]["proposals"]
        original=scene["proposals"]
        if [(p["source"],p["target"]) for p in b]!=[(p["source"],p["target"]) for p in original]:
            raise RuntimeError("Frozen 4x12 source/candidate list changed")
        for name,raw in banks.items():
            ps=raw["proposals"]
            truth=[bool(scene["kind"]=="LINK" and is_true_relation(
                p,scene["true_ends"],scene["origin"])) for p in ps]
            top=int(np.argmax([p["geometry"] for p in ps])) if ps else None
            rows.append({
                "cohort":cohort,"section":int(scene["section"]),"gap_id":str(scene["id"]),
                "world":str(scene["kind"]),"method":name,
                "n_eligible_source_ports_before_cap":raw.get("n_eligible_source_ports_before_cap",0),
                "n_used_source_ports":raw["source_count"],
                "n_candidates":len(ps),"n_pre_cap_pairs":raw.get("n_proposals_before_cap",len(ps)),
                "has_true_candidate":int(any(truth)),
                "geometry_top1_correct":int(top is not None and truth[top]),
                "status":raw["status"],
            })
    return rows


def summarize(rows):
    out={}
    for cohort in sorted({r["cohort"] for r in rows}):
        original={r["gap_id"]:r for r in rows
                  if r["cohort"]==cohort and r["world"]=="LINK"
                  and r["method"]=="original_4x12"}
        for name in ALLOCATIONS:
            positive=[r for r in rows if r["cohort"]==cohort
                      and r["world"]=="LINK" and r["method"]==name]
            negative=[r for r in rows if r["cohort"]==cohort
                      and r["world"]=="NO_LINK" and r["method"]==name]
            if len(positive)!=len(original) or len(negative)!=len(original):
                raise ValueError("Method/source coverage changed")
            matched={r["gap_id"]:r for r in positive}
            if set(matched)!=set(original):
                raise ValueError("Source gap identities changed")
            out[f"{cohort}/{name}"]={
                "n_controlled_positive_gaps":len(positive),
                "n_distinct_source_sections":len({r["section"] for r in positive}),
                "correct_candidate_available":sum(r["has_true_candidate"] for r in positive),
                "geometry_top1_correct":sum(r["geometry_top1_correct"] for r in positive),
                "rescued_vs_original":sum(not b["has_true_candidate"] and matched[g]["has_true_candidate"] for g,b in original.items()),
                "lost_vs_original":sum(b["has_true_candidate"] and not matched[g]["has_true_candidate"] for g,b in original.items()),
                "n_zero_source_ports":sum(r["n_used_source_ports"]==0 for r in positive),
                "n_eligible_source_ports_gt4":sum(r["n_eligible_source_ports_before_cap"]>4 for r in positive),
                "n_eligible_source_ports_gt8":sum(r["n_eligible_source_ports_before_cap"]>8 for r in positive),
                "mean_candidate_count":sum(r["n_candidates"] for r in positive)/max(len(positive),1),
                "n_full_budget_48":sum(r["n_candidates"]==48 for r in positive),
                "n_fake_nolink_worlds_with_candidate":sum(r["n_candidates"]>0 for r in negative),
                "budget":MAX_TOTAL,
            }
    return out


def execute(memmap_root,frontends_dir,out):
    import torch
    from scripts.tver_v34_anza_p0_relation_cal import _state_report
    from scripts.tver_v33_thebe_train_segmenters import read_memmap
    from scripts.tver_v33_thebe_relation_cal import _candidate_scenes,_load_model,TRAIN_SOURCE_IDS,RELATION_CAL_SOURCE_IDS
    from scripts.tver_v26_predicted_cracks_e2e import sbpp_proposals,is_true_relation
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    receipt=_state_report(frontends_dir)
    if receipt["frontends"]["anza_v1"]["SHA256"]!=V33_SHA:
        raise RuntimeError("Thebe original ANZA SHA changed")
    model=_load_model(frontends_dir,"anza_v1",receipt)
    rows=[]
    for label,src,mask,ids in (
        ("TRAIN",read_memmap(memmap_root,"train_seismic_float32.npy"),
         read_memmap(memmap_root,"train_fault_bool.npy"),TRAIN_SOURCE_IDS),
        ("REL_CAL_ALREADY_VIEWED",read_memmap(memmap_root,"val_seismic_float32.npy"),
         read_memmap(memmap_root,"val_fault_bool.npy"),RELATION_CAL_SOURCE_IDS)):
        scenes,_=_candidate_scenes(src,mask,ids,receipt["normalization"],model,FROZEN_THRESHOLD)
        rows.extend(evaluate_scenes(label,scenes,sbpp_proposals,is_true_relation))
        print("V42_COHORT",label,"worlds",len(scenes),flush=True)
    out_summary=summarize(rows)
    if (out_summary["TRAIN/original_4x12"]["correct_candidate_available"]!=116 or
        out_summary["REL_CAL_ALREADY_VIEWED/original_4x12"]["correct_candidate_available"]!=65 or
        out_summary["REL_CAL_ALREADY_VIEWED/original_4x12"]["geometry_top1_correct"]!=1):
        raise RuntimeError("V34 baseline 116 TRAIN /65 CAL /1 geometric top1 not reproduced")
    report={
        "status":"V42_REAL_THEBE_FIXED_48_PORT_ALLOCATION_POSTHOC_TEST7_UNOPENED",
        "original_V33_ANZA_SHA":V33_SHA,
        "original_V33_receipt_sha256":hashlib.sha256((Path(frontends_dir)/"v33_THEBE_frontends_topology_TRAIN_CAL_frozen.json").read_bytes()).hexdigest(),
        "same_model_threshold":FROZEN_THRESHOLD,
        "candidate_budget":MAX_TOTAL,
        "methods":ALLOCATIONS,
        "original_external_TEST7_unopened":True,
        "original_REL_CAL_already_viewed_posthoc":True,
        "source_sections_not_independent_geological_volumes":True,
        "no_natural_geological_negative_relation_truth":True,
        "summary":out_summary,
    }
    (out/"V42_RESULT.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    with (out/"V42_CASE_ROWS.csv").open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({"status":report["status"],"summary":out_summary},indent=2),flush=True)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mmap-root",type=Path,required=True)
    p.add_argument("--frontends-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args();execute(a.mmap_root,a.frontends_dir,a.out_dir)
