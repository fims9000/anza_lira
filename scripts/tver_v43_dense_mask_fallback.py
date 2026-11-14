"""Posthoc original Thebe ANZA dense mask fallback without label leakage.

Adaptive quantiles act only when original SBPP status is dense-mask STOP.
No external TEST7 access, no retraining and no more than 48 proposals.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

FROZEN_ANZA_SHA="ba0c3b8060fd0e3cb1c15f6c1598b94a977677c2b075663b5414207df382c25e"
BASE=.65
MAX_PAIRS=48
DENSE_STOP="DEGENERATE_DENSE_PREDICTED_MASK"
METHODS={"baseline_065":None,"dense_q70":.7,"dense_q80":.8,"dense_q90":.9}


def adaptive_threshold(probability,quantile):
    p=np.asarray(probability,np.float32)
    if p.shape!=(96,96) or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError("Invalid segmentation prediction")
    if not 0<quantile<1:
        raise ValueError("Quantile outside (0,1)")
    tau=max(BASE,float(np.quantile(p,quantile)))
    if float(np.mean(p>=tau))>.38:
        tau=float(np.nextafter(np.float32(tau),np.float32(np.inf)))
    if float(np.mean(p>=tau))>.38:
        raise RuntimeError("Adaptive threshold did not enforce density gate")
    return tau


def candidate_banks(scene,propose):
    """All proposal variants must be built before expert-label matching."""
    baseline=propose(scene["visible_prob"],scene["gray"],scene["rectangle"],BASE)
    def identity(p):
        return (tuple(p["source"]),tuple(p["target"]))
    if [identity(p) for p in baseline["proposals"]]!=[
            identity(p) for p in scene["proposals"]]:
        raise RuntimeError("Original frozen 0.65 candidates not reproduced")
    banks={"baseline_065":(baseline,BASE)}
    for name,q in METHODS.items():
        if q is None:
            continue
        if baseline["status"]!=DENSE_STOP:
            banks[name]=(baseline,BASE)
            continue
        tau=adaptive_threshold(scene["visible_prob"],q)
        banks[name]=(propose(scene["visible_prob"],scene["gray"],scene["rectangle"],tau),tau)
    for name,(raw,_) in banks.items():
        if len(raw["proposals"])>MAX_PAIRS:
            raise RuntimeError("Candidate budget exceeded: "+name)
        if any("y" in p or "true" in p for p in raw["proposals"]):
            raise ValueError("Ground-truth leaked into candidate proposal")
    return banks


def case_rows(cohort,scenes,propose,is_true_relation):
    rows=[]
    for scene in scenes:
        banks=candidate_banks(scene,propose)
        for name,(raw,tau) in banks.items():
            proposals=raw["proposals"]
            # Truth only used AFTER all independent proposal variants exist.
            correct=[int(scene["kind"]=="LINK" and is_true_relation(
                p,scene["true_ends"],scene["origin"])) for p in proposals]
            rank=int(np.argmax([p["geometry"] for p in proposals])) if proposals else None
            rows.append({
                "cohort":cohort,"section":int(scene["section"]),
                "gap_id":scene["id"],"world":scene["kind"],"method":name,
                "original_dense_stop":int(scene["status"]==DENSE_STOP),
                "original_status":scene["status"],"new_status":raw["status"],
                "adaptive_tau":float(tau),
                "new_mask_fraction":float(np.mean(scene["visible_prob"]>=tau)),
                "n_candidates":len(proposals),
                "has_true_candidate":int(any(correct)),
                "geometry_top1_correct":int(rank is not None and correct[rank]),
            })
    return rows


def summarize(rows):
    answer={}
    for cohort in sorted({r["cohort"] for r in rows}):
        baseline={r["gap_id"]:r for r in rows if r["cohort"]==cohort and
                  r["world"]=="LINK" and r["method"]=="baseline_065"}
        for method in METHODS:
            positive=[r for r in rows if r["cohort"]==cohort and
                      r["world"]=="LINK" and r["method"]==method]
            negative=[r for r in rows if r["cohort"]==cohort and
                      r["world"]=="NO_LINK" and r["method"]==method]
            pair={r["gap_id"]:r for r in positive}
            if set(pair)!=set(baseline) or len(positive)!=len(baseline) or len(negative)!=len(baseline):
                raise RuntimeError("Case identity changed")
            answer[cohort+"/"+method]={
                "n_positive_gaps":len(positive),
                "source_section_count":len({r["section"] for r in positive}),
                "original_dense_stops":sum(r["original_dense_stop"] for r in positive),
                "zero_candidate_cases":sum(not r["n_candidates"] for r in positive),
                "correct_candidate_available":sum(r["has_true_candidate"] for r in positive),
                "geometry_top1_correct":sum(r["geometry_top1_correct"] for r in positive),
                "rescued_vs_baseline":sum(not b["has_true_candidate"] and pair[g]["has_true_candidate"] for g,b in baseline.items()),
                "lost_vs_baseline":sum(b["has_true_candidate"] and not pair[g]["has_true_candidate"] for g,b in baseline.items()),
                "mean_candidate_count":round(sum(r["n_candidates"] for r in positive)/max(1,len(positive)),3),
                "synthetic_nolink_with_any_candidate":sum(r["n_candidates"]>0 for r in negative),
                "max_proposals":max([r["n_candidates"] for r in positive],default=0),
            }
    return answer


def execute(memmap_root,frontends_dir,out):
    import torch
    from scripts.tver_v34_anza_p0_relation_cal import _state_report
    from scripts.tver_v33_thebe_train_segmenters import read_memmap
    from scripts.tver_v33_thebe_relation_cal import (
        _candidate_scenes,_load_model,TRAIN_SOURCE_IDS,RELATION_CAL_SOURCE_IDS)
    from scripts.tver_v26_predicted_cracks_e2e import sbpp_proposals,is_true_relation
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    receipt=_state_report(frontends_dir)
    if receipt["frontends"]["anza_v1"]["SHA256"]!=FROZEN_ANZA_SHA:
        raise RuntimeError("Frozen original ANZA checksum mismatch")
    model=_load_model(frontends_dir,"anza_v1",receipt)
    rows=[]
    for cohort,src,mask,sections in (
        ("TRAIN",read_memmap(memmap_root,"train_seismic_float32.npy"),
         read_memmap(memmap_root,"train_fault_bool.npy"),TRAIN_SOURCE_IDS),
        ("REL_CAL_ALREADY_VIEWED",read_memmap(memmap_root,"val_seismic_float32.npy"),
         read_memmap(memmap_root,"val_fault_bool.npy"),RELATION_CAL_SOURCE_IDS),
    ):
        scenes,_=_candidate_scenes(src,mask,sections,receipt["normalization"],model,BASE)
        rows.extend(case_rows(cohort,scenes,sbpp_proposals,is_true_relation))
        print("V43_SOURCE_COHORT",cohort,"n_scenes",len(scenes),flush=True)
    summary=summarize(rows)
    if (summary["TRAIN/baseline_065"]["correct_candidate_available"]!=116
        or summary["REL_CAL_ALREADY_VIEWED/baseline_065"]["correct_candidate_available"]!=65
        or summary["REL_CAL_ALREADY_VIEWED/baseline_065"]["geometry_top1_correct"]!=1
        or summary["REL_CAL_ALREADY_VIEWED/baseline_065"]["original_dense_stops"]!=18):
        raise RuntimeError("Original Thebe/ANZA/SBPP V34 baseline not reproduced")
    report={
        "status":"V43_DENSE_MASK_ADAPTIVE_QUANTILE_POSTHOC_EVALUATED_TEST7_UNOPENED",
        "original_ANZA_SHA256":FROZEN_ANZA_SHA,
        "original_checkpoint_receipt_SHA256":hashlib.sha256(
            (Path(frontends_dir)/"v33_THEBE_frontends_topology_TRAIN_CAL_frozen.json").read_bytes()).hexdigest(),
        "original_threshold":BASE,"fallback_quantiles":[.7,.8,.9],
        "never_read_Thebe_TEST7":True,
        "previously_viewed_REL_CAL":True,
        "no_natural_geological_pair_identity":True,
        "summary":summary,
    }
    (out/"V43_RESULT.json").write_text(json.dumps(report,indent=2)+"\n")
    with (out/"V43_CASE_ROWS.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({"status":report["status"],"summary":summary},indent=2),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mmap-root",type=Path,required=True)
    p.add_argument("--frontends-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    execute(a.mmap_root,a.frontends_dir,a.out_dir)
