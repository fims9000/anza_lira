"""V44: equal-label-budget NO_LINK hard negatives on original real Thebe ANZA.

Exploratory posthoc on already inspected relation-CAL; never open TEST7.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scripts.tver_v30_p0_context import (
    fit_matched_logistic,score_logistic,paired_training_examples)

ANZA_SHA="ba0c3b8060fd0e3cb1c15f6c1598b94a977677c2b075663b5414207df382c25e"
METHODS=("matched_original","mixed_half","mixed_all")

def swap_nolink_negatives(train,features,labels,baseline,mode):
    if mode not in ("half","all"):
        raise ValueError("Unknown negative mix")
    f=np.asarray(features,float);y=np.asarray(labels,int)
    if f.ndim!=2 or f.shape[1]!=12 or len(f)%2 or len(f)<16 or not np.isfinite(f).all():
        raise ValueError("Invalid matched 12-feature pairs")
    if not (np.all(y[::2]==1) and np.all(y[1::2]==0)):
        raise ValueError("Original matched labels/order changed")
    positives=[];negatives=defaultdict(list)
    for scene in train:
        key=(int(scene["section"]),str(scene["id"]))
        for p in scene["proposals"]:
            z=np.asarray(p["features"],float)
            if z.shape!=(12,) or not np.isfinite(z).all():
                raise ValueError("Invalid candidate feature")
            if scene["kind"]=="LINK" and int(p["y"])==1:
                positives.append((key,z))
            elif scene["kind"]=="NO_LINK":
                if int(p["y"])!=0:raise ValueError("Unexpected NO_LINK positive pair")
                negatives[key].append(z)
    if len(positives)*2!=len(f):
        raise ValueError("Original positive pair budget changed")
    out=f.copy();used=defaultdict(int);replaced=missing=0
    for i,(key,z) in enumerate(positives):
        if not np.allclose(z,f[2*i],rtol=1e-9,atol=1e-10):
            raise RuntimeError("Original positive pair order changed")
        if mode=="half" and i%2:
            continue
        pool=negatives[key]
        if not pool:
            missing+=1;continue
        x=np.asarray(pool,float)
        # Hard negative chosen by TRAIN-only original model confidence.
        order=np.argsort(-baseline.predict_proba(x)[:,1],kind="stable")
        out[2*i+1]=x[int(order[used[key]%len(order)])]
        used[key]+=1;replaced+=1
    return out,{"positive_pairs":len(positives),
                "label_budget":len(y),"replaced_no_link_negatives":replaced,
                "missing_same_gap_no_link":missing,
                "baseline_hard_negative_selector_TRAIN_only":True,
                "mode":mode}

def score_cal(scenes,models):
    rows=[]
    for scene in scenes:
        for name,model in models.items():
            scores=score_logistic(model,scene)
            if len(scores)!=len(scene["proposals"]):raise RuntimeError("Candidate length drift")
            k=int(np.argmax(scores)) if len(scores) else None
            rows.append({
                "method":name,"section":int(scene["section"]),
                "gap_id":str(scene["id"]),"world":scene["kind"],
                "top1_score":float(scores[k]) if k is not None else None,
                "top1_correct":int(k is not None and scene["kind"]=="LINK" and scene["proposals"][k]["y"]==1),
                "candidate_available":int(scene["has_true_proposal"]),
                "candidate_count":len(scene["proposals"]),
            })
    return rows

def evaluate(rows):
    out={}
    for name in METHODS:
        rs=[r for r in rows if r["method"]==name]
        pos=[r for r in rs if r["world"]=="LINK"]
        no=[r for r in rs if r["world"]=="NO_LINK"]
        if len(pos)!=120 or len(no)!=120 or len({r["section"] for r in pos})!=60 or sum(r["candidate_available"] for r in pos)!=65:
            raise ValueError("Original Thebe REL_CAL cohort/candidate recall changed")
        best_zero=best_emp=0
        thresholds=sorted({r["top1_score"] for r in rs if r["top1_score"] is not None},reverse=True)
        for t in thresholds:
            accepted=[r for r in rs if r["top1_score"] is not None and r["top1_score"]>=t]
            ncorrect=sum(r["top1_correct"] for r in accepted if r["world"]=="LINK")
            wronglink={r["section"] for r in accepted if r["world"]=="LINK" and not r["top1_correct"]}
            wrongno={r["section"] for r in accepted if r["world"]=="NO_LINK"}
            if not wronglink and not wrongno:best_zero=max(best_zero,ncorrect)
            if len(wronglink)<=6 and len(wrongno)<=6:best_emp=max(best_emp,ncorrect)
        out[name]={"correct_top1":sum(r["top1_correct"] for r in pos),
                   "zero_observed_wrong_sources_best_correct_POSTHOC":best_zero,
                   "both_empirical_errors_le_10pct_best_correct_POSTHOC":best_emp,
                   "unique_thresholds_examined_POSTHOC":len(thresholds)}
    base={r["gap_id"]:r for r in rows if r["world"]=="LINK" and r["method"]=="matched_original"}
    for name in METHODS[1:]:
        alt={r["gap_id"]:r for r in rows if r["world"]=="LINK" and r["method"]==name}
        if set(base)!=set(alt):raise ValueError("Paired case identity mismatch")
        out[name]["only_original_right"]=sum(base[g]["top1_correct"] and not alt[g]["top1_correct"] for g in base)
        out[name]["only_new_right"]=sum(alt[g]["top1_correct"] and not base[g]["top1_correct"] for g in base)
    return out

def execute(memmap_root,frontends_dir,out):
    import torch
    from scripts.tver_v34_anza_p0_relation_cal import _state_report
    from scripts.tver_v33_thebe_train_segmenters import read_memmap
    from scripts.tver_v33_thebe_relation_cal import (
        _candidate_scenes,_load_model,TRAIN_SOURCE_IDS,RELATION_CAL_SOURCE_IDS)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    receipt=_state_report(frontends_dir)
    if receipt["frontends"]["anza_v1"]["SHA256"]!=ANZA_SHA:
        raise RuntimeError("Frozen ANZA SHA mismatch")
    seg=_load_model(frontends_dir,"anza_v1",receipt)
    train,_=_candidate_scenes(read_memmap(memmap_root,"train_seismic_float32.npy"),
                             read_memmap(memmap_root,"train_fault_bool.npy"),
                             TRAIN_SOURCE_IDS,receipt["normalization"],seg,.65)
    if sum(s["has_true_proposal"] for s in train if s["kind"]=="LINK")!=116:
        raise RuntimeError("Original V34 TRAIN baseline not reproduced")
    _,labels,features,passport=paired_training_examples(train,.65)
    baseline=fit_matched_logistic(features,labels)
    models={"matched_original":baseline}
    controls={}
    for key,mode in (("mixed_half","half"),("mixed_all","all")):
        feats,receipt_new=swap_nolink_negatives(train,features,labels,baseline,mode)
        models[key]=fit_matched_logistic(feats,labels)
        controls[key]=receipt_new
    cal,_=_candidate_scenes(read_memmap(memmap_root,"val_seismic_float32.npy"),
                           read_memmap(memmap_root,"val_fault_bool.npy"),
                           RELATION_CAL_SOURCE_IDS,receipt["normalization"],seg,.65)
    rows=score_cal(cal,models)
    metrics=evaluate(rows)
    if metrics["matched_original"]["correct_top1"]!=36:
        raise RuntimeError("Original V34 matched logistic 36/120 baseline not reproduced")
    report={"status":"V44_THEBE_TRAIN_NOLINK_NEGATIVE_ABLATION_POSTHOC_COMPLETE_TEST7_UNOPENED",
            "frozen_ANZA_SHA":ANZA_SHA,"original_pair_receipt":passport,
            "TRAIN_control_receipts":controls,
            "Thebe_TEST7_unopened":True,"REL_CAL_previously_inspected":True,
            "synthetic_NOLINK_not_real_geological_fault_instance_labels":True,
            "CAL_thresholds_POSTHOC_not_for_deployment":True,
            "metrics":metrics}
    (out/"V44_RESULT.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    with (out/"V44_CASE_ROWS.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({"status":report["status"],"metrics":metrics,"TRAIN":controls},indent=2),flush=True)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mmap-root",type=Path,required=True)
    p.add_argument("--frontends-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args();execute(a.mmap_root,a.frontends_dir,a.out_dir)
