#!/usr/bin/env python3
"""Fair 84-column HGB ablation on SAVED real CCTA-derived CT28 OOF scores.

No medical images, labels from held-out patients, K10/15 weights, or
clinical segmentation masks are loaded. These are the committed historical
local-HGB scores: geometry_score and ct_score each learned on patient-OOF
train candidates and from full train models for the 5 validation subjects.

Unlike merely comparing 42-column geometry_only vs 84-column CT_conditioned,
this experiment holds the SAME 84-column relation feature schema, HGB
hyperparameters, train/val patient split, sample weighting and random seed:
* real_CT: native geometry and CT scores;
* geometry_duplicate: CT score replaced by identical geometry score;
* constant_CT: CT score constant 0.5 per candidate;
* sham_CT (N replications): CT scores randomly permuted within each patient
  and candidate type independently for train and validation, never by label.
Report both relation-class log loss and raw/tau accuracy, paired per-patient.

This does NOT isolate all CT use in the full Graph-LIRA because PAIR identity
also depends on geometry_plus_ct_score. It does NOT evaluate ANZA or K10/15.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,balanced_accuracy_score,f1_score,log_loss,
)

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import run_ct28_graph_lira as graph
from ct28_cardinality_benchmark import load_frozen,TRAIN_IDS,VAL_IDS,CLASSES

ARTIFACTS=ROOT/"artifacts/varvara/ct28_graph_lira"
MODEL=ARTIFACTS/"ablations/local_hgb/relation_models.joblib"
SCORES=ARTIFACTS/"ablations/local_hgb/relation_local_scores_train_val.csv"
CANDIDATES=ARTIFACTS/"scene_candidates_train_val.csv"
PLAN=ARTIFACTS/"scene_plan_train_val.csv"
TEST_IDS={954,958,972,973,980,984}
VALIDATION_SCENES=368
TRAIN_SCENES=1036
TOTAL_CANDIDATES=15092

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for piece in iter(lambda:f.read(1024*1024),b""):
            h.update(piece)
    return h.hexdigest()


def checked_inputs(*,model_file:Path,score_file:Path,candidate_file:Path,plan_file:Path):
    train_plan,val_plan=load_frozen(plan_file)
    plans=pd.concat((train_plan,val_plan),ignore_index=True)
    candidates=pd.read_csv(candidate_file,low_memory=False)
    needed={"scene_id","candidate_id","candidate_type"}
    if not needed<=set(candidates) or len(candidates)!=TOTAL_CANDIDATES:
        raise ValueError("Historical CT28 candidate table changed")
    if candidates[["scene_id","candidate_id"]].duplicated().any():
        raise ValueError("Duplicate CCTA candidate scene+id")
    scores=pd.read_csv(score_file)
    required=needed|{"geometry_score","ct_score","geometry_plus_ct_score",
                     "scan_id","split","local_oof_fold"}
    if not required<=set(scores) or len(scores)!=TOTAL_CANDIDATES:
        raise ValueError("Historical CT28 OOF score table changed")
    if scores[["scene_id","candidate_id"]].duplicated().any():
        raise ValueError("Duplicate CCTA score scene+id")
    joined=scores.merge(
        plans[["scene_id","scan_id","split"]],
        on=["scene_id","scan_id","split"],validate="many_to_one",
    )
    if len(joined)!=len(scores):
        raise ValueError("Score input mismatches exact patient split")
    if not set(scores.scan_id.astype(int))<=TRAIN_IDS|VAL_IDS or (
        set(scores.scan_id.astype(int))&TEST_IDS
    ):
        raise ValueError("Held-out CT28 patient appeared")
    if set(scores.loc[scores.split.eq("train"),"scan_id"].astype(int))!=TRAIN_IDS:
        raise ValueError("Missing training patient score")
    if set(scores.loc[scores.split.eq("val"),"scan_id"].astype(int))!=VAL_IDS:
        raise ValueError("Missing validation patient score")
    if (
        scores.loc[scores.split.eq("train"),"local_oof_fold"].lt(0).any()
        or scores.loc[scores.split.eq("val"),"local_oof_fold"].ne(-1).any()
    ):
        raise ValueError("Historical train OOF vs validation score lineage changed")
    if not np.isfinite(scores[["geometry_score","ct_score","geometry_plus_ct_score"]].to_numpy()).all():
        raise ValueError("Nonfinite saved CT28 candidate scores")
    if set(map(tuple,candidates[["scene_id","candidate_id"]].to_numpy()))!=set(
        map(tuple,scores[["scene_id","candidate_id"]].to_numpy())
    ):
        raise ValueError("Candidate score keys do not match frozen scenes")
    model_pack=joblib.load(model_file)
    if len(model_pack["ct_conditioned_relation_features"])!=84:
        raise ValueError("Frozen 84-column CT28 relation schema mismatch")
    if set(model_pack["ct_conditioned_relation_features"]) != set(
        graph.relation_feature_rows(
            plans,candidates,scores,("geometry_score","ct_score")
        )[1]
    ):
        raise ValueError("Unexpected 84 feature column names")
    return train_plan,val_plan,candidates,scores,model_pack


def transform_scores(original:pd.DataFrame,variant:str,seed:int=0)->pd.DataFrame:
    if variant not in {"real_CT","geometry_duplicate","constant_CT","sham_CT"}:
        raise ValueError("Unknown evidence-ablation variant")
    out=original.copy()
    if variant=="geometry_duplicate":
        out["ct_score"]=out["geometry_score"].to_numpy(copy=True)
    elif variant=="constant_CT":
        out["ct_score"]=.5
    elif variant=="sham_CT":
        rng=np.random.default_rng(seed)
        for (_,_,_),subset in original.groupby(
            ["split","scan_id","candidate_type"],sort=True
        ):
            indices=subset.index.to_numpy()
            if len(indices)>1:
                out.loc[indices,"ct_score"]=rng.permutation(
                    subset.ct_score.to_numpy(copy=True)
                )
    if not (
        original[["scene_id","candidate_id","scan_id","split",
                  "geometry_score","geometry_plus_ct_score"]].equals(
            out[["scene_id","candidate_id","scan_id","split",
                 "geometry_score","geometry_plus_ct_score"]]
        )
    ):
        raise ValueError("Ablation changed candidate geometry or identity provenance")
    if variant=="sham_CT":
        a=original.groupby(["split","scan_id","candidate_type"]).ct_score.agg(["count","sum"])
        b=out.groupby(["split","scan_id","candidate_type"]).ct_score.agg(["count","sum"])
        if not np.array_equal(a["count"],b["count"]) or not np.allclose(
            a["sum"],b["sum"],rtol=0,atol=1e-9
        ):
            raise ValueError("CT sham changed patient/kind signal distribution")
    return out


def fit_relation(
    *,
    plans:pd.DataFrame,candidates:pd.DataFrame,
    scores:pd.DataFrame,head:object,columns:list[str],
    variant:str,replication:int|None,
)->tuple[dict,pd.DataFrame]:
    rows,cols=graph.relation_feature_rows(
        plans,candidates,scores,("geometry_score","ct_score")
    )
    if cols!=columns:
        raise ValueError("Any ablation must keep the EXACT 84 feature positions")
    features=pd.DataFrame(rows)
    if len(features)!=TRAIN_SCENES+VALIDATION_SCENES:
        raise ValueError("Missing CT28 scene rows")
    x=features[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite relation features after ablation")
    labels=np.asarray([CLASSES.index(x) for x in features.relation_class],dtype=int)
    tr=features.split.eq("train").to_numpy()
    val=features.split.eq("val").to_numpy()
    if tr.sum()!=TRAIN_SCENES or val.sum()!=VALIDATION_SCENES:
        raise ValueError("Train/val scene count changed")
    started=perf_counter()
    model=clone(head)
    model.fit(x[tr],labels[tr])
    seconds=perf_counter()-started
    if not np.array_equal(model.classes_,np.arange(4)):
        raise ValueError("HGB relation class order changed")
    p=model.predict_proba(x[val])
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1.):
        raise ValueError("Invalid HGB relation probabilities")
    y=labels[val]
    predicted=np.argmax(p,axis=1)
    tau=np.where(p.max(axis=1)>=.85,predicted,0)
    report={
        "variant":variant,
        "replication":replication,
        "raw_accuracy":float(accuracy_score(y,predicted)),
        "balanced_accuracy":float(balanced_accuracy_score(y,predicted)),
        "macro_f1":float(f1_score(y,predicted,labels=[0,1,2,3],average="macro",zero_division=0)),
        "tau_085_accuracy":float(accuracy_score(y,tau)),
        "log_loss":float(log_loss(y,p,labels=[0,1,2,3])),
        "correct_raw":int(np.sum(y==predicted)),
        "train_seconds":float(seconds),
        "train_scenes":int(np.sum(tr)),
        "val_scenes":int(np.sum(val)),
        "relation_features":len(cols),
    }
    val_rows=features.loc[val,["scene_id","scan_id","relation_class"]].copy()
    val_rows["variant"]=variant
    val_rows["replication"]=-1 if replication is None else int(replication)
    val_rows["predicted_relation"]=[CLASSES[i] for i in predicted]
    val_rows["raw_correct"]=(predicted==y).astype(int)
    val_rows["tau_correct"]=(tau==y).astype(int)
    val_rows["sample_log_loss"]=-np.log(np.maximum(p[np.arange(len(y)),y],1e-15))
    return report,val_rows


def patient_paired_bootstrap(
    *,
    base:pd.DataFrame,comparison:pd.DataFrame,
    n:int=4000,seed:int=20261103,
)->dict:
    if n<100:
        raise ValueError("Require 100+ prespecified patient bootstrap draws")
    b=base[["scene_id","scan_id","raw_correct","sample_log_loss"]]
    c=comparison[["scene_id","scan_id","raw_correct","sample_log_loss"]]
    m=b.merge(c,on=["scene_id","scan_id"],validate="one_to_one",suffixes=("_full","_control"))
    if len(m)!=VALIDATION_SCENES or set(m.scan_id.astype(int))!=VAL_IDS:
        raise ValueError("Patient-matched comparison lost validation scenes")
    patients=sorted(VAL_IDS)
    vals={int(pid):m.loc[m.scan_id.eq(pid)] for pid in patients}
    per=[]
    for pid,subset in vals.items():
        per.append({
            "patient":pid,"scenes":len(subset),
            "delta_raw_accuracy_full_minus_control":float(
                (subset.raw_correct_full-subset.raw_correct_control).mean()
            ),
            "delta_log_loss_control_minus_full":float(
                (subset.sample_log_loss_control-subset.sample_log_loss_full).mean()
            ),
        })
    mean_delta={
        "full_minus_control_accuracy":float((m.raw_correct_full-m.raw_correct_control).mean()),
        "control_minus_full_log_loss":float((m.sample_log_loss_control-m.sample_log_loss_full).mean()),
    }
    rng=np.random.default_rng(seed)
    intervals={}
    for metric,a,b in (
        ("accuracy","raw_correct_full","raw_correct_control"),
        ("log_loss_improvement","sample_log_loss_control","sample_log_loss_full")
    ):
        samples=[]
        for _ in range(n):
            chosen=rng.choice(patients,size=len(patients),replace=True)
            frames=[vals[int(pid)] for pid in chosen]
            numerator=sum((x[a]-x[b]).sum() for x in frames)
            denominator=sum(len(x) for x in frames)
            samples.append(float(numerator/denominator))
        intervals[metric]=[float(x) for x in np.quantile(samples,[.025,.975])]
    return {
        "micro_paired_differences":mean_delta,
        "patient_cluster_descriptive_95pct":intervals,
        "per_patient":per,
        "patient_bootstrap_draws":n,
        "patient_bootstrap_seed":seed,
    }


def run_study(
    *,
    model_file:Path=MODEL,
    score_file:Path=SCORES,
    candidate_file:Path=CANDIDATES,
    plan_file:Path=PLAN,
    sham_repetitions:int=12,
    bootstrap:int=2000,
    seed:int=20261103,
)->tuple[dict,pd.DataFrame,pd.DataFrame]:
    if sham_repetitions<3 or sham_repetitions>100:
        raise ValueError("Require 3..100 prespecified CT-sham trials")
    t,v,c,s,pack=checked_inputs(
        model_file=model_file,score_file=score_file,
        candidate_file=candidate_file,plan_file=plan_file
    )
    plans=pd.concat((t,v),ignore_index=True)
    head=pack["relation_heads"]["ct_conditioned"]
    columns=list(pack["ct_conditioned_relation_features"])
    reports=[]
    scene_frames={}
    raw=[]
    for variant in ("real_CT","geometry_duplicate","constant_CT"):
        result,scenes=fit_relation(
            plans=plans,candidates=c,
            scores=transform_scores(s,variant),
            head=head,columns=columns,variant=variant,replication=None
        )
        reports.append(result)
        scene_frames[variant]=scenes
        raw.append(scenes)
    for i in range(sham_repetitions):
        result,scenes=fit_relation(
            plans=plans,candidates=c,
            scores=transform_scores(s,"sham_CT",seed=seed+i),
            head=head,columns=columns,variant="sham_CT",replication=i
        )
        reports.append(result)
        raw.append(scenes)
    table=pd.DataFrame(reports)
    predictions=pd.concat(raw,ignore_index=True)
    primary=scene_frames["real_CT"]
    paired={}
    for name in ("geometry_duplicate","constant_CT"):
        paired[name]=patient_paired_bootstrap(
            base=primary,comparison=scene_frames[name],
            n=bootstrap,seed=seed
        )
    shams=table.loc[table.variant.eq("sham_CT")]
    report={
        "status":"historical_HGB_84_feature_matched_CT_reliance_development_val",
        "method":"refit_same_HGB_hyperparameters_every_variant",
        "historical_scope":"original local HGB OOF candidate scores, NOT selected K10/15",
        "same_feature_count_all":84,
        "train_scenes":TRAIN_SCENES,
        "val_scenes":VALIDATION_SCENES,
        "train_patients":sorted(TRAIN_IDS),
        "val_patients":sorted(VAL_IDS),
        "sham_repetitions":sham_repetitions,
        "seed":seed,
        "variants":{
            result["variant"]:result for result in reports
            if result["variant"]!="sham_CT"
        },
        "sham_summary":{
            "mean_accuracy":float(shams.raw_accuracy.mean()),
            "std_accuracy":float(shams.raw_accuracy.std(ddof=1)),
            "min_accuracy":float(shams.raw_accuracy.min()),
            "max_accuracy":float(shams.raw_accuracy.max()),
            "mean_log_loss":float(shams.log_loss.mean()),
            "count_sham_raw_accuracy_at_least_real":int(
                (shams.raw_accuracy>=reports[0]["raw_accuracy"]).sum()
            )
        },
        "paired_full_CT_vs_controls":paired,
        "source_sha256":{
            "model":sha(model_file),"local_scores":sha(score_file),
            "candidates":sha(candidate_file),"plans":sha(plan_file)
        },
        "versions":{"sklearn":sklearn.__version__},
        "held_out_test_accessed":False,
        "warnings":[
            "CT-only ablation is 84 columns throughout, but duplicate/constant columns have lower effective capacity.",
            "Re-fitting HGB after nulling CT scores tests whether CT is needed for relation classification; it does not isolate CT use in PAIR identity/path.",
            "Validation patients already participated in broader model-selection; all improvements and intervals descriptive.",
            "No medical CCTA volumes, final segmentations, new ANZA or CNN checkpoints are evaluated.",
        ]
    }
    return report,table,predictions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--sham-repetitions",type=int,default=12)
    p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--seed",type=int,default=20261103)
    a=p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError("Matched CT relation study output must be create-only")
    report,comparisons,scenes=run_study(
        sham_repetitions=a.sham_repetitions,bootstrap=a.bootstrap,seed=a.seed
    )
    a.output_dir.mkdir(parents=True,exist_ok=False)
    (a.output_dir/"research_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n"
    )
    comparisons.to_csv(a.output_dir/"variant_metrics.csv",index=False)
    scenes.to_csv(a.output_dir/"validation_predictions.csv",index=False)
    print(json.dumps({
        "status":report["status"],"variants":report["variants"],
        "sham_summary":report["sham_summary"],
        "held_out_test_accessed":False
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
