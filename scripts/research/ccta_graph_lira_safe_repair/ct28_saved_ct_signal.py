#!/usr/bin/env python3
"""Real saved-CT-score reliance diagnostic, no medical volumes or model tuning.

This negative-control experiment uses the *committed*, HISTORICAL local-HGB
CT28 relation head, and the already saved per-candidate out-of-fold scores.
It is deliberately not the UNTRACKED selected K10/15 HGB (52/206) pack.

The key perturbation shuffles each validation subject's CT candidate scores
within the same candidate type, leaving scene counts, patient score
distributions, geometry/PAIR fusion scores, model weights and identities
unchanged. Candidate-to-image alignment is broken. The statistic is raw
relation-class accuracy or log loss on the same 368 validation scenes.

No ImageCAS volume, test label, training or threshold search occurs here.
A CT benefit suggested by this test still needs per-patient, anatomically
plausible natural-mask and independent test confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

import run_ct28_graph_lira as graph
from ct28_cardinality_benchmark import load_frozen, CLASSES

ROOT=Path(__file__).resolve().parents[3]
ROOT_ARTIFACTS=ROOT/"artifacts/varvara/ct28_graph_lira"
DEFAULT_MODEL=ROOT_ARTIFACTS/"ablations/local_hgb/relation_models.joblib"
DEFAULT_SCORES=ROOT_ARTIFACTS/"ablations/local_hgb/relation_local_scores_train_val.csv"
DEFAULT_CANDIDATES=ROOT_ARTIFACTS/"scene_candidates_train_val.csv"
DEFAULT_PLAN=ROOT_ARTIFACTS/"scene_plan_train_val.csv"
EXPECTED_VALIDATION_IDS=frozenset({957,961,965,966,974})
EXPECTED_VAL_SCENES=368
EXPECTED_CANDIDATES=4071
EXPECTED_RELATION_CLASSES=tuple(CLASSES)


def sha(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def _calculate(
    *,
    plans:pd.DataFrame,
    candidates:pd.DataFrame,
    candidate_scores:pd.DataFrame,
    model:object,
    columns:list[str],
)->tuple[dict, pd.DataFrame]:
    rows,features=graph.relation_feature_rows(
        plans,candidates,candidate_scores,("geometry_score","ct_score")
    )
    if features!=columns or len(features)!=84:
        raise ValueError("Saved historical HGB relation feature schema mismatch")
    frame=pd.DataFrame(rows)
    if len(frame)!=EXPECTED_VAL_SCENES or frame.scene_id.duplicated().any():
        raise ValueError("Incomplete validation scene feature rows")
    if set(frame.scan_id.astype(int))!=EXPECTED_VALIDATION_IDS:
        raise ValueError("Out-of-cohort patient in relation features")
    x=frame[columns].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite engineered scene features")
    probabilities=model.predict_proba(x)
    if probabilities.shape!=(EXPECTED_VAL_SCENES,4) or not np.isfinite(probabilities).all():
        raise ValueError("HGB probabilities have wrong dimensions")
    if not np.array_equal(model.classes_,np.arange(4)):
        raise ValueError("HGB class order changed")
    if not np.allclose(probabilities.sum(axis=1),1.,atol=1e-7):
        raise ValueError("HGB probabilities no longer sum to 1")
    expected={label:i for i,label in enumerate(EXPECTED_RELATION_CLASSES)}
    y=np.array([expected[str(x)] for x in frame.relation_class],dtype=int)
    prediction=probabilities.argmax(axis=1)
    raw_acc=float(accuracy_score(y,prediction))
    thresholded=np.where(probabilities.max(axis=1)>=.85,prediction,0)
    sample=frame[["scene_id","scan_id","relation_class"]].copy()
    sample["raw_predicted_relation"]=[
        EXPECTED_RELATION_CLASSES[i] for i in prediction
    ]
    sample["raw_correct"]=(y==prediction).astype(int)
    sample["thresholded_correct"]=(y==thresholded).astype(int)
    sample["log_loss_each"]=-np.log(
        np.maximum(probabilities[np.arange(len(y)),y],1e-15)
    )
    report={
        "raw_accuracy":raw_acc,
        "balanced_accuracy":float(balanced_accuracy_score(y,prediction)),
        "thresholded_accuracy":float(accuracy_score(y,thresholded)),
        "log_loss":float(log_loss(y,probabilities,labels=[0,1,2,3])),
        "correct_scenes":int(np.sum(y==prediction)),
        "validation_scenes":len(y),
    }
    return report,sample


def _permuted_ct_scores(
    original:pd.DataFrame,
    *,
    seed:int,
    mode:str="within_patient_type",
)->pd.DataFrame:
    if mode not in {"within_patient_type","global_within_type"}:
        raise ValueError("Unknown CT negative control")
    result=original.copy()
    group_cols=["candidate_type"] if mode=="global_within_type" else ["scan_id","candidate_type"]
    generator=np.random.default_rng(seed)
    for _,part in original.groupby(group_cols,sort=True,dropna=False):
        idx=part.index.to_numpy()
        if len(idx)>1:
            result.loc[idx,"ct_score"]=generator.permutation(
                part["ct_score"].to_numpy(copy=True)
            )
    return result


def run_signal_diagnostic(
    *,
    model_file:Path=DEFAULT_MODEL,
    score_file:Path=DEFAULT_SCORES,
    candidate_file:Path=DEFAULT_CANDIDATES,
    plan_file:Path=DEFAULT_PLAN,
    repetitions:int=30,
    seed:int=20261103,
)->tuple[dict,pd.DataFrame,pd.DataFrame]:
    if repetitions<5 or repetitions>1000:
        raise ValueError("Require 5..1000 fixed CT-sham replications")
    train,val=load_frozen(plan_file)
    if set(train.scan_id.astype(int)) & set(val.scan_id.astype(int)):
        raise ValueError("Patient leakage across train/validation")
    expected_val=set(val.scene_id)
    candidates=pd.read_csv(candidate_file)
    candidates=candidates.loc[
        candidates.scene_id.isin(expected_val),
        ["scene_id","candidate_id","candidate_type"],
    ].copy()
    if len(candidates)!=EXPECTED_CANDIDATES:
        raise ValueError("Wrong number of frozen val candidates")
    if candidates[["scene_id","candidate_id"]].duplicated().any():
        raise ValueError("Duplicate val scene-candidate key")
    scores=pd.read_csv(score_file)
    val_scores=scores.loc[scores.split.eq("val")].copy()
    if len(val_scores)!=EXPECTED_CANDIDATES:
        raise ValueError("Missing CT28 validation local-HGB scores")
    if val_scores[["scene_id","candidate_id"]].duplicated().any():
        raise ValueError("Duplicate candidate score identity")
    if set(val_scores.scan_id.astype(int))!=EXPECTED_VALIDATION_IDS:
        raise ValueError("Saved score source contains held-out patient")
    joined=val_scores.merge(
        val[["scene_id","scan_id","split"]],
        on=["scene_id","scan_id","split"],validate="many_to_one"
    )
    if len(joined)!=len(val_scores):
        raise ValueError("CT28 local score rows mismatch scene-plan split")
    if set(map(tuple,joined[["scene_id","candidate_id"]].to_numpy())) != set(map(tuple,candidates[["scene_id","candidate_id"]].to_numpy())):
        raise ValueError("Saved score/candidate keys don't match")
    if not np.isfinite(val_scores[["geometry_score","ct_score","geometry_plus_ct_score"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite candidate local scores")
    model_pack=joblib.load(model_file)
    model=model_pack["relation_heads"]["ct_conditioned"]
    columns=list(model_pack["ct_conditioned_relation_features"])
    baseline,baseline_rows=_calculate(
        plans=val,candidates=candidates,
        candidate_scores=val_scores,
        model=model,columns=columns,
    )
    all_runs=[]
    for repeat in range(repetitions):
        sham=_permuted_ct_scores(
            val_scores,seed=seed+repeat,mode="within_patient_type"
        )
        if not np.array_equal(
            sham[["scene_id","candidate_id","geometry_score","geometry_plus_ct_score"]].to_numpy(),
            val_scores[["scene_id","candidate_id","geometry_score","geometry_plus_ct_score"]].to_numpy(),
        ):
            raise ValueError("CT-only sham unexpectedly changed geometry or identity")
        if not np.allclose(
            val_scores.groupby(["scan_id","candidate_type"])["ct_score"].sum().sort_index().to_numpy(),
            sham.groupby(["scan_id","candidate_type"])["ct_score"].sum().sort_index().to_numpy(),
            atol=1e-9,rtol=0,
        ):
            raise ValueError("CT shuffle did not preserve patient-and-kind score distribution")
        metrics,scene=_calculate(
            plans=val,candidates=candidates,
            candidate_scores=sham,
            model=model,columns=columns,
        )
        if set(scene.scene_id)!=set(baseline_rows.scene_id):
            raise RuntimeError("CT shuffled or removed scenes")
        all_runs.append({"replication":repeat,"seed":seed+repeat,**metrics})
    runs=pd.DataFrame(all_runs)
    diff=runs["raw_accuracy"].to_numpy()-baseline["raw_accuracy"]
    report={
        "status":"historical_local_HGB_CT_candidate_alignment_sham_development_val",
        "model_scope":"committed_historical_local_HGB_NOT_selected_K10_15",
        "n_validation_patients":len(EXPECTED_VALIDATION_IDS),
        "validation_patients":sorted(EXPECTED_VALIDATION_IDS),
        "n_validation_scenes":EXPECTED_VAL_SCENES,
        "n_validation_candidates":EXPECTED_CANDIDATES,
        "baseline":baseline,
        "CT_sham":{
            "type":"within_patient_and_candidate_type_score_permutation",
            "replications":repetitions,
            "seed_start":seed,
            "accuracy_mean":float(runs.raw_accuracy.mean()),
            "accuracy_min":float(runs.raw_accuracy.min()),
            "accuracy_max":float(runs.raw_accuracy.max()),
            "log_loss_mean":float(runs.log_loss.mean()),
            "difference_sham_minus_baseline_accuracy_mean":float(np.mean(diff)),
            "sham_at_least_baseline_accuracy_count":int(np.sum(diff>=0)),
        },
        "files_sha256":{
            "historical_HGB_relation_model":sha(model_file),
            "historical_HGB_score_frame":sha(score_file),
            "scene_candidates":sha(candidate_file),
            "scene_plan":sha(plan_file),
        },
        "held_out_test_accessed":False,
        "truth_used_for_sham_permutation":False,
        "warnings":[
            "This is the historical committed local-HGB, not selected K10/15.",
            "Validation was used in broader model selection; all intervals/descriptions exploratory.",
            "Preserving per-patient/type score distribution is not proof of medical anatomical specificity.",
            "Only relation-class scores are assessed, not selected candidate identity, path or repaired mask.",
        ],
    }
    return report,runs,baseline_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--repetitions",type=int,default=30)
    p.add_argument("--seed",type=int,default=20261103)
    args=p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Immutable CT signal output already exists")
    report,runs,scenes=run_signal_diagnostic(
        repetitions=args.repetitions,seed=args.seed
    )
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/"ct_signal_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    runs.to_csv(args.output_dir/"ct_sham_replicates.csv",index=False)
    scenes.to_csv(args.output_dir/"baseline_scene_predictions.csv",index=False)
    print(json.dumps({
        "status":report["status"],
        "baseline":report["baseline"],
        "CT_sham":report["CT_sham"],
        "held_out_test_accessed":False,
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
