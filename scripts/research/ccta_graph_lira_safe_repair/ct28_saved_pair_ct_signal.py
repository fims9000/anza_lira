#!/usr/bin/env python3
"""CCTA feature reliance of PAIR *identity*, using committed CT28 local HGB.

Different question from relation-class sham:
- CT score shuffling at relation HGB stage measures marginal CT-score effect on
  NONE/PAIR/JUNCTION/BOTH classification.
- This experiment shuffles the original 38-dimensional *image-derived CT
  feature vectors* among validation PAIR candidates within each patient and
  re-applies the FROZEN HGB PAIR geometry+CT identity model. All geometric
  features, candidate pools, labels, model parameters and class prevalence
  stay unchanged. It measures the reliance on candidate-specific CT.
No raw CT volumes or held-out patients are opened; no retraining/tuning.
This is a historical committed HGB, NOT Varvara's selected K10/15 pack.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score

from ct28_saved_ct_signal import (
    DEFAULT_MODEL,DEFAULT_SCORES,sha,
)

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_FEATURES=ROOT/"artifacts/varvara/ct28_graph_lira/scene_candidate_features_train_val.csv"
VAL_IDS={957,961,965,966,974}
TEST_IDS={954,958,972,973,980,984}
PAIR_CT_COLUMNS=tuple(f"pair_ct{i}" for i in range(38))


def candidate_metrics(rows:pd.DataFrame,scores:np.ndarray)->dict:
    truth=rows["y"].to_numpy(dtype=np.int8)
    if len(scores)!=len(truth) or not np.isfinite(scores).all():
        raise ValueError("Invalid PAIR predicted score vector")
    if set(truth)!={0,1}:
        raise ValueError("PAIR val must contain both classes")
    predictions=rows[["scene_id","scan_id","y"]].copy()
    predictions["score"]=scores
    best=predictions.loc[
        predictions.groupby("scene_id")["score"].idxmax()
    ]
    positives=predictions.groupby("scene_id")["y"].sum()
    if not positives.isin([0,1]).all():
        raise ValueError("More than one ground-truth PAIR per scene")
    selected=best[best["scene_id"].isin(positives[positives==1].index)]
    return {
        "roc_auc":float(roc_auc_score(truth,scores)),
        "average_precision":float(average_precision_score(truth,scores)),
        "scenes_with_real_pair":len(selected),
        "top1_correct_on_real_pair_scenes":int(selected["y"].sum()),
        "top1_accuracy_on_real_pair_scenes":float(selected["y"].mean()),
        "pair_candidate_count":len(predictions),
        "pair_positive_candidates":int(truth.sum()),
    }


def paired_top1_identity_audit(
    rows:pd.DataFrame,geometry_scores:np.ndarray,CT_fused_scores:np.ndarray,
    *,seed:int=20261103,bootstrap:int=5000,
)->dict:
    """Paired top-1 PAIR identity improvements; five-patient descriptive CI."""
    if bootstrap<100:
        raise ValueError("Need at least 100 patient bootstrap samples")
    pieces=[]
    for name,scores in (
        ("geometry",geometry_scores),("geometry_plus_CT",CT_fused_scores)
    ):
        vals=rows[["scene_id","scan_id","y"]].copy()
        vals["score"]=scores
        best=vals.loc[vals.groupby("scene_id")["score"].idxmax()]
        true=vals.groupby("scene_id")["y"].sum()
        best=best[best.scene_id.isin(true[true==1].index)]
        if len(best)!=170 or set(best.scan_id.astype(int))!=VAL_IDS:
            raise ValueError("PAIR top1 comparison source scene/patient mismatch")
        pieces.append(best.set_index("scene_id")["y"].astype(int).rename(name))
    comparison=pd.concat(pieces,axis=1)
    names=rows[["scene_id","scan_id"]].drop_duplicates("scene_id")
    comparison=comparison.merge(names,on="scene_id",validate="one_to_one")
    if len(comparison)!=170 or comparison[["geometry","geometry_plus_CT"]].isna().any().any():
        raise ValueError("Different PAIR scene sets in paired comparison")
    g=comparison["geometry"].to_numpy(dtype=int)
    c=comparison["geometry_plus_CT"].to_numpy(dtype=int)
    rows_by_patient={
        int(patient):comparison.loc[comparison.scan_id.eq(patient)]
        for patient in sorted(VAL_IDS)
    }
    descriptive=[]
    for patient,frame in rows_by_patient.items():
        n=len(frame)
        descriptive.append({
            "patient_id":patient,
            "true_PAIR_scenes":n,
            "geometry_top1_correct":int(frame.geometry.sum()),
            "geometry_plus_CT_top1_correct":int(frame.geometry_plus_CT.sum()),
            "paired_difference":float((frame.geometry_plus_CT-frame.geometry).mean()),
        })
    rng=np.random.default_rng(seed)
    patients=sorted(VAL_IDS)
    deltas=[]
    for _ in range(bootstrap):
        choices=rng.choice(patients,size=len(patients),replace=True)
        samples=pd.concat([rows_by_patient[int(p)] for p in choices],ignore_index=True)
        deltas.append(float((samples.geometry_plus_CT-samples.geometry).mean()))
    intervals=np.quantile(deltas,[.025,.975]).tolist()
    return {
        "pair_scenes":len(comparison),
        "fusion_correct_geometry_wrong":int(np.sum((c==1)&(g==0))),
        "geometry_correct_fusion_wrong":int(np.sum((c==0)&(g==1))),
        "both_correct":int(np.sum((c==1)&(g==1))),
        "both_wrong":int(np.sum((c==0)&(g==0))),
        "net_improved_pair_scenes":int(c.sum()-g.sum()),
        "top1_absolute_improvement":float(c.mean()-g.mean()),
        "per_patient":descriptive,
        "patient_cluster_descriptive_95pct":[float(v) for v in intervals],
        "bootstrap_seed":seed,
        "bootstrap_repetitions":bootstrap,
        "inference_input_contains_truth":False,
    }


def _shuffle_ct_vectors(
    frame:pd.DataFrame,*,seed:int,
)->pd.DataFrame:
    rng=np.random.default_rng(seed)
    result=frame.copy()
    for patient,part in frame.groupby("scan_id",sort=True):
        idx=part.index.to_numpy()
        if len(idx)>1:
            shuffled=rng.permutation(len(idx))
            result.loc[idx,list(PAIR_CT_COLUMNS)]=part[list(PAIR_CT_COLUMNS)].to_numpy()[shuffled]
    return result


def run_pair_signal(
    *,
    source_features:Path=DEFAULT_FEATURES,
    source_scores:Path=DEFAULT_SCORES,
    source_model:Path=DEFAULT_MODEL,
    repetitions:int=20,
    seed:int=20261103,
)->tuple[dict,pd.DataFrame]:
    if repetitions<5 or repetitions>500:
        raise ValueError("Expected 5..500 fixed within-patient CT sham replications")
    source=pd.read_csv(
        source_features,dtype={"ct_sampling_error":"string"},
        low_memory=False,
    )
    pair=source.loc[
        source.split.eq("val")&source.candidate_type.eq("PAIR")
    ].copy().reset_index(drop=True)
    if len(pair)!=2298 or set(pair.scan_id.astype(int))!=VAL_IDS:
        raise ValueError("Original CT28 val PAIR patient/pool contract mismatch")
    if set(source.loc[source.split.eq("val"),"scan_id"].astype(int))&TEST_IDS:
        raise ValueError("Held-out test CT in saved source")
    if pair[["scene_id","candidate_id"]].duplicated().any():
        raise ValueError("Duplicate PAIR candidate")
    models=joblib.load(source_model)
    geometry=models["local_models"]["PAIR"]["geometry"]
    fusion=models["local_models"]["PAIR"]["geometry_plus_ct"]
    geom_columns=list(geometry.feature_names_in_)
    fusion_columns=list(fusion.feature_names_in_)
    if len(geom_columns)!=7 or len(fusion_columns)!=45:
        raise ValueError("Local HGB PAIR feature schema unexpectedly changed")
    if not set(PAIR_CT_COLUMNS)<=set(fusion_columns):
        raise ValueError("PAIR fused identity missing original 38 CT features")
    if not set(fusion_columns)<=set(pair):
        raise ValueError("Missing saved local PAIR CCTA features")
    original_geo=geometry.predict_proba(pair[geom_columns])[:,1]
    original_fused=fusion.predict_proba(pair[fusion_columns])[:,1]
    saved=pd.read_csv(source_scores)
    saved=saved.loc[
        saved.split.eq("val")&saved.candidate_type.eq("PAIR"),
        ["scene_id","candidate_id","geometry_score","geometry_plus_ct_score"],
    ]
    if len(saved)!=2298 or saved[["scene_id","candidate_id"]].duplicated().any():
        raise ValueError("Missing local-HGB validation PAIR probability evidence")
    checked=pair[["scene_id","candidate_id"]].merge(
        saved,on=["scene_id","candidate_id"],validate="one_to_one"
    )
    if len(checked)!=len(pair):
        raise ValueError("PAIR score evidence contains different candidate pool")
    # The previously published model weights MUST reproduce the stored
    # probabilities before we allow conclusions from CT shuffling.
    max_error=max(
        np.max(np.abs(original_geo-checked.geometry_score.to_numpy())),
        np.max(np.abs(original_fused-checked.geometry_plus_ct_score.to_numpy())),
    )
    if max_error>1e-8:
        raise ValueError(f"Fitted local-HGB model no longer reproduces saved scores: {max_error}")
    original_geo_metrics=candidate_metrics(pair,original_geo)
    original_fusion_metrics=candidate_metrics(pair,original_fused)
    paired_top1=paired_top1_identity_audit(
        pair,original_geo,original_fused,seed=seed,bootstrap=5000,
    )
    rows=[]
    for rep in range(repetitions):
        changed=_shuffle_ct_vectors(pair,seed=seed+rep)
        if not changed[geom_columns].equals(pair[geom_columns]):
            raise ValueError("CT-sham changed PAIR geometry features")
        if not changed[["scene_id","candidate_id","scan_id","y"]].equals(
            pair[["scene_id","candidate_id","scan_id","y"]]
        ):
            raise ValueError("CT-sham changed candidate identities or labels")
        score=fusion.predict_proba(changed[fusion_columns])[:,1]
        result=candidate_metrics(pair,score)
        rows.append({"replication":rep,"seed":seed+rep,**result})
    shams=pd.DataFrame(rows)
    report={
        "status":"committed_historical_local_HGB_pair_identity_CT_feature_sham",
        "model_scope":"historical_saved_LOCAL_HGB_not_selected_K10_15",
        "frozen_Pair_geometry":original_geo_metrics,
        "frozen_Pair_geometry_plus_CT":original_fusion_metrics,
        "paired_top1_identity_comparison":paired_top1,
        "within_patient_CT_vector_shuffling":{
            "count":repetitions,
            "seed_start":seed,
            "mean_AUPRC":float(shams.average_precision.mean()),
            "min_AUPRC":float(shams.average_precision.min()),
            "max_AUPRC":float(shams.average_precision.max()),
            "mean_AUROC":float(shams.roc_auc.mean()),
            "mean_top1_on_pair_scenes":float(shams.top1_accuracy_on_real_pair_scenes.mean()),
            "mean_AUPRC_drop_vs_fusion":float(
                original_fusion_metrics["average_precision"]-shams.average_precision.mean()
            ),
            "shams_with_AUPRC_at_least_original":int(
                np.sum(shams.average_precision>=original_fusion_metrics["average_precision"])
            ),
        },
        "CT_sham_policy":"reassign entire 38-feature CT vector within each validation patient, keep PAIR geometry, labels and model frozen",
        "sha256":{
            "candidate_features":sha(source_features),
            "saved_HGB_local_scores":sha(source_scores),
            "saved_HGB_model_pack":sha(source_model),
        },
        "held_out_test_accessed":False,
        "warnings":[
            "Validation reused for model development; these results are exploratory.",
            "This is candidate identity on artificial scene candidate pools, not clinical 3D masks.",
            "Pair CT shuffling may create physically inconsistent CT+geometry examples.",
            "Only the PAIR fused local HGB head is tested here, not selected K10/15.",
        ],
    }
    return report,shams


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--repetitions",type=int,default=20)
    p.add_argument("--seed",type=int,default=20261103)
    args=p.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("Immutable PAIR CT-sham output exists")
    report,replicates=run_pair_signal(repetitions=args.repetitions,seed=args.seed)
    args.out_dir.mkdir(parents=True,exist_ok=False)
    (args.out_dir/"pair_ct_identity_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    replicates.to_csv(args.out_dir/"pair_ct_identity_sham_replicates.csv",index=False)
    print(json.dumps({
        "status":report["status"],
        "geometry":report["frozen_Pair_geometry"],
        "geometry_plus_CT":report["frozen_Pair_geometry_plus_CT"],
        "shams":report["within_patient_CT_vector_shuffling"],
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
