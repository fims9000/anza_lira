#!/usr/bin/env python3
"""Real CT28 candidate-level image-signal experiment from *committed* CCTA features.

No private CCTA volumes, image downloads, model weights or held-out CT28 test
subjects are opened. The 15,092 precomputed candidate rows contain real
image-derived CT features and a fixed 17-train/5-development-validation split.

Compare matched-family geometry, CT, geometry+CT, and geometry+patient-shuffled
CT predictors for both PAIR and JUNCTION, evaluating exact patient-level
validation. The shuffled-CT arm is a negative control, not a deployable model.

This study is candidate-level AP/AUROC, NOT end-to-end 3D mask reconstruction,
NOT a proof of ANZA, and NOT a new independent blind test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

REPO=Path(__file__).resolve().parents[3]
DEFAULT_SOURCE=REPO/"artifacts/varvara/ct28_graph_lira/scene_candidate_features_train_val.csv"
TRAIN={953,955,956,959,960,963,964,967,969,970,971,975,976,977,979,982,983}
VAL={957,961,965,966,974}
TEST={954,958,972,973,980,984}
JUNCTION_GEO=(
    "junction_span_mm","junction_residual_mean_mm",
    "junction_residual_max_mm","junction_forward_min_mm",
    "junction_forward_mean_mm","junction_alignment_min",
    "junction_alignment_mean","junction_center_distance_max_mm",
    "junction_center_distance_std_mm",
)
KINDS=("PAIR","JUNCTION")
VARIANTS=("geometry_only","ct_only","geometry_plus_ct","geometry_plus_shuffled_ct")
DEFAULT_HGB={
    "max_iter":100,
    "max_leaf_nodes":15,
    "learning_rate":.06,
    "l2_regularization":2.,
    "early_stopping":False,
    "class_weight":"balanced",
    "random_state":0,
}


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda:source.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def expected_columns(frame:pd.DataFrame,kind:str)->tuple[list[str],list[str]]:
    if kind=="PAIR":
        geometry=[f"pair_g{i}" for i in range(7)]
        ct=[f"pair_ct{i}" for i in range(38)]
    elif kind=="JUNCTION":
        geometry=list(JUNCTION_GEO)
        ct=[
            name for name in frame
            if name.startswith("junction_") and name not in JUNCTION_GEO
        ]
    else:
        raise ValueError("PAIR or JUNCTION required")
    if len(ct)!=(38 if kind=="PAIR" else 152):
        raise ValueError("Frozen CT28 candidate feature schema changed")
    if not set(geometry+ct)<=set(frame):
        raise ValueError("Missing geometry/image features")
    if len(set(geometry+ct))!=len(geometry+ct):
        raise ValueError("Unexpected duplicate CT28 input feature")
    return geometry,ct


def read_frozen(path:Path)->pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError("Missing committed CT28 CCTA feature artifact")
    frame=pd.read_csv(path)
    required={"scene_id","candidate_id","candidate_type","scan_id","split","y"}
    if not required<=set(frame):
        raise ValueError("Candidate source lacks identities, patient split or targets")
    if len(frame)!=15092:
        raise ValueError("Expected exactly 15,092 frozen CT28 candidates")
    if frame[["scene_id","candidate_id"]].duplicated().any():
        raise ValueError("Duplicate structural candidates")
    if set(frame.candidate_type)!=set(KINDS):
        raise ValueError("Unexpected candidate kinds")
    if set(frame.y.dropna().astype(int))!={0,1} or frame.y.isna().any():
        raise ValueError("Candidate labels are not strictly binary")
    if set(frame.split)!={"train","val"}:
        raise ValueError("Unexpected CT28 split")
    if set(frame.loc[frame.split=="train","scan_id"].astype(int))!=TRAIN:
        raise ValueError("Train patients changed; test firewall")
    if set(frame.loc[frame.split=="val","scan_id"].astype(int))!=VAL:
        raise ValueError("Val patients changed; test firewall")
    if set(frame.scan_id.astype(int))&TEST:
        raise ValueError("Held-out patient present in feature source")
    if not all(frame.loc[frame.candidate_type==kind,"y"].nunique()==2 for kind in KINDS):
        raise ValueError("Missing positive/negative class")
    return frame


def make_matrix(
    frame:pd.DataFrame,
    *,
    geometry:list[str],ct:list[str],
    variant:str,seed:int,
)->np.ndarray:
    if variant not in VARIANTS:
        raise ValueError("Unknown ablation variant")
    selected={
        "geometry_only":geometry,
        "ct_only":ct,
        "geometry_plus_ct":geometry+ct,
        "geometry_plus_shuffled_ct":geometry+ct,
    }[variant]
    x=frame[selected].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=np.float64)
    if np.isinf(x).any():
        raise ValueError("Infinite values in frozen CT28 evidence")
    if variant=="geometry_plus_shuffled_ct":
        # One *joint* permutation of all CT columns per patient (never
        # individual shuffles per feature). Preserve patient-specific CT
        # distributions and all geometry rows while breaking candidate/CT
        # spatial correspondence. Labels are NEVER used for shuffling.
        ct_start=len(geometry)
        groups=frame["scan_id"].to_numpy()
        for patient in np.unique(groups):
            row_ids=np.flatnonzero(groups==patient)
            rng=np.random.default_rng(seed+int(patient))
            permutation=rng.permutation(len(row_ids))
            x[row_ids,ct_start:]=x[row_ids[permutation],ct_start:]
    return x


def fit_predict(
    train_x:np.ndarray,train_y:np.ndarray,val_x:np.ndarray,
    *,max_iter:int=100,
)->tuple[np.ndarray,float]:
    if np.unique(train_y).size!=2:
        raise ValueError("Need both classes in training")
    classifier=HistGradientBoostingClassifier(
        **{**DEFAULT_HGB,"max_iter":max_iter}
    )
    start=time.perf_counter()
    classifier.fit(train_x,train_y)
    result=classifier.predict_proba(val_x)[:,1].astype(float)
    duration=time.perf_counter()-start
    if not np.isfinite(result).all() or not np.all((result>=0)&(result<=1)):
        raise ValueError("Invalid probability prediction")
    return result,duration


def metrics(y:np.ndarray,p:np.ndarray)->dict:
    positive=int(y.sum())
    result={
        "n":len(y),"positives":positive,
        "prevalence":positive/len(y) if len(y) else None,
        "auroc":None,"average_precision":None,
    }
    if len(np.unique(y))==2:
        result["auroc"]=float(roc_auc_score(y,p))
        result["average_precision"]=float(average_precision_score(y,p))
    return result


def patient_macro_bootstrap(
    rows:list[dict],kind:str,comparison:tuple[str,str],
    *,reps:int,seed:int,
)->dict:
    """Paired patient-macro AP/AUROC differences, descriptive with N=5."""
    left,right=comparison
    patients=sorted(VAL)
    values=[]
    per={}
    for p in patients:
        group=[r for r in rows if r["candidate_type"]==kind and r["scan_id"]==p]
        y=np.asarray([r["y"] for r in group],dtype=int)
        a=np.asarray([r[left] for r in group],dtype=float)
        b=np.asarray([r[right] for r in group],dtype=float)
        if len(np.unique(y))!=2:
            per[str(p)]={"n":len(y),"positives":int(y.sum()),"ap_diff":None,"auroc_diff":None}
            continue
        a_ap=float(average_precision_score(y,a));b_ap=float(average_precision_score(y,b))
        a_roc=float(roc_auc_score(y,a));b_roc=float(roc_auc_score(y,b))
        per[str(p)]={
            "n":len(y),"positives":int(y.sum()),
            "ap_diff":a_ap-b_ap,
            "auroc_diff":a_roc-b_roc,
        }
        values.append((p,a_ap-b_ap,a_roc-b_roc))
    result={"comparison":left+" minus "+right,"per_patient":per}
    if len(values)!=len(patients):
        result["descriptive_interval"]=None
        result["warning"]="At least one patient lacks two classes; five-patient macro undefined"
        return result
    rng=random.Random(seed)
    for metric_index,label in ((1,"AP"),(2,"AUROC")):
        vals=[v[metric_index] for v in values]
        draws=sorted(
            sum(rng.choice(vals) for _ in range(len(vals)))/len(vals)
            for _ in range(reps)
        )
        result[label+"_macro_diff"]={
            "mean":float(np.mean(vals)),
            "descriptive_95pct":[
                draws[int(.025*(reps-1))],
                draws[int(.975*(reps-1))],
            ],
            "positive_patient_deltas":sum(v>0 for v in vals),
        }
    return result


def run_study(
    *,
    source:Path,
    output:Path,
    max_iter:int=100,
    seed:int=20261102,
    bootstrap:int=1000,
)->dict:
    if max_iter<10 or bootstrap<100:
        raise ValueError("Require >=10 HGB iterations and >=100 bootstrap draws")
    if output.exists():
        raise FileExistsError("Create-only scientific evidence output")
    frame=read_frozen(source)
    rows=[]
    summary={}
    for kind in KINDS:
        candidate=frame.loc[frame.candidate_type==kind].copy()
        geometry,ct=expected_columns(candidate,kind)
        mask_train=candidate.split.eq("train").to_numpy()
        mask_val=candidate.split.eq("val").to_numpy()
        y_train=candidate.loc[mask_train,"y"].to_numpy(dtype=int)
        y_val=candidate.loc[mask_val,"y"].to_numpy(dtype=int)
        columns={}
        measurements={}
        for variant in VARIANTS:
            matrix=make_matrix(
                candidate,geometry=geometry,ct=ct,
                variant=variant,seed=seed,
            )
            prediction,sec=fit_predict(
                matrix[mask_train],y_train,matrix[mask_val],
                max_iter=max_iter,
            )
            columns[variant]=prediction
            measurements[variant]={
                "validation":metrics(y_val,prediction),
                "train_n":len(y_train),
                "train_positives":int(y_train.sum()),
                "train_feature_count":len(geometry) if variant=="geometry_only" else len(ct) if variant=="ct_only" else len(geometry)+len(ct),
                "training_and_inference_wall_seconds":sec,
            }
        for i,row in enumerate(candidate.loc[mask_val].itertuples(index=False)):
            rows.append({
                "scene_id":row.scene_id,
                "candidate_id":row.candidate_id,
                "candidate_type":kind,
                "scan_id":int(row.scan_id),
                "split":"val",
                "y":int(row.y),
                **{variant:float(columns[variant][i]) for variant in VARIANTS},
            })
        summary[kind]={
            "features_geometry":geometry,
            "features_CT":ct,
            "models":measurements,
        }
    paired={}
    for kind in KINDS:
        paired[kind]={
            "real_CT_gain_over_geometry_only":patient_macro_bootstrap(
                rows,kind,("geometry_plus_ct","geometry_only"),
                reps=bootstrap,seed=seed,
            ),
            "real_CT_gain_over_patient_shuffled_CT":patient_macro_bootstrap(
                rows,kind,("geometry_plus_ct","geometry_plus_shuffled_ct"),
                reps=bootstrap,seed=seed+1,
            ),
        }
    output.mkdir(parents=True,exist_ok=False)
    pred_file=output/"ct_signal_candidate_val_predictions.csv"
    with pred_file.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report={
        "status":"completed_real_precomputed_CCTA_feature_ablation_development_validation",
        "source":"committed_real_CT28_CCTA_geometry_and_CT_numeric_features",
        "source_sha256":sha(source),
        "input_rows":len(frame),
        "train_patient_ids":sorted(TRAIN),
        "val_patient_ids":sorted(VAL),
        "held_out_test_accessed":False,
        "train_fit_only":True,
        "model_family":"HistGradientBoostingClassifier",
        "model_parameters":{**DEFAULT_HGB,"max_iter":max_iter},
        "seed":seed,
        "patient_bootstrap_draws":bootstrap,
        "sources_only_precomputed_image_features":True,
        "results":summary,
        "paired_patient_macro":paired,
        "validation_predictions_sha256":sha(pred_file),
        "packages":{
            "python":platform.python_version(),
            "numpy":np.__version__,
            "pandas":pd.__version__,
            "scipy":scipy.__version__,
            "scikit_learn":sklearn.__version__,
        },
        "limitations":[
            "Candidate-level PAIR/JUNCTION identity, not relation NONE/PAIR/JUNCTION/BOTH or repaired 3D mask.",
            "CT features were already extracted from medical volumes in previous research, but original volumes are not opened.",
            "Five validation patients were used previously for model selection; no independent generalization claim.",
            "Patient-wise CT permutation is a dependence control and may also alter CT/geometry copula; interpret with clinical caution.",
            "HGB input dimensionality is different for geometry/CT variants; identical HGB family/hyperparameters do not imply equal model capacity.",
            "Not an ANZA versus CNN comparison, and no medical 3D volume was processed in this run.",
        ],
    }
    (output/"ct_signal_benchmark.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"
    )
    return report


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=DEFAULT_SOURCE)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--max-iter",type=int,default=100)
    p.add_argument("--bootstrap",type=int,default=1000)
    p.add_argument("--seed",type=int,default=20261102)
    a=p.parse_args()
    result=run_study(
        source=a.source,output=a.output_dir,max_iter=a.max_iter,
        bootstrap=a.bootstrap,seed=a.seed,
    )
    for kind in KINDS:
        print(f"=== {kind}: original precomputed CT28 candidate features ===")
        for variant in VARIANTS:
            val=result["results"][kind]["models"][variant]["validation"]
            print(f"{variant:32s} n={val['n']} pos={val['positives']} "
                  f"AP={val['average_precision']:.6f} ROC-AUC={val['auroc']:.6f}")
        comparisons=result["paired_patient_macro"][kind]
        for key,val in comparisons.items():
            print(key,json.dumps({
                m:val.get(m+"_macro_diff")
                for m in ("AP","AUROC")
            }))
    print(f"source sha256={result['source_sha256']}")
    print(f"report sha256={sha(a.output_dir/'ct_signal_benchmark.json')}")
    print(f"predictions sha256={result['validation_predictions_sha256']}")


if __name__=="__main__":
    main()
