#!/usr/bin/env python3
"""Audit train-patient-OOF neural probability calibration without val/test labels.

Produce *sidecar diagnostics only*. The frozen 84-feature HGB and K10/15
inference are not silently modified. Before promoting calibration into the
relation gate, train a new HGB with explicitly calibrated, patient-OOF scores
and evaluate the entire unchanged 5-subject development protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ccta_3d_neural_calibration import (
    fit_train_only_platt,patient_crossfit_platt,save_calibrator,
)
from extract_ct28_scene_candidate_features import EXPECTED_PATIENT_IDS

ROOT=Path(__file__).resolve().parents[3]
SCENES=ROOT/"artifacts/varvara/ct28_graph_lira"


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda:stream.read(1024*1024),b""):
            digest.update(part)
    return digest.hexdigest()


def audit(
    *,
    kind:str,variant:str,oof_dir:Path,scenes_dir:Path,out_dir:Path,
    expected_train:frozenset[int]|None=None,
    expected_val:frozenset[int]|None=None,
)->dict:
    if kind not in {"PAIR","JUNCTION"} or not variant.endswith(("3d","_structure")):
        raise ValueError("Unknown type or neural variant")
    train_ids=(frozenset(EXPECTED_PATIENT_IDS["train"]) if expected_train is None
               else frozenset(expected_train))
    val_ids=(frozenset(EXPECTED_PATIENT_IDS["val"]) if expected_val is None
             else frozenset(expected_val))
    if not train_ids or not val_ids or train_ids&val_ids or not (
        train_ids<=EXPECTED_PATIENT_IDS["train"] and val_ids<=EXPECTED_PATIENT_IDS["val"]
    ):
        raise ValueError("Expected isolated train and development-validation patients")
    if out_dir.exists():
        raise FileExistsError("Never overwrite neural calibration study")
    manifest_path=oof_dir/"oof_protocol.json"
    source=oof_dir/f"{kind}_{variant}_oof_val.csv"
    manifest=json.loads(manifest_path.read_text())
    if (manifest.get("status")!="train_patient_OOF_experimental_CT_evidence_not_promoted_K10"
        or manifest.get("held_out_test_accessed") is not False
        or manifest.get("variant")!=variant
        or manifest.get("candidate_type")!=kind
        or manifest.get("score_sha256")!=sha256(source)):
        raise ValueError("Invalid or modified neural OOF source")
    data=pd.read_csv(source)
    required={"scene_id","candidate_id","candidate_type","scan_id","split","score","oof_fold"}
    if not required<=set(data.columns) or data.duplicated(["scene_id","candidate_id"]).any():
        raise ValueError("OOF identity schema incomplete or duplicated")
    if set(data.candidate_type)!={kind} or not np.isfinite(data.score).all():
        raise ValueError("Nonfinite or mixed-type neural OOF scores")
    if set(data.split)!={"train","val"}:
        raise ValueError("Expected both train-OOF and development-val scores")
    plans=pd.read_csv(scenes_dir/"scene_plan_train_val.csv")
    candidates=pd.read_csv(scenes_dir/"scene_candidates_train_val.csv")
    train_plan=plans.loc[plans.split.eq("train")&
                         plans.scan_id.isin(train_ids),["scene_id","scan_id"]]
    val_plan=plans.loc[plans.split.eq("val")&
                       plans.scan_id.isin(val_ids),["scene_id","scan_id"]]
    for split,part,expected in (("train",train_plan,train_ids),("val",val_plan,val_ids)):
        if set(part.scan_id.astype(int))!=expected:
            raise ValueError(f"Incomplete {split} plan patient coverage")
        rows=data.loc[data.split.eq(split)]
        if (set(rows.scan_id.astype(int))!=expected
            or (rows.oof_fold.lt(0).any() if split=="train"
                else rows.oof_fold.ne(-1).any())):
            raise ValueError("Wrong train-OOF/validation fold or patient identity")
        known=rows.merge(part,on=["scene_id","scan_id"],how="inner",
                         validate="many_to_one")
        if len(known)!=len(rows):
            raise ValueError("OOF patient scene does not agree with frozen plan")
    train=data.loc[data.split.eq("train")].copy()
    frozen_truth=candidates.loc[
        candidates.scene_id.isin(train_plan.scene_id)&
        candidates.candidate_type.eq(kind),
        ["scene_id","candidate_id","y"]
    ]
    merged=train.merge(frozen_truth,on=["scene_id","candidate_id"],
                       how="inner",validate="one_to_one")
    if len(merged)!=len(train) or len(merged)!=len(frozen_truth):
        raise ValueError("Train-OOF candidates do not reproduce frozen labeled scenes")
    # Critical: only TRAIN y values were ever read into the fit input.
    raw=merged.score.to_numpy(dtype=np.float64)
    truth=merged.y.to_numpy(dtype=float)
    groups=merged.scan_id.to_numpy(dtype=int)
    oof_calibrated,patient_receipts=patient_crossfit_platt(raw,truth,groups)
    final,final_receipt=fit_train_only_platt(raw,truth,groups)
    index={(str(row.scene_id),str(row.candidate_id)):score
           for row,score in zip(merged.itertuples(index=False),oof_calibrated)}
    output=data.copy()
    output["raw_score"]=output["score"]
    output["calibration_fit"]="all_other_train_patients"
    output["score"]=[
        (float(index[(str(row.scene_id),str(row.candidate_id))])
         if row.split=="train" else float(final.predict(np.asarray([row.score]))[0]))
        for row in output.itertuples(index=False)
    ]
    output.loc[output.split.eq("val"),"calibration_fit"]="all_train_patients"
    if not np.isfinite(output.score).all():
        raise ValueError("Incomplete calibrated candidate evidence")
    # Never replace original OOF files; no implicit use in clinical HGB.
    out_dir.mkdir(parents=True,exist_ok=False)
    calibration_path=out_dir/"train_only_platt.json"
    calibration_sha=save_calibrator(calibration_path,final_receipt)
    scored_path=out_dir/f"{kind}_{variant}_calibrated_sidecar.csv"
    output.to_csv(scored_path,index=False)
    report={
        "status":"train_only_neural_calibration_sidecar_not_activated",
        "candidate_type":kind,"variant":variant,
        "n_train_candidates":len(train),
        "n_val_candidates_scored_without_labels":int(data.split.eq("val").sum()),
        "training_patients":sorted(train_ids),
        "validation_patients_scored_only":sorted(val_ids),
        "validation_labels_used_for_fit":False,
        "held_out_CT28_test_accessed":False,
        "train_only_crossfit_brier":float(np.mean((oof_calibrated-truth)**2)),
        "train_only_raw_brier":float(np.mean((raw-truth)**2)),
        "fulltrain_calibration_sha256":calibration_sha,
        "source_OOF_csv_sha256":sha256(source),
        "source_OOF_manifest_sha256":sha256(manifest_path),
        "source_frozen_scene_plan_sha256":sha256(scenes_dir/"scene_plan_train_val.csv"),
        "source_frozen_candidates_sha256":sha256(scenes_dir/"scene_candidates_train_val.csv"),
        "calibrated_csv_sha256":sha256(scored_path),
        "patient_crossfit_provenance":patient_receipts,
        "warning":"Descriptive train-only calibration; no HGB, inference, K10 or frozen threshold has been modified.",
    }
    (out_dir/"calibration_audit.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n"
    )
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind",choices=("PAIR","JUNCTION"),required=True)
    parser.add_argument("--variant",required=True)
    parser.add_argument("--oof-dir",type=Path,required=True)
    parser.add_argument("--scenes",type=Path,default=SCENES)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    report=audit(
        kind=a.kind,variant=a.variant,oof_dir=a.oof_dir,
        scenes_dir=a.scenes,out_dir=a.out,
    )
    print(json.dumps({
        "status":report["status"],
        "training_patients":len(report["training_patients"]),
        "raw_brier":report["train_only_raw_brier"],
        "patient_crossfit_brier":report["train_only_crossfit_brier"],
        "val_labels_read":report["validation_labels_used_for_fit"],
    },indent=2))


if __name__=="__main__":
    main()
