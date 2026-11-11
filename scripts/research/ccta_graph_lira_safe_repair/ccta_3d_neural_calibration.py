"""Monotone, patient-grouped probability calibration for CT28 neural candidates.

This module is deliberately separate from the raw encoder and relation HGB.
No validation/test targets can enter the calibration fit: train patient IDs
are checked at the boundary. For train OOF scores each patient is calibrated
using OTHER train patients, then a final calibrator is fit to all 17 train
patients for development-validation/runtime predictions.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from extract_ct28_scene_candidate_features import EXPECTED_PATIENT_IDS


@dataclass(frozen=True)
class PlattTemperature:
    slope:float
    intercept:float

    def __post_init__(self):
        if not np.isfinite(self.slope) or not .001<=self.slope<=10.0:
            raise ValueError("Calibration slope must be positive and bounded")
        if not np.isfinite(self.intercept) or abs(self.intercept)>20:
            raise ValueError("Calibration intercept must be finite")

    def predict(self,raw_scores:np.ndarray)->np.ndarray:
        scores=_scores(raw_scores)
        return expit(self.slope*np.log(scores/(1-scores))+self.intercept)


def _scores(scores:np.ndarray)->np.ndarray:
    scores=np.asarray(scores,dtype=np.float64)
    if scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Need finite nonempty 1D neural scores")
    if (scores<0).any() or (scores>1).any():
        raise ValueError("Neural scores outside probability interval")
    return scores.clip(1e-6,1-1e-6)


def _labels(y)->np.ndarray:
    labels=np.asarray(y,dtype=np.float64)
    if labels.ndim!=1 or not len(labels) or not set(np.unique(labels))<={0.,1.}:
        raise ValueError("Expected binary calibration labels")
    return labels


def fit_train_only_platt(scores,labels,patients)->tuple[PlattTemperature,dict]:
    """Monotone Platt correction from *train-only*, patient-grouped OOF scores.

    Label and patient memberships are validated BEFORE any optimizer call.
    This calibrator must never be fitted using CT28 development-validation or
    held-out patients. It cannot reverse an ANN ranking (slope > 0).
    """
    p=_scores(scores)
    y=_labels(labels)
    groups=np.asarray(patients)
    if len(p)!=len(y) or len(y)!=len(groups):
        raise ValueError("Calibration score/label/patient arrays do not align")
    allowed=EXPECTED_PATIENT_IDS["train"]
    if any(int(group) not in allowed for group in groups):
        raise ValueError("Calibration must contain only CT28 train patients")
    if len(set(y))!=2 or len(set(groups))<2:
        raise ValueError("Calibration needs both labels and multiple train patients")
    logits=np.log(p/(1-p))
    # Constrained monotone slope ensures bad rankings cannot be "repaired"
    # by swapping positive/negative outputs after inspecting labels.
    def objective(params):
        slope,bias=params
        a=slope*logits+bias
        # Numerical stable logistic loss + weak shrinkage to identity.
        return float(np.mean(np.logaddexp(0.,a)-y*a)
                     +1e-3*(slope-1.)**2+1e-4*bias*bias)
    result=minimize(objective,np.array([1.,0.]),method="L-BFGS-B",
                    bounds=((.001,10.),(-20.,20.)),
                    options={"maxiter":150})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Patient-train-only monotone calibration optimization failed")
    scaler=PlattTemperature(float(result.x[0]),float(result.x[1]))
    after=scaler.predict(p)
    provenance={
        "status":"train_only_monotone_neural_candidate_calibration",
        "n_train_candidates":len(p),
        "n_train_patients":len(set(groups)),
        "train_patient_ids":sorted(int(x) for x in set(groups)),
        "n_positive":int(y.sum()),
        "train_raw_brier":float(np.mean((p-y)**2)),
        "train_calibrated_brier":float(np.mean((after-y)**2)),
        "train_auc_before":float(roc_auc_score(y,p)),
        "train_auc_after":float(roc_auc_score(y,after)),
        "held_out_CT28_test_accessed":False,
        "development_validation_fit_accessed":False,
        "parameters":asdict(scaler),
    }
    if abs(provenance["train_auc_before"]-provenance["train_auc_after"])>1e-9:
        raise AssertionError("Monotone calibration altered candidate ranking")
    return scaler,provenance


def patient_crossfit_platt(scores,labels,patients):
    """Each train patient scored using calibration fitted on other patients.

    The calibration provenance is not independent clinical evidence; it only
    ensures a training patient's own labels do not set its probability scale.
    """
    scores=_scores(scores)
    labels=_labels(labels)
    groups=np.asarray(patients)
    if len(scores)!=len(labels) or len(scores)!=len(groups):
        raise ValueError("Misaligned patient calibration arrays")
    if any(int(p) not in EXPECTED_PATIENT_IDS["train"] for p in groups):
        raise ValueError("Patient OOF calibration cannot use val/test patients")
    output=np.full(len(scores),np.nan,dtype=float)
    receipts=[]
    for patient in sorted(set(int(x) for x in groups)):
        held=groups==patient
        train=~held
        if not train.any() or len(np.unique(labels[train]))<2:
            raise ValueError("Insufficient calibration data without held patient")
        fitted,receipt=fit_train_only_platt(
            scores[train],labels[train],groups[train]
        )
        output[held]=fitted.predict(scores[held])
        receipts.append({
            "held_patient_id":patient,
            "calibration_train_ids":receipt["train_patient_ids"],
            "calibrator":asdict(fitted),
        })
        if patient in receipt["train_patient_ids"]:
            raise AssertionError("Patient calibration label leakage")
    if not np.isfinite(output).all():
        raise ValueError("Incomplete patient-OOF calibration")
    return output,receipts


def load_calibrator(path:Path)->PlattTemperature:
    receipt=json.loads(Path(path).read_text(encoding="utf-8"))
    if (receipt.get("status")!="train_only_monotone_neural_candidate_calibration"
        or receipt.get("development_validation_fit_accessed") is not False
        or receipt.get("held_out_CT28_test_accessed") is not False):
        raise ValueError("Unapproved or unsealed candidate calibration artifact")
    ids=receipt.get("train_patient_ids",[])
    if not ids or any(int(x) not in EXPECTED_PATIENT_IDS["train"] for x in ids):
        raise ValueError("Untrusted calibration patient provenance")
    return PlattTemperature(**receipt["parameters"])


def save_calibrator(path:Path,provenance:dict)->str:
    """Immutable JSON+hash, no pickle or untrusted external code execution."""
    path=Path(path)
    if path.exists():
        raise FileExistsError("Do not overwrite calibration receipts")
    if provenance.get("status")!="train_only_monotone_neural_candidate_calibration":
        raise ValueError("Wrong calibration provenance")
    path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(provenance,sort_keys=True,indent=2).encode()+b"\n"
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
