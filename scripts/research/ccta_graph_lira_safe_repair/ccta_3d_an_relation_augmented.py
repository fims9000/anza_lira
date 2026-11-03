#!/usr/bin/env python3
"""Match K10/15's 15 TRAIN endpoint perturbations with neural 3D CT OOF scores.

The clean OOF relation head has a train-augmentation confound relative to K10.
This script reproduces the selected historical HGB geometry OOF scores and
uses FOLD-SPECIFIC (not patient-trained) CNN/ANZA CT candidate encoders to
score the EXACT same 15 position/tangent perturbations used by K10/15.

It fits one relation HGB with 1 clean + 15 train-perturbed scenes (equal total
per-scene weights; 0.5 clean, 0.5 distributed across 15 perturbations).
Validation is scored by FULL-TRAIN neural encoders, with no validation fit.
No held-out CT28 test image, label or hyperparameter enters this process.

This is an optional explicit scientific upgrade; the existing clean relation
head and Varvara's K10 pack are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from models.ccta3d_candidate_encoder import Candidate3dEncoder
from ccta_3d_an_patches import PatchConfig,extract_candidate_patch
from ccta_3d_an_relation import (
    SELECTED,SCENES,MODEL_SHA,TEST_IDS,
    _validated_oof_scores,substitute_neural_relation_stream,
)
from ccta_3d_an_oof import sha256
import train_ct28_perturbation_augmented_relation_head as augmentation
import train_ct28_selected_relation_augmentation as selected_training
from extract_ct28_scene_candidate_features import EXPECTED_PATIENT_IDS


TRAIN_IDS=frozenset(EXPECTED_PATIENT_IDS["train"])
VAL_IDS=frozenset(EXPECTED_PATIENT_IDS["val"])
EVAL_REPETITIONS=15
EVAL_SEED=20260920
POSITION_NOISE_MM=.8
TANGENT_NOISE_DEG=20.


class FoldNeuralCTTransform:
    """Replace ONLY the CT score in 15x training scenes with patient-OOF NN."""

    def __init__(
        self,*,pair_dir:Path,junction_dir:Path,variant:str,device:str="cpu",
        min_inside_fraction:float=.75,
    ):
        if variant not in {"anza3d","conv3d"}:
            raise ValueError("Unknown neural encoder type")
        if not 0<=min_inside_fraction<=1:
            raise ValueError("CT patch acceptance fraction outside [0,1]")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.variant=variant
        self.device=device
        self.min_inside_fraction=min_inside_fraction
        self.by_kind={}
        self.provenance={}
        configs=[]
        for kind,folder in (("PAIR",pair_dir),("JUNCTION",junction_dir)):
            _validated_oof_scores(folder,variant,kind)
            manifest=json.loads((folder/"oof_protocol.json").read_text())
            if manifest.get("source_input_control_mode","full_ct")!="full_ct":
                raise ValueError("Sham CT negative-control model cannot become medical evidence")
            if manifest.get("held_out_test_accessed") is not False:
                raise ValueError("OOF manifest contains held-out test evidence")
            if manifest.get("n_train_patients")!=17 or manifest.get("n_validation_patients")!=5:
                raise ValueError("OOF model must cover complete 17 train / 5 val split")
            patch_cfg=manifest["train_patch_sources"][0]["patch_protocol"]
            for m in manifest["train_patch_sources"]+manifest["validation_patch_sources"]:
                if m["patch_protocol"]!=patch_cfg:
                    raise ValueError("Mixed neural patch physical scales in OOF fold")
            configs.append(patch_cfg)
            fold_map={}
            models={}
            for item in manifest["fold_assignments"]:
                fold=int(item["fold"])
                trained=set(int(p) for p in item["train_patients"])
                held=set(int(p) for p in item["held_patients"])
                if held&trained or not held or held&TEST_IDS or not trained<=TRAIN_IDS or not held<=TRAIN_IDS:
                    raise ValueError("Invalid OOF patient isolation")
                if fold in models:
                    raise ValueError("Duplicate OOF fold")
                weights=folder/f"{kind}_{variant}_fold{fold}.pt"
                if sha256(weights)!=item["model_sha256"]:
                    raise ValueError("Train-OOF model checkpoint SHA mismatch")
                network=Candidate3dEncoder(
                    variant,voxel_mm=float(patch_cfg["voxel_mm"])
                ).to(device).eval()
                network.load_state_dict(
                    torch.load(weights,map_location="cpu",weights_only=True),
                    strict=True,
                )
                models[fold]=network
                for patient in held:
                    if patient in fold_map:
                        raise ValueError("Train patient assigned to multiple fold encoders")
                    fold_map[patient]=fold
            if set(fold_map)!=TRAIN_IDS:
                raise ValueError("Train patient OOF fold partition is incomplete")
            self.by_kind[kind]={"models":models,"patient_fold":fold_map}
            self.provenance[kind]={
                "candidate_evidence_sha256":manifest["score_sha256"],
                "fold_assignments":manifest["fold_assignments"],
                "original_patch_provenance":patch_cfg,
            }
        if configs[0]!=configs[1]:
            raise ValueError("PAIR and JUNCTION use different image patch contracts")
        self.patch_config=PatchConfig(**configs[0])
        self.tracked_candidate_scenes=0
        self.tracked_candidate_predictions=0

    def __call__(
        self,patient:int,repetition:int,endpoints:pd.DataFrame,
        candidates:pd.DataFrame,ct:np.ndarray,inverse_affine:np.ndarray,
        scores:pd.DataFrame,
    )->pd.DataFrame:
        if patient not in TRAIN_IDS or not 0<=repetition<EVAL_REPETITIONS:
            raise ValueError("Neural augmentation requires 17 train IDs and 15 frozen repeats")
        if len(candidates)!=len(scores):
            raise ValueError("Candidate/scores mismatch before NN augmentation")
        if endpoints[["scene_id","endpoint_id"]].duplicated().any():
            raise ValueError("Duplicate perturbed endpoint physical identity")
        lookup=endpoints.set_index(["scene_id","endpoint_id"])
        affine=np.linalg.inv(inverse_affine)
        output=scores.copy(deep=True)
        original_keys=list(zip(output["scene_id"],output["candidate_id"]))
        if len(set(original_keys))!=len(original_keys):
            raise ValueError("Repeated candidate identity in NN augmentation")
        row_lookup={key:i for i,key in enumerate(original_keys)}
        for kind in ("PAIR","JUNCTION"):
            group=candidates.loc[candidates.candidate_type.eq(kind)]
            if not len(group):
                continue
            fold=self.by_kind[kind]["patient_fold"][patient]
            model=self.by_kind[kind]["models"][fold]
            gathered=[]
            identifiers=[]
            for row in group.itertuples(index=False):
                ids=str(row.endpoint_ids).split("|")
                if len(ids)!=(2 if kind=="PAIR" else 3):
                    raise ValueError("PAIR/JUNCTION candidate degree changed")
                idx=[(row.scene_id,identifier) for identifier in ids]
                coords=lookup.loc[idx,["x_lps","y_lps","z_lps"]].to_numpy(dtype=float)
                patch,record=extract_candidate_patch(
                    ct,affine,coords,self.patch_config
                )
                if record["sampling_inside_volume_fraction"]<self.min_inside_fraction:
                    raise ValueError("Neural CT patch crosses unsupported CT volume boundary")
                gathered.append(patch)
                identifiers.append((row.scene_id,row.candidate_id))
            if not gathered:
                continue
            data=np.stack(gathered)
            out=[]
            with torch.no_grad():
                for start in range(0,len(data),8):
                    batch=torch.from_numpy(data[start:start+8]).to(self.device)
                    out.extend(torch.sigmoid(model(batch)).cpu().numpy().tolist())
            for key,p in zip(identifiers,out):
                if key not in row_lookup or not np.isfinite(p) or not 0<=p<=1:
                    raise ValueError("OOF score identity or probability invalid")
                output.loc[output.index[row_lookup[key]],"ct_score"]=float(p)
            self.tracked_candidate_predictions+=len(identifiers)
        self.tracked_candidate_scenes+=len(set(candidates.scene_id))
        return output


def train_augmented_neural_relation(
    *,
    pair_oof:Path,junction_oof:Path,variant:str,
    output_dir:Path,baseline_dir:Path=SELECTED,
    scene_dir:Path=SCENES,
    archive_dir:Path=augmentation.DEFAULT_ARCHIVE_DIR,
    device:str="cpu",
)->dict:
    if variant not in {"conv3d","anza3d"}:
        raise ValueError("Unknown encoder")
    if output_dir.exists():
        raise FileExistsError("Never overwrite existing neural relation experiment")
    if sha256(baseline_dir/"relation_models.joblib")!=MODEL_SHA:
        raise ValueError("Selected K10/15 model pack SHA mismatch")
    args=argparse.Namespace(
        scene_dir=scene_dir,clean_evidence_dir=baseline_dir,
        archive_dir=archive_dir,out_dir=output_dir,
        perturbations=EVAL_REPETITIONS,seed=EVAL_SEED,
        position_noise_mm=POSITION_NOISE_MM,
        tangent_noise_deg=TANGENT_NOISE_DEG,tau=.85,
    )
    augmentation.require_frozen_settings(args)
    loaded=augmentation.load_inputs(args)
    if set(loaded["plans"].scan_id.astype(int))!=(TRAIN_IDS|VAL_IDS):
        raise ValueError("Unexpected train/val patient set in selected K10/15 evidence")
    geometry_folds,max_error=selected_training.fold_models_for_patients(loaded)
    transform=FoldNeuralCTTransform(
        pair_dir=pair_oof,junction_dir=junction_oof,
        variant=variant,device=device,
    )
    neural=pd.concat([
        _validated_oof_scores(pair_oof,variant,"PAIR"),
        _validated_oof_scores(junction_oof,variant,"JUNCTION"),
    ],ignore_index=True)
    clean,columns,_=substitute_neural_relation_stream(
        loaded["plans"],loaded["candidates"],loaded["scores"],neural,
    )
    if columns!=list(loaded["models"]["ct_conditioned_relation_features"]):
        raise ValueError("Selected K10 84-feature schema has changed")
    clean_train=[dict(row) for row in clean if row["split"]=="train"]
    clean_val=[dict(row) for row in clean if row["split"]=="val"]
    if len(clean_train)!=1036 or len(clean_val)!=368:
        raise ValueError("Frozen relation scene counts changed")
    for row in clean_train:
        row["augmentation_index"]=0
        row["sample_weight"]=.5
    augmented,noisy_columns,audit=augmentation.build_augmented_train_rows(
        args,loaded["plans"],loaded["endpoints"],loaded["candidates"],
        geometry_folds,{patient:patient for patient in geometry_folds},
        score_transform=transform,
    )
    if columns!=noisy_columns or len(augmented)!=1036*EVAL_REPETITIONS:
        raise ValueError("Incomplete 15-copy neural train relation augmentation")
    for row in augmented:
        row["sample_weight"]=.5/EVAL_REPETITIONS
    all_train=clean_train+augmented
    augmentation.validate_augmentation_weights(all_train,EVAL_REPETITIONS)
    fitted,metrics,predictions=augmentation.fit_augmented_head(
        all_train,clean_val,columns,.85
    )
    if list(fitted.classes_)!=[0,1,2,3]:
        raise ValueError("Neural augmentation relation head lost a class")
    if len(predictions)!=368:
        raise ValueError("Validation prediction count changed")
    output_dir.mkdir(parents=True,exist_ok=False)
    model_path=output_dir/"augmented_relation_head.joblib"
    predictions_path=output_dir/"augmented_relation_predictions_val.csv"
    joblib.dump(fitted,model_path)
    pd.DataFrame(predictions).to_csv(predictions_path,index=False)
    pd.DataFrame(audit).to_csv(output_dir/"train_augmentation_audit.csv",index=False)
    manifest={
        "status":"experimental_patient_OOF_15copy_augmented_relation_head",
        "variant":variant,
        "source_K10_model_sha256":MODEL_SHA,
        "relation_head_filename":model_path.name,
        "model_sha256":sha256(model_path),
        "predictions_sha256":sha256(predictions_path),
        "pair_oof_sha256":sha256(pair_oof/f"PAIR_{variant}_oof_val.csv"),
        "junction_oof_sha256":sha256(junction_oof/f"JUNCTION_{variant}_oof_val.csv"),
        "train_scenes":len(clean_train),
        "validation_scenes":len(clean_val),
        "augmented_train_scenes":len(augmented),
        "relation_training_copies_per_scene":EVAL_REPETITIONS,
        "clean_relation_sample_weight":.5,
        "perturbation_sample_weight_each":.5/EVAL_REPETITIONS,
        "perturbation_seed":EVAL_SEED,
        "position_noise_mm":POSITION_NOISE_MM,
        "tangent_noise_deg":TANGENT_NOISE_DEG,
        "geometry_OOF_reproduction_max_abs_error":max_error,
        "neural_patient_OOF_provenance":transform.provenance,
        "neural_candidate_scores_created":transform.tracked_candidate_predictions,
        "frozen_identity_policy":"PAIR is controlled at inference by explicit identity source; JUNCTION geometry-only",
        "validation_metrics":metrics,
        "held_out_test_accessed":False,
        "limitations":[
            "Relation head uses 15 train perturb copies and separate patient-OOF neural CT scores.",
            "Neural PAIR/JUNCTION encoder itself was trained on clean candidate patches, not 15 position augmentations.",
            "Neither selected K10/15 nor real CCTA path v2 medical mask performance is reproduced by this relation-only head.",
            "Five development-validation patients are not independent confirmatory test.",
        ],
    }
    (output_dir/"relation_protocol.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"
    )
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair-oof",type=Path,required=True)
    p.add_argument("--junction-oof",type=Path,required=True)
    p.add_argument("--variant",choices=("anza3d","conv3d"),required=True)
    p.add_argument("--baseline-dir",type=Path,default=SELECTED)
    p.add_argument("--scene-dir",type=Path,default=SCENES)
    p.add_argument("--archive-dir",type=Path,default=augmentation.DEFAULT_ARCHIVE_DIR)
    p.add_argument("--device",default="cpu")
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    result=train_augmented_neural_relation(
        pair_oof=a.pair_oof,junction_oof=a.junction_oof,
        variant=a.variant,output_dir=a.output_dir,
        baseline_dir=a.baseline_dir,scene_dir=a.scene_dir,
        archive_dir=a.archive_dir,device=a.device,
    )
    print(json.dumps({
        "status":result["status"],
        "variant":result["variant"],
        "train_scenes":result["train_scenes"],
        "copies":result["relation_training_copies_per_scene"],
        "candidate_scores":result["neural_candidate_scores_created"],
        "validation_metrics":result["validation_metrics"],
        "held_out_test_accessed":False,
    },indent=2))


if __name__=="__main__":
    main()
