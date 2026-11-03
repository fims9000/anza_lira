#!/usr/bin/env python3
"""Train-only controlled 3D ANZA versus compact 3D CNN candidate experiment.

The two models use the *identical* CT+endpoint candidate patch cache,
patient-disjoint train/validation groups, class weighting, optimizer, epochs
and seed. Validation does not select hyperparameters/checkpoints.

THIS IS A CANDIDATE-LEVEL EXPLORATORY ABLATION, NOT A REPLACEMENT FOR THE
SELECTED K10/15 HGB PIPELINE. To replace its CT stream, a separate patient-
level out-of-fold training/scoring protocol would be required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
import sys
from pathlib import Path

# Direct CLI execution adds the script directory, not necessarily repo root.
# The encoder lives in the canonical top-level models/ package.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

from models.ccta3d_candidate_encoder import Candidate3dEncoder,parameter_budget
from ccta_3d_an_patch_cache import DEFAULT_CACHE
from extract_ct28_scene_candidate_features import EXPECTED_PATIENT_IDS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_type(
    directory:Path,
    patients:list[int],
    *,
    split:str,
    candidate_type:str,
)->tuple[np.ndarray,np.ndarray,list[dict],list[dict]]:
    arrays=[]
    labels=[]
    examples=[]
    receipts=[]
    for patient in patients:
        path=directory/f"patches_{patient}.npz"
        receipt_path=directory/f"patches_{patient}.json"
        receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("patient_id")!=patient or receipt.get("split")!=split:
            raise ValueError("Patient-level patch cache has wrong split/provenance")
        if receipt.get("cache_sha256")!=digest(path):
            raise ValueError("Patch cache SHA mismatch")
        if receipt.get("model_input_contains_ground_truth") is not False:
            raise ValueError("Ground truth included as model feature")
        if receipt.get("endpoint_outside_patch_count",0)>0:
            raise ValueError(
                "CCTA patch cache has candidate endpoints outside fixed physical FOV; "
                "choose a predeclared larger patch and rebuild before training"
            )
        data=np.load(path,allow_pickle=False)
        keep=data["candidate_type"]==candidate_type
        x=data["patches"][keep].astype(np.float32)
        y=data["y"][keep].astype(np.float32)
        if not set(np.unique(y)) <= {0.,1.}:
            raise ValueError("Nonbinary candidate targets")
        arrays.append(x)
        labels.append(y)
        for key,scene in zip(data["candidate_id"][keep],data["scene_id"][keep]):
            examples.append({
                "patient_id":patient,
                "candidate_id":str(key),
                "scene_id":str(scene),
            })
        receipts.append({
            "patient_id":patient,"cache_sha256":digest(path),
            "patch_protocol":receipt["patch_protocol"],
            "input_control_mode":receipt.get("input_control_mode","full_ct"),
            "endpoint_outside_patch_count":receipt.get("endpoint_outside_patch_count"),
            "minimum_endpoint_patch_margin_mm":receipt.get("minimum_endpoint_patch_margin_mm"),
        })
    if not arrays or not sum(a.shape[0] for a in arrays):
        raise ValueError(f"No {candidate_type} candidate patches available")
    if len({json.dumps(r["patch_protocol"],sort_keys=True) for r in receipts})!=1:
        raise ValueError("Patch protocols differ across patients")
    if len({r["input_control_mode"] for r in receipts})!=1:
        raise ValueError("Cannot mix real-CCTA and negative-control patients")
    return np.concatenate(arrays),np.concatenate(labels),examples,receipts


def classify_metrics(truth:np.ndarray,confidence:np.ndarray)->dict:
    if truth.shape!=confidence.shape or not np.isfinite(confidence).all():
        raise ValueError("Predictions/labels inconsistent or nonfinite")
    unique=np.unique(truth)
    if len(unique)!=2:
        return {
            "n":len(truth),"positives":int(truth.sum()),
            "auroc":None,"auprc":None,
            "warning":"Only one class; AUC is undefined",
        }
    return {
        "n":len(truth),"positives":int(truth.sum()),
        "auroc":float(roc_auc_score(truth,confidence)),
        "auprc":float(average_precision_score(truth,confidence)),
    }



def candidate_scene_ranking_metrics(
    truth:np.ndarray,scores:np.ndarray,examples:list[dict]
)->dict:
    """PAIR/JUNCTION identity top-1 among scenes with exactly one true edge.

    Negative/NONE scenes do not have a correct candidate, therefore a
    per-scene top-1 ranking score is undefined for them and they remain
    explicitly accounted for, not treated as perfect abstentions.
    """
    if (truth.shape!=scores.shape or len(truth)!=len(examples)
        or not np.isfinite(scores).all() or
        not set(np.unique(truth))<={0.,1.}):
        raise ValueError("Invalid rank evaluation labels, scores or identities")
    if np.any(scores<0) or np.any(scores>1):
        raise ValueError("Candidate scores outside [0,1]")
    scene_index={}
    unique_keys=set()
    for i,record in enumerate(examples):
        sid=str(record["scene_id"])
        pid=int(record["patient_id"])
        cid=str(record["candidate_id"])
        key=(sid,cid)
        if key in unique_keys:
            raise ValueError("Duplicate candidate within scene")
        unique_keys.add(key)
        entry=scene_index.setdefault(sid,{"patient":pid,"positions":[]})
        if entry["patient"]!=pid:
            raise ValueError("Same candidate scene spans multiple patients")
        entry["positions"].append((i,cid))
    per_patient={}
    scenes_with_true=0
    scenes_none=0
    correct=0
    for scene_id,record in sorted(scene_index.items()):
        selection=record["positions"]
        indices=np.asarray([index for index,_ in selection],dtype=int)
        labels=truth[indices]
        positives=int(labels.sum())
        if positives>1:
            raise ValueError("Scene contains more than one true candidate of one kind")
        pid=str(record["patient"])
        patient=per_patient.setdefault(pid,{
            "scenes":0,"positive_scenes":0,"none_scenes":0,
            "top1_correct":0,
        })
        patient["scenes"]+=1
        if positives==0:
            scenes_none+=1
            patient["none_scenes"]+=1
            continue
        # Tie-breaking by candidate ID, never using true candidate label.
        best=sorted(
            selection,key=lambda pair:(-float(scores[pair[0]]),pair[1])
        )[0][0]
        is_right=int(truth[best]==1)
        correct+=is_right
        scenes_with_true+=1
        patient["top1_correct"]+=is_right
        patient["positive_scenes"]+=1
    for patient in per_patient.values():
        n=patient["positive_scenes"]
        patient["top1_accuracy"]=(
            patient["top1_correct"]/n if n else None
        )
    return {
        "total_scenes":len(scene_index),
        "positive_scenes":scenes_with_true,
        "none_scenes":scenes_none,
        "top1_correct":correct,
        "top1_accuracy":correct/scenes_with_true if scenes_with_true else None,
        "per_patient":per_patient,
        "definition":"identity ranking only among scenes containing exactly one positive candidate; NONE accounted separately",
    }



def fit_comparison(
    *,
    cache_dir:Path,
    output_dir:Path,
    candidate_type:str,
    train_patients:list[int],
    val_patients:list[int],
    epochs:int=8,
    batch_size:int=8,
    learning_rate:float=1e-3,
    seed:int=20261102,
    device:str="cpu",
    allow_subset:bool=False,
)->dict:
    if candidate_type not in {"PAIR","JUNCTION"}:
        raise ValueError("Only PAIR or JUNCTION candidate heads supported")
    train_set=EXPECTED_PATIENT_IDS["train"]
    val_set=EXPECTED_PATIENT_IDS["val"]
    if len(set(train_patients))!=len(train_patients) or len(set(val_patients))!=len(val_patients):
        raise ValueError("Duplicate patient")
    if not set(train_patients)<=train_set or not set(val_patients)<=val_set:
        raise ValueError("Hard train/validation/test CT28 patient firewall")
    if not allow_subset and (
        set(train_patients)!=train_set or set(val_patients)!=val_set
    ):
        raise ValueError("Full comparison requires all 17 train and 5 validation subjects")
    if output_dir.exists():
        raise FileExistsError("Do not overwrite an earlier training run")
    if epochs<1 or batch_size<1 or not np.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError("Invalid training hyperparameters")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")

    train_x,train_y,train_examples,train_sources=load_type(
        cache_dir,train_patients,split="train",candidate_type=candidate_type
    )
    valid_x,valid_y,valid_examples,valid_sources=load_type(
        cache_dir,val_patients,split="val",candidate_type=candidate_type
    )
    if train_x.shape[1]!=2 or train_x.shape[2:]!=valid_x.shape[2:]:
        raise ValueError("Incompatible CT patch dimensions")
    npositive=int(train_y.sum()); nnegative=len(train_y)-npositive
    if min(npositive,nnegative)<1:
        raise ValueError("Training requires both positive and negative candidates")
    pos_weight=float(nnegative/npositive)
    patch_config=train_sources[0]["patch_protocol"]
    if patch_config!=valid_sources[0]["patch_protocol"]:
        raise ValueError("Train/validation patch source protocols differ")
    if train_sources[0]["input_control_mode"]!=valid_sources[0]["input_control_mode"]:
        raise ValueError("Train/validation CT image-control modes differ")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    az_probe=Candidate3dEncoder("anza3d",voxel_mm=patch_config["voxel_mm"])
    cnn_probe=Candidate3dEncoder("conv3d",voxel_mm=patch_config["voxel_mm"])
    budget=parameter_budget(az_probe,cnn_probe)
    if not budget["matched_parameter_budget"]:
        raise ValueError("ANZA and CNN parameter budget differs by >20%")

    output_dir.mkdir(parents=True,exist_ok=False)
    measurements={}
    for variant in ("conv3d","anza3d"):
        # Reuse exactly the same initialization seed and train batch ordering
        # for both model families. Different architectures are not claimed
        # to have identical starting functions.
        torch.manual_seed(seed)
        model=Candidate3dEncoder(variant,voxel_mm=patch_config["voxel_mm"]).to(device)
        optimizer=torch.optim.AdamW(
            model.parameters(),lr=learning_rate,weight_decay=1e-4
        )
        criterion=nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight],device=device)
        )
        generator=torch.Generator().manual_seed(seed)
        loader=DataLoader(TensorDataset(
            torch.from_numpy(train_x),
            torch.from_numpy(train_y),
        ),batch_size=batch_size,shuffle=True,generator=generator)
        epoch_losses=[]
        started=time.perf_counter()
        for epoch in range(epochs):
            model.train()
            weighted=0.0
            seen=0
            for x,y in loader:
                x=x.to(device);y=y.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction=model(x)
                loss=criterion(prediction,y)
                loss.backward()
                optimizer.step()
                weighted+=float(loss.item())*len(y)
                seen+=len(y)
            epoch_losses.append(weighted/seen)
        duration=time.perf_counter()-started
        model.eval()
        all_scores=[]
        with torch.no_grad():
            for start in range(0,len(valid_y),batch_size):
                x=torch.from_numpy(valid_x[start:start+batch_size]).to(device)
                all_scores.append(torch.sigmoid(model(x)).cpu().numpy())
        scores=np.concatenate(all_scores).astype(np.float64)
        metrics=classify_metrics(valid_y.astype(float),scores)
        scene_ranking=candidate_scene_ranking_metrics(
            valid_y.astype(float),scores,valid_examples
        )
        per_patient={}
        for patient in val_patients:
            which=np.asarray([x["patient_id"]==patient for x in valid_examples])
            if which.any():
                per_patient[str(patient)]=classify_metrics(
                    valid_y[which].astype(float),scores[which]
                )
        checkpoint=output_dir/f"{candidate_type}_{variant}.pt"
        torch.save(model.state_dict(),checkpoint)
        predictions_path=output_dir/f"{candidate_type}_{variant}_val_scores.csv"
        with predictions_path.open("w",newline="",encoding="utf-8") as stream:
            fields=["patient_id","scene_id","candidate_id","candidate_type","y","score"]
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            for example,y,score in zip(valid_examples,valid_y,scores):
                writer.writerow({**example,"candidate_type":candidate_type,
                                 "y":int(y),"score":float(score)})
        measurements[variant]={
            "val_metrics":metrics,
            "val_scene_top1_identity":scene_ranking,
            "val_by_patient":per_patient,
            "train_epoch_losses":epoch_losses,
            "training_seconds":duration,
            "checkpoint_sha256":digest(checkpoint),
            "predictions_sha256":digest(predictions_path),
            "trainable_parameters":model.parameter_count(),
            "source_cache_sha256":{
                "train":train_sources,"val":valid_sources,
            },
        }
    report={
        "status":"exploratory_candidate_ablation_not_frozen_K10",
        "candidate_type":candidate_type,
        "train_patients":train_patients,
        "validation_patients":val_patients,
        "held_out_test_accessed":False,
        "label_used_in_network_input":False,
        "epochs":epochs,"batch_size":batch_size,
        "learning_rate":learning_rate,"seed":seed,
        "device":device,"training_class_pos_weight":pos_weight,
        "candidate_patch_config":patch_config,
        "input_control_mode":train_sources[0]["input_control_mode"],
        "matched_capacity":budget,
        "results":measurements,
        "warnings":[
            "Validation metrics are development-only; no held-out patient result",
            "Single last-epoch models, no validation checkpoint selection",
            "Candidate scores are not patient-level OOF; do not inject into K10 relation head",
            "ANZA superiority requires replicated training seeds and real patient tests",
        ],
    }
    (output_dir/"experiment.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"
    )
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--candidate-type",choices=("PAIR","JUNCTION"),required=True)
    parser.add_argument("--epochs",type=int,default=8)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--learning-rate",type=float,default=.001)
    parser.add_argument("--seed",type=int,default=20261102)
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    result=fit_comparison(
        cache_dir=args.cache_dir,output_dir=args.output_dir,
        candidate_type=args.candidate_type,
        train_patients=sorted(EXPECTED_PATIENT_IDS["train"]),
        val_patients=sorted(EXPECTED_PATIENT_IDS["val"]),
        epochs=args.epochs,batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,device=args.device,
    )
    print(json.dumps({
        "status":result["status"],
        "candidate_type":result["candidate_type"],
        "parameters":result["matched_capacity"],
        "validation":{
            model:results["val_metrics"]
            for model,results in result["results"].items()
        },
    },indent=2))


if __name__=="__main__":
    main()
