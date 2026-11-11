#!/usr/bin/env python3
"""Patient-OOF 3D CNN/ANZA candidate evidence for a future Graph-LIRA head.

All 17 train patients are grouped into F folds. Each train candidate receives
a score from a model whose training patients exclude its scan_id. The five
validation patients are scored only by a full-train fit. A later, separate
relation-head experiment can consume these scores without in-sample leakage.

This DOES NOT promote the new encoder into selected K10/15: matching relation-
head augmentation, identity policy, consistency and full-mask testing remain.
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

# Preserve direct Linux CLI imports from scripts/research/... without PYTHONPATH.
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

from models.ccta3d_candidate_encoder import Candidate3dEncoder
from models.structural_bridge3d import vessel_structure_loss
from ccta_3d_an_patch_cache import DEFAULT_CACHE
from ccta_3d_an_train import load_type, load_train_type_with_vessel_targets, classify_metrics, candidate_scene_ranking_metrics
from extract_ct28_scene_candidate_features import EXPECTED_PATIENT_IDS


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def fit_score(
    train_x:np.ndarray,train_y:np.ndarray,
    inference_x:np.ndarray,
    *,
    variant:str,epochs:int,batch_size:int,learning_rate:float,
    seed:int,device:str,voxel_mm:float,
    candidate_type:str="PAIR",train_vessel_targets:np.ndarray|None=None,
)->tuple[torch.nn.Module,np.ndarray,dict]:
    if not {0,1}<=set(np.unique(train_y)):
        raise ValueError("Each training fold must contain both candidate classes")
    n_pos=int(train_y.sum())
    n_neg=len(train_y)-n_pos
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    structure_aware=variant.endswith("_structure")
    expected_channels=4 if "_sep_" in variant else 2
    if train_x.ndim!=5 or train_x.shape[1]!=expected_channels or inference_x.ndim!=5 or inference_x.shape[1:]!=train_x.shape[1:]:
        raise ValueError("OOF candidate encoder inputs do not match frozen endpoint-encoding protocol")
    if structure_aware and (train_vessel_targets is None or
                            train_vessel_targets.shape!=(len(train_y),1,*train_x.shape[2:])):
        raise ValueError("Structural 3D training requires aligned dense train-only vessel targets")
    if not structure_aware and train_vessel_targets is not None:
        raise ValueError("Do not supply dense vessel labels to baseline training")
    model=Candidate3dEncoder(variant,voxel_mm=voxel_mm,candidate_type=candidate_type).to(device)
    tensors=[torch.from_numpy(train_x),torch.from_numpy(train_y)]
    if structure_aware:
        tensors.append(torch.from_numpy(train_vessel_targets))
    loader=DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=learning_rate,weight_decay=1e-4
    )
    criterion=nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_neg/n_pos],device=device)
    )
    started=time.perf_counter()
    history=[]
    for epoch in range(epochs):
        model.train()
        total=0.0
        for batch in loader:
            x=batch[0].to(device);y=batch[1].to(device)
            optimizer.zero_grad(set_to_none=True)
            if structure_aware:
                logits,dense_logits,path_prob=model(x,return_structure=True)
                vessel_truth=batch[2].to(device)
                loss=(criterion(logits,y)
                      +.35*vessel_structure_loss(dense_logits,vessel_truth,skeleton_iterations=2)
                      +.15*criterion(torch.logit(path_prob),y))
            else:
                loss=criterion(model(x),y)
            loss.backward()
            optimizer.step()
            total+=float(loss.item())*len(y)
        history.append(total/len(train_y))
    model.eval()
    predictions=[]
    with torch.no_grad():
        for start in range(0,len(inference_x),batch_size):
            x=torch.from_numpy(inference_x[start:start+batch_size]).to(device)
            predictions.append(torch.sigmoid(model(x)).cpu().numpy())
    scores=np.concatenate(predictions).astype(np.float64)
    if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError("Invalid ANN candidate probabilities")
    return model,scores,{
        "train_examples":len(train_y),
        "score_examples":len(inference_x),
        "train_positives":n_pos,
        "epoch_losses":history,
        "structure_training":({"dense_loss_weight":.35,"path_loss_weight":.15,
                               "skeleton_iterations":2} if structure_aware else None),
        "wall_seconds":time.perf_counter()-started,
    }


def build_oof(
    *,
    cache_dir:Path,
    out_dir:Path,
    candidate_type:str,
    variant:str,
    train_patients:list[int],
    val_patients:list[int],
    folds:int=5,
    epochs:int=8,
    batch_size:int=8,
    learning_rate:float=.001,
    seed:int=20261102,
    device:str="cpu",
    allow_subset:bool=False,
)->dict:
    if candidate_type not in {"PAIR","JUNCTION"} or variant not in {"anza3d","conv3d","anza3d_structure","conv3d_structure","anza3d_noaxial_structure","anza3d_sep_structure","conv3d_sep_structure","anza3d_noaxial_sep_structure"}:
        raise ValueError("Unknown candidate/model type")
    train_known=EXPECTED_PATIENT_IDS["train"]
    val_known=EXPECTED_PATIENT_IDS["val"]
    if len(set(train_patients))!=len(train_patients) or len(set(val_patients))!=len(val_patients):
        raise ValueError("Duplicate patient")
    if not set(train_patients)<=train_known or not set(val_patients)<=val_known:
        raise ValueError("OOF split contains validation/test patient in train")
    if not allow_subset and (set(train_patients)!=train_known or set(val_patients)!=val_known):
        raise ValueError("Full OOF requires all 17 train and 5 validation patients")
    if folds<2 or folds>len(train_patients) or epochs<1 or batch_size<1 or learning_rate<=0:
        raise ValueError("Invalid patient OOF training configuration")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if out_dir.exists():
        raise FileExistsError("OOF experiment output is immutable")

    structured=variant.endswith("_structure")
    train_targets=None
    if structured:
        train_x,train_y,train_targets,train_examples,train_sources=(
            load_train_type_with_vessel_targets(
                cache_dir,train_patients,candidate_type=candidate_type
            )
        )
    else:
        train_x,train_y,train_examples,train_sources=load_type(
            cache_dir,train_patients,split="train",candidate_type=candidate_type
        )
    val_x,val_y,val_examples,val_sources=load_type(
        cache_dir,val_patients,split="val",candidate_type=candidate_type
    )
    if train_x.shape[1:]!=val_x.shape[1:]:
        raise ValueError("Train and validation 3D patches differ in shape")
    expected_encoding="separate" if "_sep_" in variant else "summed"
    if any(record["patch_protocol"].get("endpoint_encoding","summed")!=expected_encoding
           for record in train_sources+val_sources):
        raise ValueError("OOF model variant mismatches candidate endpoint-encoding")
    if train_sources[0]["patch_protocol"]!=val_sources[0]["patch_protocol"]:
        raise ValueError("Train and validation patch protocol differs")
    if train_sources[0]["input_control_mode"]!=val_sources[0]["input_control_mode"]:
        raise ValueError("Train and validation image-control modes differ")
    patch_voxel_mm=float(train_sources[0]["patch_protocol"]["voxel_mm"])
    groups=np.asarray([entry["patient_id"] for entry in train_examples])
    unique=set(groups)
    if unique!=set(train_patients):
        raise ValueError("Candidate patch rows do not cover all train patients")
    splitter=GroupKFold(n_splits=folds)
    held_assignments=np.full(len(train_y),-1,dtype=int)
    oof=np.full(len(train_y),np.nan,dtype=np.float64)
    fold_info=[]
    out_dir.mkdir(parents=True,exist_ok=False)
    for fold,(fit_idx,hold_idx) in enumerate(
        splitter.split(train_x,train_y,groups)
    ):
        trained=set(int(p) for p in groups[fit_idx])
        held=set(int(p) for p in groups[hold_idx])
        if trained&held or not held:
            raise ValueError("OOF patient leaked into fitted fold")
        model,scores,history=fit_score(
            train_x[fit_idx],train_y[fit_idx],train_x[hold_idx],
            variant=variant,epochs=epochs,batch_size=batch_size,
            learning_rate=learning_rate,seed=seed,device=device,
            voxel_mm=patch_voxel_mm,candidate_type=candidate_type,
            train_vessel_targets=(train_targets[fit_idx] if structured else None),
        )
        oof[hold_idx]=scores
        held_assignments[hold_idx]=fold
        filename=out_dir/f"{candidate_type}_{variant}_fold{fold}.pt"
        torch.save(model.state_dict(),filename)
        fold_info.append({
            "fold":fold,
            "train_patients":sorted(trained),
            "held_patients":sorted(held),
            "model_sha256":sha256(filename),
            **history,
        })
    if not np.isfinite(oof).all() or (held_assignments<0).any():
        raise RuntimeError("Patient-OOF predictions are not complete")

    model,val_scores,full_history=fit_score(
        train_x,train_y,val_x,variant=variant,
        epochs=epochs,batch_size=batch_size,
        learning_rate=learning_rate,seed=seed,device=device,
        voxel_mm=patch_voxel_mm,candidate_type=candidate_type,
        train_vessel_targets=train_targets if structured else None,
    )
    checkpoint=out_dir/f"{candidate_type}_{variant}_fulltrain.pt"
    torch.save(model.state_dict(),checkpoint)
    scores_path=out_dir/f"{candidate_type}_{variant}_oof_val.csv"
    keys=set()
    with scores_path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=[
            "scene_id","candidate_id","candidate_type","scan_id",
            "split","score","oof_fold"
        ])
        writer.writeheader()
        for rows,predictions,split,folds_for_rows in (
            (train_examples,oof,"train",held_assignments),
            (val_examples,val_scores,"val",np.full(len(val_y),-1)),
        ):
            for sample,score,fold in zip(rows,predictions,folds_for_rows):
                key=(sample["scene_id"],sample["candidate_id"])
                if key in keys:
                    raise ValueError("Duplicate CT28 candidate identity in OOF evidence")
                keys.add(key)
                writer.writerow({
                    "scene_id":sample["scene_id"],
                    "candidate_id":sample["candidate_id"],
                    "scan_id":sample["patient_id"],
                    "candidate_type":candidate_type,
                    "split":split,"score":float(score),
                    "oof_fold":int(fold),
                })
    report={
        "status":"train_patient_OOF_experimental_CT_evidence_not_promoted_K10",
        "structure_objective":({"vessel_supervision":"train_only",
                               "dense_loss_weight":.35,"path_loss_weight":.15,
                               "cldice_skeleton_iterations":2} if structured else None),
        "variant":variant,
        "candidate_type":candidate_type,
        "n_train_patients":len(train_patients),
        "n_validation_patients":len(val_patients),
        "fold_count":folds,
        "fold_assignments":fold_info,
        "fixed_train_hyperparameters":{
            "epochs":epochs,"batch_size":batch_size,
            "learning_rate":learning_rate,"seed":seed,"device":device,
        },
        "train_OOF_metrics":classify_metrics(train_y,oof),
        "train_OOF_scene_top1_identity":candidate_scene_ranking_metrics(
            train_y,oof,train_examples
        ),
        "validation_fulltrain_metrics":classify_metrics(val_y,val_scores),
        "validation_scene_top1_identity":candidate_scene_ranking_metrics(
            val_y,val_scores,val_examples
        ),
        "fulltrain_history":full_history,
        "fulltrain_model_sha256":sha256(checkpoint),
        "score_sha256":sha256(scores_path),
        "train_patch_sources":train_sources,
        "validation_patch_sources":val_sources,
        "held_out_test_accessed":False,
        "source_input_control_mode":train_sources[0]["input_control_mode"],
        "warnings":[
            "Candidate-level validation, not complete end-to-end Graph-LIRA test",
            "OOF leakage excluded at patient level; candidate windows remain correlated",
            "Relation head and its train augmentation must be retrained before model promotion",
        ],
    }
    (out_dir/"oof_protocol.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"
    )
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--candidate-type",choices=("PAIR","JUNCTION"),required=True)
    parser.add_argument("--variant",choices=("conv3d","anza3d"),required=True)
    parser.add_argument("--folds",type=int,default=5)
    parser.add_argument("--epochs",type=int,default=8)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--learning-rate",type=float,default=.001)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--seed",type=int,default=20261102)
    args=parser.parse_args()
    report=build_oof(
        cache_dir=args.cache_dir,out_dir=args.output_dir,
        candidate_type=args.candidate_type,variant=args.variant,
        train_patients=sorted(EXPECTED_PATIENT_IDS["train"]),
        val_patients=sorted(EXPECTED_PATIENT_IDS["val"]),
        folds=args.folds,epochs=args.epochs,batch_size=args.batch_size,
        learning_rate=args.learning_rate,seed=args.seed,device=args.device,
    )
    print(json.dumps({
        "status":report["status"],
        "variant":report["variant"],
        "candidate_type":report["candidate_type"],
        "OOF":report["train_OOF_metrics"],
        "validation":report["validation_fulltrain_metrics"],
    },indent=2))


if __name__=="__main__":
    main()
