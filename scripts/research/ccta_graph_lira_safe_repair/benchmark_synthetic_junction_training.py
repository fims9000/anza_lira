#!/usr/bin/env python3
"""Train CNN/ANZA JUNCTION with existing synthetic volumetric three-arm generator.

Completely separate train/validation/test generator seeds. All candidates are
synthetic and are checked against voxelized 26-connected vessel-mask truth.
This is NOT ImageCAS, coronary patient data or an end-to-end repair claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from benchmark_structural_tubes import (
    fit,make_phantoms,normalize_synthetic_hu_histograms,
)


def synthetic_data(n:int,seed:int,side:int,*,same_histogram:bool,endpoints_only:bool):
    x,y,gt=make_phantoms(n,seed,"JUNCTION",side=side,
                         separate_endpoints=True,close_junction=True)
    if same_histogram:
        x=normalize_synthetic_hu_histograms(x)
    if endpoints_only:
        x=x.copy()
        x[:,0]=0
    return x,y,gt


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--train-samples",type=int,default=64)
    p.add_argument("--val-samples",type=int,default=24)
    p.add_argument("--test-samples",type=int,default=48)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--seeds",type=int,nargs="+",default=[17,41,73])
    p.add_argument("--side",type=int,default=24)
    p.add_argument("--device",default="cpu")
    p.add_argument("--histogram-control",action="store_true")
    p.add_argument("--endpoints-only",action="store_true")
    p.add_argument("--save-checkpoints",action="store_true")
    p.add_argument("--models",nargs="+",default=[
        "conv3d_sep_structure","anza3d_sep_structure","anza3d_contrast_sep_structure",
    ],choices=["conv3d_sep_structure","anza3d_sep_structure",
               "anza3d_contrast_sep_structure"])
    args=p.parse_args()
    if (args.out.exists() or args.train_samples<16 or args.val_samples<16 or
        args.test_samples<16 or args.side<12 or args.epochs<1 or
        not args.seeds or len(set(args.seeds))!=len(args.seeds)):
        raise ValueError("Invalid immutable JUNCTION pretraining protocol")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Selected GPU is not available")
    torch.set_num_threads(2)
    train=synthetic_data(args.train_samples,20261112,args.side,
                         same_histogram=args.histogram_control,
                         endpoints_only=args.endpoints_only)
    validation=synthetic_data(args.val_samples,20261912,args.side,
                              same_histogram=args.histogram_control,
                              endpoints_only=args.endpoints_only)
    independent=synthetic_data(args.test_samples,20261114,args.side,
                               same_histogram=args.histogram_control,
                               endpoints_only=args.endpoints_only)
    reports=[]
    for seed in args.seeds:
        for variant in args.models:
            score=fit(
                variant,"JUNCTION",train,validation,epochs=args.epochs,
                batch_size=8,seed=seed,device=args.device,
                checkpoint_path=(
                    args.out.parent/"checkpoints"/f"{variant}_seed{seed}.pt"
                    if args.save_checkpoints else None
                ),
                independent_test=independent,independent_paired_test=False,
            )
            assert score["independent_test"]["n_examples"]==args.test_samples
            reports.append(score)
            print(json.dumps({
                "variant":variant,"seed":seed,
                "test_AUROC":score["independent_test"]["AUROC"],
                "test_AUPRC":score["independent_test"]["AUPRC"],
                "true_accepted_at_085":score["independent_test"]["true_accepted_at_085"],
                "false_accepted_at_085":score["independent_test"]["wrong_accepted_at_085"],
            }),flush=True)
    summary={}
    for name in args.models:
        group=[r for r in reports if r["variant"]==name]
        auc=np.array([x["independent_test"]["AUROC"] for x in group])
        auprc=np.array([x["independent_test"]["AUPRC"] for x in group])
        summary[name]={
            "mean_test_AUROC":float(auc.mean()),
            "std_test_AUROC":float(auc.std(ddof=1) if len(auc)>1 else 0.),
            "mean_test_AUPRC":float(auprc.mean()),
            "seeds":[x["seed"] for x in group],
            "false_accepted_at_085_per_seed":[x["independent_test"]["wrong_accepted_at_085"] for x in group],
            "true_accepted_at_085_per_seed":[x["independent_test"]["true_accepted_at_085"] for x in group],
        }
    report={
        "status":"synthetic_only_JUNCTION_pretraining_no_CT28",
        "n_train":len(train[1]),"n_validation":len(validation[1]),
        "n_independent_test":len(independent[1]),
        "train_seed":20261112,"validation_seed":20261912,
        "test_seed":20261114,
        "clinical_data_accessed":False,
        "variants":args.models,"training_seeds":args.seeds,
        "epochs":args.epochs,"patch_side":args.side,
        "same_histogram":bool(args.histogram_control),
        "endpoints_only":bool(args.endpoints_only),
        "training_input_sha256":hashlib.sha256(train[0].tobytes()).hexdigest(),
        "validation_input_sha256":hashlib.sha256(validation[0].tobytes()).hexdigest(),
        "test_input_sha256":hashlib.sha256(independent[0].tobytes()).hexdigest(),
        "models":summary,"measurements":reports,
        "warning":"Generated JUNCTION topology labels are not medical 3D CT labels.",
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":report["status"],"models":summary},indent=2),flush=True)


if __name__=="__main__":
    main()
