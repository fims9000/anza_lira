#!/usr/bin/env python3
"""Frangi on identical S9 synthetic-from-clicks splits. Real BMP pixel truth absent."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from skimage.filters import frangi
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s7_original_retrain import (
    RunConfig,make_datasets,apply_stress,metrics_counts,metric_from_counts,split_fingerprints)

def filter_one(x):
    return frangi(np.asarray(x,dtype=np.float32),sigmas=(1,2,3),black_ridges=False).astype(np.float32)

def evaluate(sets,cfg):
    val,test=sets["val"],sets["test"]
    v=np.stack([filter_one(x) for x,_,_ in val])
    vs=np.stack([filter_one(apply_stress(torch.tensor(x[None,None],dtype=torch.float32),
             torch.tensor([i]),cfg)[0,0].numpy()) for i,(x,_,_) in enumerate(val)])
    yv=np.stack([y for _,y,_ in val])
    thresholds=np.geomspace(.0001,.5,81)
    scores=[]
    for t in thresholds:
        clean=metric_from_counts(metrics_counts(v>=t,yv>=.5))["dice"]
        noisy=metric_from_counts(metrics_counts(vs>=t,yv>=.5))["dice"]
        scores.append((min(clean,noisy),.5*(clean+noisy)))
    best=max(range(len(scores)),key=lambda i:scores[i])
    t=float(thresholds[best])
    normal=np.stack([filter_one(x) for x,_,_ in test])
    stress=np.stack([filter_one(apply_stress(torch.tensor(x[None,None],dtype=torch.float32),
                 torch.tensor([i]),cfg)[0,0].numpy()) for i,(x,_,_) in enumerate(test)])
    yt=np.stack([y for _,y,_ in test])
    return {"threshold_selected_on_validation":t,
        "validation_objective_worst_condition_dice":float(scores[best][0]),
        "clean":metric_from_counts(metrics_counts(normal>=t,yt>=.5)),
        "stress":metric_from_counts(metrics_counts(stress>=t,yt>=.5)),
        "n_val":len(val),"n_test":len(test),
        "unique_test_frames":len({e[2] for e in test}),
        "source_fingerprints":split_fingerprints(sets)}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--measurements",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261024)
    a=p.parse_args()
    cfg=RunConfig(source="visualmetro",seed=a.seed,size=64,train_limit=120,val_limit=26,test_limit=25,
                  widths=(4,6,8,12),num_rules=2,robust_train=True,az_residual=True)
    ds,source=make_datasets(cfg,a.measurements)
    result=evaluate(ds,cfg)
    result.update({"seed":a.seed,"source_kind":source["source_kind"],"pixel_ground_truth_real":False})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"clean_dice":result["clean"]["dice"],"stress_dice":result["stress"]["dice"]}))
