"""Paired frame-cluster uncertainty for frozen held-out S5 model predictions.

Post-hoc descriptive diagnostic only. Never tune hyperparameters or choose
operating thresholds from these paired held-out statistics.
"""
from __future__ import annotations
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np


def paired_frame_delta(csv_path:Path, model_a:str, model_b:str, seed:int=20261024, repetitions:int=2000):
    rows=list(csv.DictReader(Path(csv_path).open(newline='',encoding='utf-8')))
    a={};b={}
    for row in rows:
        if row['model'] not in (model_a,model_b):continue
        key=tuple(row[k] for k in ('frame_id','view','scene_id','left_id','right_id','candidate_rank'))
        entry=(row['truth_same_track'].lower() in {'true','1'},row['accepted_offline'].lower() in {'true','1'})
        if row['model']==model_a:
            if key in a:raise ValueError('duplicate A prediction')
            a[key]=entry
        if row['model']==model_b:
            if key in b:raise ValueError('duplicate B prediction')
            b[key]=entry
    if not a or set(a)!=set(b):
        raise ValueError('model predictions not paired on same candidates')
    by_frame=defaultdict(lambda:np.zeros(5,dtype=float))
    for key, (true_a, pred_a) in a.items():
        true_b,pred_b=b[key]
        if true_a!=true_b: raise ValueError('ground truth mismatch')
        vec=by_frame[key[0]]
        vec[0]+=int(true_a)
        vec[1]+=int(true_a and pred_a)
        vec[2]+=int(true_a and pred_b)
        vec[3]+=int(not true_a and pred_a)
        vec[4]+=int(not true_a and pred_b)
    counts=np.stack(list(by_frame.values()))
    if repetitions<1:raise ValueError('repetitions must be positive')
    total=counts.sum(axis=0)
    delta_recall=(total[2]-total[1])/max(total[0],1)
    rng=np.random.default_rng(seed)
    sample=rng.integers(0,len(counts),size=(repetitions,len(counts)))
    summed=counts[sample].sum(axis=1)
    rec=(summed[:,2]-summed[:,1])/np.maximum(summed[:,0],1)
    false=(summed[:,4]-summed[:,3])/len(counts)
    return {
        'test_frames':len(counts),
        'positive_candidates':int(total[0]),
        'A':model_a,'B':model_b,
        'test_tp_A':int(total[1]),'test_tp_B':int(total[2]),
        'test_fp_A':int(total[3]),'test_fp_B':int(total[4]),
        'B_minus_A_recall':round(float(delta_recall),6),
        'B_minus_A_recall_frame_bootstrap_ci95':[round(float(v),6) for v in np.quantile(rec,[.025,.975])],
        'B_minus_A_false_links_per_frame':round(float((total[4]-total[3])/len(counts)),6),
        'B_minus_A_false_links_per_frame_ci95':[round(float(v),6) for v in np.quantile(false,[.025,.975])],
        'interpretation':'within-film, model comparison exploratory: this is not segmentation performance',
    }
