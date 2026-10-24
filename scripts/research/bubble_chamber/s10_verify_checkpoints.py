#!/usr/bin/env python3
"""Recompute exact held-out Dice from S10 best_final.pt, without retraining.

Only open checkpoints created by this codebase; torch.load(..., weights_only=False)
uses pickle and must not be used on untrusted files.
"""
from __future__ import annotations
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s7_original_retrain import (
    RunConfig,verify_sources,build_models,make_datasets,split_fingerprints,
    TrackDataset,predict_probabilities,metric_from_counts,metrics_counts,
)


def replay(root:Path,measurements:Path,seeds=(20261024,20261025,20261026)):
    verify_sources(ROOT)
    findings=[]
    for seed in seeds:
        out=root/'results'/f's10_fixed_seed{seed}'
        expected=json.loads((out/'metrics.json').read_text())
        cfg=RunConfig(**expected['config'])
        if cfg.seed!=seed or cfg.data_seed!=20261024:
            raise ValueError('Unexpected data seed or model seed')
        torch.set_num_threads(cfg.cpu_threads)
        sets,source=make_datasets(cfg,measurements)
        fresh=split_fingerprints(sets)
        for part in ('train','val','test'):
            if fresh[part]['sha256']!=expected['source_data'][f'{part}_examples']['sha256']:
                raise ValueError(f'{seed} dataset {part} SHA mismatch')
        models=build_models(cfg)
        for name in ('plain_unet','original_azunet'):
            path=out/name/'best_final.pt'
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            if checkpoint['model']!=name or checkpoint['source_blobs']!=expected['original_source_git_blobs']:
                raise ValueError('Model/source fingerprint mismatch')
            if checkpoint['data_fingerprints']!=fresh:
                raise ValueError('Checkpoint training data differ from current data')
            model=models[name]
            model.load_state_dict(checkpoint['state_dict'],strict=True)
            model.eval()
            threshold=float(checkpoint['validation_selected_threshold'])
            scores={}
            data=TrackDataset(sets['test'])
            for condition in ('clean','stress'):
                probs,labels=predict_probabilities(model,data,cfg,stress=condition=='stress')
                metric=metric_from_counts(metrics_counts(probs>=threshold,labels>=0.5))
                recorded=expected['results'][name][f'test_{condition}']
                for key in ('tp','fp','fn','dice','precision','recall'):
                    if key in ('tp','fp','fn'):
                        if int(metric[key])!=int(recorded[key]):
                            raise AssertionError(f'{seed} {name} {condition} {key} differs')
                    elif not np.isclose(metric[key],recorded[key],atol=1e-12,rtol=0):
                        raise AssertionError(f'{seed} {name} {condition} {key} differs')
                scores[condition]=metric['dice']
            findings.append({'seed':seed,'model':name,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                             'checkpoint_verified':True,'clean_dice':scores['clean'],'stress_dice':scores['stress']})
    return {'status':'FULL_CHECKPOINT_REPLAY_PASSED','checkpoints':findings,'source':'synthetic pixels from real measured geometry',
            'real_camera_accuracy_proven':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    result=replay(args.root,args.measurements)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'models_checked':len(result['checkpoints'])}))

if __name__=='__main__':main()
