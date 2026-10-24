#!/usr/bin/env python3
"""Fixed-source, varying-initialization comparison of original AZUNet vs U-Net.

This is a *synthetic image* benchmark rendered from real expert-clicked geometry.
Three model randomizations condition on the same scene/render/test frames, so
uncertainty from data and model-seed variation must not be conflated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEEDS=(20261024,20261025,20261026)
CONDITIONS=('clean','stress')
MODELS=('plain_unet','original_azunet')


def dice_from_counts(counts):
    c=np.asarray(counts,dtype=np.float64)
    tp,fp,fn=c.sum(axis=0)
    return float(2*tp / max(2*tp+fp+fn,1))


def paired_frame_bootstrap(rows: list[dict], condition: str, *, seed: int=20261024,
                           repetitions: int=2000) -> dict:
    """Resample 25 original *frames*, keeping all model seeds and methods paired."""
    if not rows: raise ValueError('no rows')
    reference=rows[0]['test_groups']
    if len(reference)!=len(set(reference)) or len(reference)<2:
        raise ValueError('Expected unique original frames in held-out set')
    if any(row['test_groups']!=reference for row in rows):
        raise ValueError('Test frame IDs differ across model seeds')
    y=[]
    for row in rows:
        c=row['conditions'][condition]
        a=np.asarray(c['original_azunet'],dtype=np.int64)
        b=np.asarray(c['plain_unet'],dtype=np.int64)
        if a.shape!=(len(reference),3) or b.shape!=a.shape:
            raise ValueError('bad per-frame confusion array')
        # Each frame's positive ground truth must match across models and seeds.
        if not np.array_equal(a[:,0]+a[:,2],b[:,0]+b[:,2]):
            raise ValueError('mismatched masks across methods')
        y.append(a[:,0]+a[:,2])
    if any(not np.array_equal(y[0],v) for v in y[1:]):
        raise ValueError('Test labels do not match across model seeds')
    aa=np.stack([np.asarray(r['conditions'][condition]['original_azunet'],np.int64) for r in rows])
    bb=np.stack([np.asarray(r['conditions'][condition]['plain_unet'],np.int64) for r in rows])
    full=dice_from_counts(aa.reshape(-1,3))-dice_from_counts(bb.reshape(-1,3))
    rng=np.random.default_rng(seed)
    results=np.zeros(repetitions,dtype=np.float64)
    for i in range(repetitions):
        chosen=rng.integers(0,len(reference),size=len(reference))
        results[i]=dice_from_counts(aa[:,chosen,:].reshape(-1,3))-dice_from_counts(bb[:,chosen,:].reshape(-1,3))
    return {'delta_pooled_micro_dice':float(full),
            'ci95_by_test_source_frame':[float(z) for z in np.percentile(results,[2.5,97.5])],
            'bootstrap_repeats':repetitions, 'resampling_unit':'source frame, all seeds paired',
            'n_unique_test_frames':len(reference),
            'model_seeds':[int(r['model_seed']) for r in rows],
            'conditional_on_these_model_seeds':True}


def sha(path:Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root:Path, *, seeds=SEEDS, data_seed=20261024, repeats=2000):
    rows=[];metrics=[]
    canonical=None
    for seed in seeds:
        path=root/'results'/f's10_fixed_seed{seed}'
        m=json.loads((path/'metrics.json').read_text())
        row=json.loads((path/'s10_test_frame_counts.json').read_text())
        if row['model_seed']!=seed or row['data_seed']!=data_seed:
            raise ValueError('S10 model/data seed mismatch')
        if m['config'].get('data_seed')!=data_seed or m['config']['seed']!=seed:
            raise ValueError('Config does not preserve fixed data split')
        if m['source_data']['real_pixel_labels'] is not False or m['film_holdout'] is not False:
            raise ValueError('Expected synthetic masks from a single film')
        if not (m['config']['az_residual'] and m['config']['robust_train']):
            raise ValueError('S10 robust+AZ residual configuration required')
        for name in MODELS:
            model=m['results'][name]
            if model['trained_epochs']!=8:raise ValueError('Training unfinished')
            cp=path/name/'best_final.pt'
            if not cp.exists():raise ValueError(f'Missing checkpoint {cp}')
        fingerprints={k:m['source_data'][f'{k}_examples']['sha256'] for k in ('train','val','test')}
        if canonical is None: canonical=fingerprints
        if canonical!=fingerprints:raise ValueError('Data/GT differs between seeds')
        rows.append(row);metrics.append(m)
    frangi=json.loads((root/'results'/'s10_fixed_frangi.json').read_text())
    if frangi['source_fingerprints']['test']['sha256']!=canonical['test'] or frangi['source_fingerprints']['val']['sha256']!=canonical['val']:
        raise ValueError('Frangi source or val differs from networks')
    per_run=[]
    for seed,m in zip(seeds,metrics):
        per_run.append({'seed':seed,'model_scores':{
            name:{'dice_clean':float(m['results'][name]['test_clean']['dice']),
                  'dice_stress':float(m['results'][name]['test_stress']['dice']),
                  'cldice_clean':float(m['results'][name]['test_clean_topology']['macro_cldice']),
                  'cldice_stress':float(m['results'][name]['test_stress_topology']['macro_cldice']),
                  'time_seconds':float(m['results'][name]['elapsed_seconds_this_invocation']),
                  'checkpoint_sha256':sha(root/'results'/f's10_fixed_seed{seed}'/name/'best_final.pt')}
            for name in MODELS}})
    paired={key:paired_frame_bootstrap(rows,key,seed=data_seed+991,repetitions=repeats) for key in CONDITIONS}
    means={name:{'clean':float(np.mean([r['model_scores'][name]['dice_clean'] for r in per_run])),
                 'stress':float(np.mean([r['model_scores'][name]['dice_stress'] for r in per_run]))}
           for name in MODELS}
    return {'experiment':'S10_FIXED_SOURCE_RANDOM_INITIALIZATION_ORIGINAL_AZUNET',
            'status':'EXPLORATORY_NO_REAL_PIXEL_GT',
            'data_seed':data_seed,'model_seeds':list(seeds),
            'data_provenance_sha256':metrics[0]['source_data']['audit_sha256'],
            'fixed_split_fingerprints':canonical,
            'n_test_original_frames':len(rows[0]['test_groups']),
            'film_id':'070','original_pixels_available_in_training':False,
            'source_blobs':metrics[0]['original_source_git_blobs'],
            'per_model_seed':per_run,'mean_dice_per_model':means,
            'frangi_fixed_data':{'clean':frangi['clean']['dice'],'stress':frangi['stress']['dice'],
                                 'threshold_val':frangi['threshold_selected_on_validation']},
            'paired_az_minus_unet':paired,
            'limitations':['Synthetic pixels from real measured trajectories, no verified real-film masks',
                           'All frames belong to a single film (070); no independent-camera holdout',
                           'Only three model-initialization seeds, same small network widths and 8 epochs',
                           'Bootstrap resamples the fixed 25 test frames conditioned on the three seeds',
                           'Cannot infer universal ANZA superiority from synthetic benchmark']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=2000)
    a=p.parse_args()
    result=collect(a.root,repeats=a.repeats)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'means':result['mean_dice_per_model'],
                      'delta':result['paired_az_minus_unet']},indent=2))

if __name__=='__main__':main()
