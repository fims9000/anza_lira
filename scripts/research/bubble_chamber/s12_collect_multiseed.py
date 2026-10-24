#!/usr/bin/env python3
"""S12 paired 3-seed evaluation on fixed film-070 source frames.

The uncertainty interval resamples 25 source frames, retaining *all* model
seeds within each frame. It is conditional on the three chosen training seeds
and one source film, not a guarantee of real-camera or other-run transfer.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np


def dice(count):
    tp,fp,fn=np.asarray(count,dtype=np.int64)
    return 2*int(tp)/max(2*int(tp)+int(fp)+int(fn),1)


def pooled(arms:dict,base:str,alternate:str,*,condition:str,seed:int=17,draws:int=2000):
    common=sorted(arms)
    if not common:raise ValueError('No seeds')
    records=[];frame_ids=None
    for run in common:
        models=arms[run]
        b=models[base]['test'][condition]
        a=models[alternate]['test'][condition]
        x={k:np.asarray(v,dtype=np.int64) for k,v in b['per_frame_counts'].items()}
        y={k:np.asarray(v,dtype=np.int64) for k,v in a['per_frame_counts'].items()}
        if not x or set(x)!=set(y):raise ValueError('Mismatched source-frame identities')
        if frame_ids is None:frame_ids=sorted(x)
        elif frame_ids!=sorted(x):raise ValueError('Different frame split across model seeds')
        records.append((np.stack([x[k] for k in frame_ids]),np.stack([y[k] for k in frame_ids])))
    b=np.stack([x[0] for x in records]).sum(0)
    a=np.stack([x[1] for x in records]).sum(0)
    real_diff=dice(a.sum(0))-dice(b.sum(0))
    rng=np.random.default_rng(seed)
    pick=rng.integers(0,len(frame_ids),(draws,len(frame_ids)))
    av=a[pick].sum(1);bv=b[pick].sum(1)
    delta=(2*av[:,0]/np.maximum(2*av[:,0]+av[:,1]+av[:,2],1)-
           2*bv[:,0]/np.maximum(2*bv[:,0]+bv[:,1]+bv[:,2],1))
    return {'pooled_dice_difference':float(real_diff),'frame_cluster_CI95':list(map(float,np.quantile(delta,[.025,.975]))),
            'bootstrap_p_delta_gt_zero':float(np.mean(delta>0)),
            'test_frames':len(frame_ids),'model_seeds':len(common),'bootstrap_draws':draws,
            'pooled_baseline_dice':dice(b.sum(0)),'pooled_candidate_dice':dice(a.sum(0)),
            'ablation':f'{alternate} - {base}', 'conditional_on_film':True}


def collect(paths:list[Path],out:Path):
    reports={}
    for path in paths:
        d=json.loads(path.read_text())
        if d['schema']!='S12_DIRECT_GT_TANGENT_TO_AUTHENTIC_AZCONV_V1':
            raise ValueError(f'Unexpected study schema: {path}')
        seed=int(d['config']['seed'])
        if seed in reports:raise ValueError('Duplicate model-seed')
        reports[seed]=d
    if not reports:raise ValueError('No study metrics')
    exemplar=next(iter(reports.values()))
    common_fingerprints=exemplar['data_fingerprints']
    arms=set(exemplar['results'])
    for report in reports.values():
        if report['data_fingerprints']!=common_fingerprints:
            raise ValueError('Data or train/test groups changed across model seeds')
        if report['source_git_blobs']!=exemplar['source_git_blobs']:
            raise ValueError('Original source changed between model seeds')
        if set(report['results'])!=arms:
            raise ValueError('Different model ablations')
        for arm in arms:
            a=exemplar['results'][arm]['test']['clean']['source_frames']
            b=report['results'][arm]['test']['clean']['source_frames']
            if a!=b:raise ValueError('Test source frame groups differ')
    structs={seed:report['results'] for seed,report in reports.items()}
    summary={'schema':'S12_FIXED_FRAME_THREE_INITIALIZATIONS_V1',
        'model_seeds':sorted(reports),'data_fingerprints':common_fingerprints,
        'source_git_blobs':exemplar['source_git_blobs'],
        'paper_claim':'Synthetic-pixel segmentation only; no independent photo labels or particle track identities.',
        'condition_means':{},'paired':{}}
    for condition in ('clean','stress'):
        summary['condition_means'][condition]={}
        for arm in sorted(arms):
            ds=[reports[seed]['results'][arm]['test'][condition]['counts']['dice'] for seed in sorted(reports)]
            cd=[reports[seed]['results'][arm]['test'][condition]['macro_cldice'] for seed in sorted(reports)]
            summary['condition_means'][condition][arm]={'individual_dice':ds,'mean_dice_descriptive':float(np.mean(ds)),
                'mean_macro_cldice_descriptive':float(np.mean(cd))}
        for variant,base in [('az_direct05','az_base'),('az_direct_cldice05','az_base'),('az_direct_cldice05','az_direct05'),
                             ('az_direct05','unet_base'),('az_direct_cldice05','unet_base'),('unet_cldice05','unet_base')]:
            summary['paired'][f'{condition}:{variant}-minus-{base}']=pooled(structs,base,variant,condition=condition)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--metrics',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    r=collect(args.metrics,args.out)
    for condition,arms in r['condition_means'].items():
        print(condition,{name:round(val['mean_dice_descriptive'],4) for name,val in arms.items()})
    for name,stats in r['paired'].items():
        print(name,'delta',round(stats['pooled_dice_difference'],5),'CI',list(round(x,5) for x in stats['frame_cluster_CI95']))

if __name__=='__main__':main()
