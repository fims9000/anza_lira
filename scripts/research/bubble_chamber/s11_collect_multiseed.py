#!/usr/bin/env python3
"""Aggregate matched-frame S11 trials without pretending pixels are independent."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np


def counts_dice(counts):
    tp,fp,fn=map(int,counts)
    return 2*tp/max(2*tp+fp+fn,1)


def aggregate(reports:list[dict], *, boot_seed:int=33019, draws:int=2000):
    if len(reports)<2:raise ValueError('Require multiple independent model seeds')
    first=reports[0]
    arms=set(first['results'])
    sets=[]
    for report in reports:
        if set(report['results'])!=arms:raise ValueError('Missing arm or nonmatched ablation')
        if report['config']['data_seed']!=first['config']['data_seed']:
            raise ValueError('Data seed changed between model seeds')
        if report['source_meta']['split_fingerprints']!=first['source_meta']['split_fingerprints']:
            raise ValueError('Renderer/source-frame drift across model seeds')
        sets.append(int(report['config']['seed']))
    if len(set(sets))!=len(sets):raise ValueError('Model seeds must be different')
    sources=sorted(first['results'][next(iter(arms))]['evaluation']['clean']['per_frame_counts'])
    if len(sources)<5:raise ValueError('Not enough independent held-out frame groups')
    per={}
    for report in reports:
        for arm in arms:
            for cond in ('clean','stress'):
                frames=report['results'][arm]['evaluation'][cond]['per_frame_counts']
                if sorted(frames)!=sources:raise ValueError('Mismatched source frame set')
    for arm in sorted(arms):
        per[arm]={}
        for cond in ['clean','stress']:
            cubes=[];cld=[];seed_dice=[]
            for report in reports:
                val=report['results'][arm]['evaluation'][cond]
                counts=np.stack([val['per_frame_counts'][g] for g in sources])
                cubes.append(counts)
                seed_dice.append(counts_dice(counts.sum(0)))
                # Each measured S10 frame yields exactly one patch; enforce this
                # rather than assigning incorrect clDice to source frames.
                ids=list(val['per_frame_counts'])
                if len(ids)!=len(val['cldice_per_image']) or set(ids)!=set(sources):
                    raise ValueError('Expected one held-out patch per frame for macro clDice')
                idx={id:i for i,id in enumerate(ids)}
                cld.append(np.array([val['cldice_per_image'][idx[g]] for g in sources]))
            stacked=np.stack(cubes,axis=0)
            per[arm][cond]={'per_seed_dice':seed_dice,
                            'mean_seed_dice_descriptive':float(np.mean(seed_dice)),
                            'micro_dice_all_seeds':float(counts_dice(stacked.sum((0,1)))),
                            'per_seed_cldice_macro':[float(x.mean()) for x in cld],
                            'macro_cldice_mean':float(np.stack(cld).mean()),
                            'test_frame_counts':int(len(sources)),
                            '_counts':stacked,'_cldice':np.stack(cld)}
    rng=np.random.default_rng(boot_seed)
    frames_idx=rng.integers(0,len(sources),size=(draws,len(sources)))
    contrasts=[('az_cldice','az_base'),('az_cldice_axis','az_cldice'),
              ('az_cldice_axis','unet_cldice'),('az_cldice','unet_cldice'),
              ('unet_cldice','unet_base')]
    deltas={}
    for condition in ('clean','stress'):
        for cand,base in contrasts:
            if cand not in arms or base not in arms:continue
            c=per[cand][condition];b=per[base][condition]
            cb=c['_counts'][:,frames_idx,:].sum(axis=(0,2))
            bb=b['_counts'][:,frames_idx,:].sum(axis=(0,2))
            dc=2*cb[:,0]/np.maximum(2*cb[:,0]+cb[:,1]+cb[:,2],1)
            db=2*bb[:,0]/np.maximum(2*bb[:,0]+bb[:,1]+bb[:,2],1)
            ca=c['_cldice'][:,frames_idx].mean(axis=(0,2))
            ba=b['_cldice'][:,frames_idx].mean(axis=(0,2))
            key=f'{condition}_{cand}_minus_{base}'
            deltas[key]={'delta_micro_dice':c['micro_dice_all_seeds']-b['micro_dice_all_seeds'],
                         'delta_micro_dice_ci95':[float(x) for x in np.quantile(dc-db,[.025,.975])],
                         'delta_macro_cldice':c['macro_cldice_mean']-b['macro_cldice_mean'],
                         'delta_macro_cldice_ci95':[float(x) for x in np.quantile(ca-ba,[.025,.975])],
                         'bootstrap_unit':'source_frame_with_all_seeds_kept',
                         'frames':len(sources),'model_seeds':len(reports)}
    cleaned={name:{cond:{k:v for k,v in stats.items() if not k.startswith('_')}
                       for cond,stats in conds.items()} for name,conds in per.items()}
    return {'schema':'S11_FIXED_SPLIT_THREE_SEED_AGGREGATE_V1','model_seeds':sets,
            'data_seed':first['config']['data_seed'],'source':first['source_meta'],
            'arms':cleaned,'comparisons':deltas,'bootstrap_draws':draws,
            'limitations':['Only film070 real measured geometry, all image pixels are generated',
                           'Noise and appearance are simulated, not calibrated to H50Z',
                           'Frame bootstrap conditions on selected model seeds and shared physical film',
                           'Not a false-bridge, true particle ID, momentum or mass accuracy measurement']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reports',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    d=aggregate([json.loads(f.read_text()) for f in args.reports])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
    print('S11_MULTI_SEED',args.out)
    for name,x in d['arms'].items():
        print(name,'clean',round(x['clean']['mean_seed_dice_descriptive'],4),
                   'stress',round(x['stress']['mean_seed_dice_descriptive'],4),
                   'stress_cldice',round(x['stress']['macro_cldice_mean'],4))

if __name__=='__main__':main()
