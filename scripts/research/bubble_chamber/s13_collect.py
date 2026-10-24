#!/usr/bin/env python3
"""Frame-clustered comparison of frozen S13 matched-gate ANZA experiments."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.research.bubble_chamber.s12_collect_multiseed import pooled

CONTRASTS=(('az_guided','unet_base'),('az_guided','conv_guided'),('az_guided','az_pure'),('conv_guided','unet_base'))

def collect(paths:list[Path],out:Path,*,draws:int=2000):
    reports={}
    for p in paths:
        r=json.loads(p.read_text())
        if r.get('schema')!='S13_ORIGINAL_AZ_GUIDED_UNET_AND_CONTROLS_V1':raise ValueError('Unexpected S13 report schema')
        seed=int(r['config']['seed'])
        if seed in reports:raise ValueError('Duplicate model seed')
        reports[seed]=r
    if len(reports)<2:raise ValueError('Need multiple model initializations')
    exemplar=next(iter(reports.values()))
    for r in reports.values():
        if r['data_fingerprints']!=exemplar['data_fingerprints'] or r['exact_source_blob_sha']!=exemplar['exact_source_blob_sha']:
            raise ValueError('Frame split, synthetic inputs or source code changed')
        if r['config']['data_seed']!=exemplar['config']['data_seed'] or set(r['results'])!=set(exemplar['results']):
            raise ValueError('Mismatched fixed data seed or ablation arms')
    arms={k:v['results'] for k,v in reports.items()}
    result={'schema':'S13_FIXED_FRAMES_MULTI_SEED_V1','model_seeds':sorted(reports),
            'data_seed':exemplar['config']['data_seed'],'source':exemplar['source'],
            'source_blob_sha':exemplar['exact_source_blob_sha'],'data_fingerprints':exemplar['data_fingerprints'],
            'conditions':{},'comparisons':{},'limitations':['One film 070; genuine manual click geometry but artificial pixels/masks',
                'No real-image or instance-level false joining evidence','Bootstrap resamples whole frames, conditions on chosen 3 model seeds']}
    for condition in ('clean','stress'):
        result['conditions'][condition]={}
        for arm in sorted(exemplar['results']):
            vals=[reports[s]['results'][arm]['test'][condition]['counts']['dice'] for s in sorted(reports)]
            cls=[reports[s]['results'][arm]['test'][condition]['macro_cldice'] for s in sorted(reports)]
            result['conditions'][condition][arm]={'dice_by_seed':vals,'dice_mean_descriptive':float(np.mean(vals)),
                      'cldice_by_seed':cls,'cldice_mean_descriptive':float(np.mean(cls))}
        for cand,base in CONTRASTS:
            if cand in exemplar['results'] and base in exemplar['results']:
                result['comparisons'][f'{condition}:{cand}-minus-{base}']=pooled(arms,base,cand,condition=condition,draws=draws)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result

def main():
    a=argparse.ArgumentParser();a.add_argument('--metrics',type=Path,nargs='+',required=True)
    a.add_argument('--out',type=Path,required=True);args=a.parse_args()
    r=collect(args.metrics,args.out)
    for cond,models in r['conditions'].items():
        print(cond,{k:round(v['dice_mean_descriptive'],4) for k,v in models.items()})
    for k,v in r['comparisons'].items():
        print(k,round(v['pooled_dice_difference'],5),[round(x,5) for x in v['frame_cluster_CI95']])
if __name__=='__main__':main()
