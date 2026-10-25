#!/usr/bin/env python3
"""S33 fixed synthetic track-vs-scratch benchmark on heldout H50Z photo 006.

Tracks/scratches in the experiment are ADDED ARTIFICIALLY to authentic images.
Nothing here measures true (unknown) native photo segmentation accuracy.
"""
from __future__ import annotations
import argparse, json,sys
from pathlib import Path
import numpy as np
import cv2
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s33_photometric_domain import image_input
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model,verify_model_source
from scripts.research.bubble_chamber.s32_heldout_injection_control import independent_injections
from scripts.research.bubble_chamber.s26_two_photo_baselines import classical_score

TRAIN_SEED_BY_INJECTION={20261029:20261025,20261030:20261026,20261031:20261025,20261032:20261026}
MODES=('raw','quantile','local','raw_aug')
ARCH=('azunet','unet')


def load_model(folder:Path,arch:str,mode:str,seed:int):
    path=folder/f'{arch}_{mode}_seed{seed}.pt'
    checkpoint=torch.load(path,weights_only=True,map_location='cpu')
    if any([checkpoint.get('schema')!='s33_machine_draft_photo_normalized_v1',
            checkpoint.get('architecture')!=arch,checkpoint.get('mode')!=mode,
            checkpoint.get('seed')!=seed,checkpoint.get('photo_train')!='005',
            checkpoint.get('physical_auto_joins_authorized') is not False]):
        raise ValueError('S33 checkpoint provenance/physical safety mismatch')
    model=build_model(arch,widths=(4,6,8,12),rules=2)
    model.load_state_dict(checkpoint['model_state']);model.eval()
    return model


def photo_perturb(x:np.ndarray,mode:str)->np.ndarray:
    if mode=='identity':return x
    if mode=='bright':return np.clip(x.astype(np.float32)*.80+30,0,255).astype(np.uint8)
    if mode=='dark':return np.clip(x.astype(np.float32)*1.15-28,0,255).astype(np.uint8)
    raise ValueError('Unsupported fixed test perturbation')


def evaluate_fixed_seed(tiles:Path,measurements:Path,folder:Path,*,seed:int,n:int=48):
    if seed not in TRAIN_SEED_BY_INJECTION:raise ValueError('Unregistered injection seed')
    verify_model_source();torch.set_num_threads(2)
    train_seed=TRAIN_SEED_BY_INJECTION[seed]
    cases,provenance=independent_injections(tiles,measurements,seed=seed,n=n)
    models={arch+'_'+mode:(mode,load_model(folder,arch,mode,train_seed))
            for arch in ARCH for mode in MODES}
    records=[]
    with torch.no_grad():
        for perturb in ('identity','bright','dark'):
            pooled={name:[] for name in list(models)+['steerable_ridge','local_contrast']}
            y=[];groups=[];clipped=[]
            for idx,case in enumerate(cases):
                fg=case.synthetic_track_mask.astype(bool)
                neg=case.synthetic_scratch_mask.astype(bool)
                if (fg&neg).any():raise AssertionError('Synthetic labels overlap')
                known=fg|neg
                gray=photo_perturb(case.gray,perturb)
                clipped.append(float((gray==0).mean()+(gray==255).mean()))
                y.extend(fg[known].astype(int).tolist());groups.extend([idx]*int(known.sum()))
                precomputed={q:torch.from_numpy(image_input(gray,q)[None,None]) for q in ('raw','quantile','local')}
                for name,(mode,model) in models.items():
                    input_mode='raw' if mode=='raw_aug' else mode
                    p=torch.sigmoid(model(precomputed[input_mode]))[0,0].numpy()
                    pooled[name].extend(p[known].tolist())
                for name in ('steerable_ridge','local_contrast'):
                    pooled[name].extend(classical_score(gray,name)[known].tolist())
            y=np.asarray(y,dtype=int);groups=np.asarray(groups,dtype=int)
            if set(np.unique(y))!={0,1}:raise ValueError('Missing injected control class')
            vals={name:np.asarray(scores,float) for name,scores in pooled.items()}
            metrics={}
            for name,scores in vals.items():
                if len(scores)!=len(y) or not np.isfinite(scores).all():raise ValueError('bad test predictions')
                metrics[name]={'auroc_injected_only':float(roc_auc_score(y,scores)),
                    'auprc_injected_only':float(average_precision_score(y,scores))}
            records.append({'perturbation':perturb,'seed':seed,'train_seed':train_seed,
                'n_injected_scenes':len(cases),'positive_injected_pixels':int(y.sum()),
                'negative_injected_scratch_pixels':int(len(y)-sum(y)),
                'clip_fraction':float(np.mean(clipped)),'metrics':metrics})
            # Paired test is applied on unaltered photo 006 (new seeds held out).
            if perturb=='identity':
                rows=[]
                for arch in ARCH:
                    baseline=vals[arch+'_raw'];base_auc=metrics[arch+'_raw']['auroc_injected_only']
                    for mode in ('quantile','local','raw_aug'):
                        alternative=vals[arch+'_'+mode]
                        delta=metrics[arch+'_'+mode]['auroc_injected_only']-base_auc
                        rng=np.random.default_rng(seed+37);bootstrap=[]
                        for _ in range(260):
                            ids=rng.integers(0,n,size=n)
                            w=np.bincount(ids,minlength=n)[groups]
                            positive=w>0
                            if len(np.unique(y[positive]))!=2:continue
                            a=roc_auc_score(y[positive],alternative[positive],sample_weight=w[positive]);b=roc_auc_score(y[positive],baseline[positive],sample_weight=w[positive])
                            bootstrap.append(float(a-b))
                        ci=[float(v) for v in np.quantile(bootstrap,[.025,.975])]
                        rows.append({'architecture':arch,'alternative':mode,
                                     'delta_AUROC_vs_raw':float(delta),
                                     'paired_patch_bootstrap_delta_CI95':ci,
                                     'source_photo':'006','independent_run':False})
                paired=rows
    return {'schema':'s33_matched_heldout_synthetic_injection_v1',
        'injection_seed':seed,'train_seed':train_seed,'provenance':provenance,
        'native_photo_real_track_quality_measured':False,
        'synthetic_injected_trace_truth_only':True,'real_expert_track_mask_count':0,
        'independent_physical_run_count':0,'source_photo_test':'006',
        'test_values_not_used_for_training_or_epoch_selection':True,
        'actual_physics_auto_join_permitted':False,
        'evaluation':records,'paired_patch_bootstrap':paired}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--tiles',type=Path,required=True)
    p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--n',type=int,default=48)
    a=p.parse_args()
    r=evaluate_fixed_seed(a.tiles,a.measurements,a.models,seed=a.seed,n=a.n)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
    for x in r['evaluation']:
        print('S33',a.seed,x['perturbation'],{k:round(v['auroc_injected_only'],4) for k,v in x['metrics'].items()},flush=True)
