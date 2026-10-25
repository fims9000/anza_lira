#!/usr/bin/env python3
"""S35 heldout synthetic photo-gap test, both directions, fixed unseen seeds.

Every test label is a GENERATED overlay track or GENERATED scratch. No native
H50Z track pixels or real physical track joins are evaluated. All original
photo pixels not altered by synthetic injection remain UNKNOWN to metrics.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s35_photometric_consistency import S35Protocol,view_disagreement
from bubble_chamber.s33_photometric_domain import image_input
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model,verify_model_source
from scripts.research.bubble_chamber.s33_evaluate_photometric import photo_perturb
from scripts.research.bubble_chamber.s32_heldout_injection_control import independent_injections
from scripts.research.bubble_chamber.s33_reverse_validation import generate_holdout_005
from scripts.research.bubble_chamber.s35_train_consistency import ARM_BY_ARCH


def _model_load(folder,arch,branch,source_photo,seed):
    path=Path(folder)/f'{arch}_{branch}_train{source_photo}_seed{seed}.pt'
    ck=torch.load(path,map_location='cpu',weights_only=True)
    required={'schema':'s35_machine_only_photo_view_consistency_v1',
        'architecture':arch,'mode':ARM_BY_ARCH[arch],'train_photo':source_photo,
        'test_photo_not_used_for_training':('006' if source_photo=='005' else '005'),
        'seed':seed,'branch':branch,'labels_are_human_certified':False,
        'physical_auto_joins_authorized':False}
    if any(ck.get(k)!=v for k,v in required.items()):raise ValueError('Mismatched source model identity/provenance '+str(path))
    model=build_model(arch,widths=(4,6,8,12),rules=2)
    model.load_state_dict(ck['model_state']);model.eval()
    return model


def _batched_prob(model,images,mode,batchsize=16):
    if not images:return []
    allp=[]
    with torch.no_grad():
        for i in range(0,len(images),batchsize):
            inp=np.stack([image_input(x,mode) for x in images[i:i+batchsize]])[:,None]
            p=torch.sigmoid(model(torch.from_numpy(inp)))[...,0,:,:].numpy()
            allp.extend(p)
    return allp


def evaluate_source_direction(tiles,measurements,models,*,train_photo:str,protocol:S35Protocol=S35Protocol(),n=48):
    protocol.check();verify_model_source();torch.set_num_threads(2)
    if train_photo not in ('005','006') or n!=48:raise ValueError('Predeclared S35 independent-injection budget')
    test_photo='006' if train_photo=='005' else '005'
    seeds=(protocol.test_forward_seeds if train_photo=='005' else protocol.test_reverse_seeds)
    result={'schema':'s35_cross_photo_machine_aug_consistency_synthetic_only_v1',
        'protocol':vars(protocol),'train_photo':train_photo,'test_photo':test_photo,
        'train_labels':'S31 MACHINE ONLY, not expert','real_photo_track_ground_truth':False,
        'test_bmp_tracks_human_verified':0,'physical_auto_join_authorized':False,
        'test_photo_used_for_training':False,'same_physical_run':True,
        'registered_test_injections_per_seed':48,'results':[]}
    for j,seed in enumerate(seeds):
        train_seed=protocol.train_seeds[j%len(protocol.train_seeds)]
        if train_photo=='005':cases,provenance=independent_injections(tiles,measurements,seed=seed,n=n)
        else:cases,provenance=generate_holdout_005(tiles,measurements,seed,n)
        models_loaded={f'{arch}_{branch}':(arch,_model_load(models,arch,branch,train_photo,train_seed))
            for arch in ('azunet','unet') for branch in ('matched_two_view','photo_consistency')}
        for perturb in ('identity','bright','dark'):
            imgs=[photo_perturb(x.gray,perturb) for x in cases]
            valid=[(x.synthetic_track_mask>0)|(x.synthetic_scratch_mask>0) for x in cases]
            y=np.concatenate([x.synthetic_track_mask[m].astype(int) for x,m in zip(cases,valid)])
            groups=np.concatenate([np.full(int(m.sum()),i,int) for i,m in enumerate(valid)])
            if set(np.unique(y))!={0,1}:raise ValueError('Synthetic external control must have both classes')
            # Consistent threshold / calibration-free ranking; AUROC/AP only.
            scored={};consistency={}
            for name,(arch,model) in models_loaded.items():
                probs=_batched_prob(model,imgs,ARM_BY_ARCH[arch]);scores=np.concatenate([p[m] for p,m in zip(probs,valid)])
                if not np.isfinite(scores).all():raise ValueError('Nonfinite model prediction')
                scored[name]=scores
                consistency[name]=float(np.mean([np.mean(np.abs(p-.5)) for p in probs]))
            met={name:{'AUROC_on_generated_track_vs_generated_scratch_pixels':float(roc_auc_score(y,p)),
                'AUPRC_generated_only':float(average_precision_score(y,p))} for name,p in scored.items()}
            comparisons={}
            for arch in ('azunet','unet'):
                p=scored[arch+'_photo_consistency'];q=scored[arch+'_matched_two_view']
                comparisons[arch]={'paired_AUROC_delta_consistency_minus_two_view':float(roc_auc_score(y,p)-roc_auc_score(y,q)),
                    'paired_AUPRC_delta':float(average_precision_score(y,p)-average_precision_score(y,q))}
            result['results'].append({'test_seed':seed,'train_seed':train_seed,'n_cases':n,
                'perturbation':perturb,'label_positive_generated_pixels':int(y.sum()),
                'label_negative_generated_scratch_pixels':int(len(y)-y.sum()),
                'metrics':met,'paired_deltas':comparisons,
                'source_provenance':provenance if perturb=='identity' else None})
            print('S35 EVAL',train_photo,seed,perturb,{k:round(v['AUROC_on_generated_track_vs_generated_scratch_pixels'],4) for k,v in met.items()},flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tiles',type=Path,required=True)
    p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--train-photo',choices=('005','006'),required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=evaluate_source_direction(a.tiles,a.measurements,a.models,train_photo=a.train_photo)
    a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
