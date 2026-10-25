#!/usr/bin/env python3
"""S34 untouched injection seeds, comparing original AZUNet local vs axial variant.

No real native H50Z track GT is introduced; evaluates injected lines/scratches.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s33_evaluate_photometric import load_model,photo_perturb
from scripts.research.bubble_chamber.s32_heldout_injection_control import independent_injections
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model,verify_model_source
from bubble_chamber.s33_photometric_domain import image_input

SEEDS={20261033:20261025,20261034:20261026,20261035:20261025,20261036:20261026}


def run(tiles,measurements,models,axisfolder,*,seed,n=48):
    if seed not in SEEDS:raise ValueError('Unregistered held-out S34 injection seed')
    verify_model_source();torch.set_num_threads(2)
    source_seed=SEEDS[seed]
    baseline=load_model(Path(models),'azunet','local',source_seed)
    unet=load_model(Path(models),'unet','local',source_seed)
    # Independent original weights, same initial random seed and sparse train budget.
    ck=torch.load(Path(axisfolder)/f'azunet_local_axis_seed{source_seed}.pt',weights_only=True,map_location='cpu')
    if ck['schema']!='s34_machine_only_AZUNet_axial_ablation_v1' or ck['seed']!=source_seed or ck['pixel_labels_are_human_certified'] is not False:
        raise ValueError('Invalid machine axial checkpoint')
    axial=build_model('azunet',widths=(4,6,8,12),rules=2);axial.load_state_dict(ck['model_state']);axial.eval()
    cases,source=independent_injections(tiles,measurements,seed=seed,n=n)
    methods={'azunet_local':baseline,'azunet_local_axis':axial,'unet_local':unet}
    study={'schema':'s34_unseen_synthetic_track_vs_scratch_test_v1','seed':seed,'model_train_seed':source_seed,
           'n_injected_samples':n,'native_real_photographic_ground_truth':False,
           'source_photo':'006','independent_physical_run':False,
           'new_injection_seeds_reserved_before_training':list(SEEDS),
           'source':source,'results':[],
           'physics_auto_joins_authorized':False}
    for transform in ('identity','bright','dark'):
        data={key:[] for key in methods};truth=[];group=[]
        with torch.no_grad():
            for i,c in enumerate(cases):
                y=c.synthetic_track_mask>0;neg=c.synthetic_scratch_mask>0;valid=y|neg
                if (y&neg).any():raise ValueError('invalid injected labels')
                gray=photo_perturb(c.gray,transform)
                input=torch.from_numpy(image_input(gray,'local')[None,None])
                truth.extend(y[valid].astype(int).tolist());group.extend([i]*int(valid.sum()))
                for name,model in methods.items():
                    score=torch.sigmoid(model(input))[0,0].numpy()
                    data[name].extend(score[valid].tolist())
        y=np.asarray(truth,int);group=np.asarray(group,int)
        scored={name:np.asarray(s,float) for name,s in data.items()}
        met={name:{'AUROC_injected_only':float(roc_auc_score(y,s)),
                   'AUPRC_injected_only':float(average_precision_score(y,s))}
             for name,s in scored.items()}
        rec={'transform':transform,'metrics':met,'positive_injected_px':int(y.sum()),
             'negative_scratch_px':int((y==0).sum())}
        if transform=='identity':
            rng=np.random.default_rng(seed)
            diffs=[]
            for _ in range(450):
                sample=rng.integers(0,n,n);w=np.bincount(sample,minlength=n)[group]
                keep=w>0
                if len(set(y[keep]))<2:continue
                diff=roc_auc_score(y[keep],scored['azunet_local_axis'][keep],sample_weight=w[keep])-roc_auc_score(y[keep],scored['azunet_local'][keep],sample_weight=w[keep])
                diffs.append(float(diff))
            rec['paired_delta_axis_minus_plain']={
                'AUROC':float(met['azunet_local_axis']['AUROC_injected_only']-met['azunet_local']['AUROC_injected_only']),
                'patch_bootstrap_CI95':[float(v) for v in np.quantile(diffs,[.025,.975])]}
        study['results'].append(rec)
    return study

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tiles',type=Path,required=True)
 p.add_argument('--measurements',type=Path,required=True)
 p.add_argument('--models',type=Path,required=True)
 p.add_argument('--axis-models',type=Path,required=True)
 p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 r=run(a.tiles,a.measurements,a.models,a.axis_models,seed=a.seed)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
 for x in r['results']:
  print('S34',a.seed,x['transform'],{k:round(v['AUROC_injected_only'],4) for k,v in x['metrics'].items()},flush=True)
