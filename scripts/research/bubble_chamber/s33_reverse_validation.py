#!/usr/bin/env python3
"""S33 opposite cross-photo direction: train 006, synthetic control test 005.

Original images remain UNKNOWN native truth; injected tracks vs scratches only.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model,verify_model_source
from scripts.research.bubble_chamber.s26_two_photo_baselines import render_one,classical_score
from scripts.research.bubble_chamber.s33_evaluate_photometric import photo_perturb
from bubble_chamber.s33_photometric_domain import image_input
from bubble_chamber.large_archive import audited_large_manifest

TRAIN_SEED={20261037:20261025,20261038:20261026,20261039:20261025,20261040:20261026}


def generate_holdout_005(tiles,measurements,seed,n):
    original=sorted(Path(tiles).glob('H50Z_030_01_005_*.png'))
    if len(original)!=21:raise ValueError('Need all 21 source 005 review crops')
    data=audited_large_manifest(Path(measurements))
    tracks=[t for f in data['frames'] for t in f['tracks'] if t['n_points']>=7]
    if len(tracks)<200:raise ValueError('too few original measured shapes')
    rng=np.random.default_rng(seed);cases=[]
    for i in range(n):
        file=original[i%len(original)];src=cv2.imread(str(file),cv2.IMREAD_GRAYSCALE)
        x=int(rng.integers(35,512-35));y=int(rng.integers(35,512-35))
        bg=src[y-32:y+32,x-32:x+32].copy()
        shape=tracks[(i*131+seed*7)%len(tracks)]
        cases.append(render_one(bg,shape,seed,i+8192,'005',file.stem,(x,y)))
    return cases,{'test_photo':'005','crop_count':len(original),'expert_photo_track_truth':False,
                  'measured_shape_source_sha256':data['provenance_sha256']}


def run(tiles,measurements,folder,seed,n=48):
    if seed not in TRAIN_SEED:raise ValueError('Unregistered reverse injection seed')
    verify_model_source();torch.set_num_threads(2)
    source_seed=TRAIN_SEED[seed]
    cases,prov=generate_holdout_005(tiles,measurements,seed,n)
    models={}
    for arch in ('azunet','unet'):
        for mode in ('raw','quantile','local'):
            ck=torch.load(Path(folder)/f'{arch}_{mode}_seed{source_seed}.pt',weights_only=True,map_location='cpu')
            if ck.get('photo_train')!='006' or ck.get('test_photo_NOT_used_for_training')!='005' or ck.get('mode')!=mode or ck.get('seed')!=source_seed:
                raise ValueError('Source test-photo leakage or invalid checkpoint')
            model=build_model(arch,widths=(4,6,8,12),rules=2)
            model.load_state_dict(ck['model_state']);model.eval()
            models[f'{arch}_{mode}']=(mode,model)
    reports=[]
    with torch.no_grad():
        for perturb in ('identity','bright','dark'):
            scores={name:[] for name in models};truth=[];groups=[]
            for i,e in enumerate(cases):
                mask=e.synthetic_track_mask>0;scratch=e.synthetic_scratch_mask>0
                assert not (mask&scratch).any()
                valid=mask|scratch
                truth.extend(mask[valid].astype(int).tolist());groups.extend([i]*int(valid.sum()))
                gray=photo_perturb(e.gray,perturb)
                x={mode:torch.from_numpy(image_input(gray,mode)[None,None]) for mode in ('raw','quantile','local')}
                for name,(mode,model) in models.items():
                    p=torch.sigmoid(model(x[mode]))[0,0].numpy()
                    scores[name].extend(p[valid].tolist())
            y=np.asarray(truth,int);groups=np.asarray(groups,int)
            if set(np.unique(y))!={0,1}:raise ValueError('No synthetic classes')
            metrics={name:{'AUROC_synthetic_only':float(roc_auc_score(y,s)),
                           'AUPRC_synthetic_only':float(average_precision_score(y,s))}
                    for name,s in scores.items()}
            reports.append({'perturbation':perturb,'n_synthetic_samples':n,
                            'positive_synthetic_pixels':int(y.sum()),
                            'negative_synthetic_scratch_pixels':int((1-y).sum()),
                            'metrics':metrics})
    return {'schema':'s33_reverse_006_train_005_test_synthetic_only_v1',
            'test_seed':seed,'train_seed':source_seed,
            'source_photo_train':'006','source_photo_test':'005',
            'native_H50Z_track_ground_truth':False,'physics_auto_join_allowed':False,
            'independent_physical_run':False,
            'test_photo_not_used_for_epoch_selection':True,
            'data_provenance':prov,'evaluations':reports}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tiles',type=Path,required=True)
 p.add_argument('--measurements',type=Path,required=True)
 p.add_argument('--models',type=Path,required=True)
 p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();r=run(a.tiles,a.measurements,a.models,a.seed)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n')
 for e in r['evaluations']:
  print('REVERSE',a.seed,e['perturbation'],{k:round(v['AUROC_synthetic_only'],4) for k,v in e['metrics'].items()},flush=True)
