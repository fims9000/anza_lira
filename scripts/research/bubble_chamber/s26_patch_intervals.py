#!/usr/bin/env python3
"""Paired patch-cluster bootstrap of 006 synthetic mark AUC differences.

All tests reuse THE SAME held-out photo per seed. Intervals quantify patch
variation conditional on the single source photo / renderer; they DO NOT
bound across independent films or runs.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s26_two_photo_baselines import sample_two_photos,pixel_input
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model


def model_predictions(ck_path,entries):
    ck=torch.load(ck_path,weights_only=True,map_location='cpu')
    name=ck['architecture'];mode=ck['mode']
    model=build_model(name,widths=(4,6,8,12),rules=2);model.load_state_dict(ck['state']);model.eval()
    x=torch.from_numpy(np.stack([pixel_input(e.gray,mode) for e in entries])[:,None])
    out=[]
    with torch.no_grad():
        for part in x.split(8):out.extend(torch.sigmoid(model(part))[:,0].numpy())
    return np.asarray(out)


def experiment(data,out,seed,repeats=3000):
    test=data['test'];y=[e.sparse[e.sparse!=255].astype(int) for e in test]
    cols={}
    for arch in ('unet','azunet'):
        for mode in ('raw','local'):
            ck=out/f'{arch}_{mode}/best.pt'
            scores=model_predictions(ck,test)
            cols[f'{arch}_{mode}']=[p[e.sparse!=255] for p,e in zip(scores,test)]
    auc={k:np.asarray([roc_auc_score(y0,pr) for y0,pr in zip(y,probs)],float) for k,probs in cols.items()}
    rng=np.random.default_rng(seed+40)
    ix=rng.integers(0,len(test),size=(repeats,len(test)))
    contrasts={}
    for left,right in [('azunet_local','azunet_raw'),('unet_local','unet_raw'),('azunet_local','unet_local')]:
        delta=auc[left]-auc[right]
        sampled=delta[ix].mean(axis=1)
        contrasts[f'{left}_minus_{right}']={'paired_patch_auc_delta':float(delta.mean()),
            'bootstrap_95pct_ci_conditional_on_one_photo':[float(x) for x in np.quantile(sampled,[.025,.975])],
            'positive_patch_fraction':float(np.mean(delta>0))}
    result={'seed':seed,'photo':'006','patch_count':len(test),'per_method_macro_patch_auc':{k:float(v.mean()) for k,v in auc.items()},
            'contrasts':contrasts,'bootstrap_repeats':repeats,
            'true_native_track_labels':0,'film_independent_inference':False,
            'note':'Patches from a single source photograph share noise/acquisition; bootstrap CIs are conditional synthetic-label diagnostics only'}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bmps',type=Path,required=True);p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2)
    res=[]
    for seed in (20261027,20261028):
        data,_=sample_two_photos(args.bmps,args.measurements,seed,train_n=144,val_n=48,test_n=96)
        res.append(experiment(data,args.runs/f'seed{seed}_full',seed))
    args.out.write_text(json.dumps({'schema':'s26_conditional_patch_intervals', 'seeds':res},ensure_ascii=False,indent=2)+'\n')
    for r in res:print(r['seed'],r['per_method_macro_patch_auc'],r['contrasts'])
