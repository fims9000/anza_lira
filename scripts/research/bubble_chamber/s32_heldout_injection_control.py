#!/usr/bin/env python3
"""S32 injection-only negative control on REAL 006 crop backgrounds.

Synthetic strokes generated from distinct visualMetro measured track shapes;
unsupervised native source photo pixels NEVER become truth.
"""
from __future__ import annotations
import argparse,json,hashlib,sys
from pathlib import Path
import cv2,numpy as np,torch
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model,verify_model_source
from bubble_chamber.large_archive import audited_large_manifest
from scripts.research.bubble_chamber.s26_two_photo_baselines import render_one,classical_score


def load_ck(path,architecture):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('kind')!=architecture or ck.get('label_type')!='MACHINE_DRAFT_NOT_HUMAN_EXPERT':
        raise ValueError('Wrong architecture or falsely certified checkpoint')
    if ck.get('physical_auto_joins_authorized',False):raise ValueError('Unsafe checkpoint')
    model=build_model(architecture,widths=(4,6,8,12),rules=2)
    model.load_state_dict(ck['model_state']);model.eval();return model


def independent_injections(tiles,measurements,seed=20261029,n=48):
    if n<16:raise ValueError('At least 16 synthetic controlled examples')
    originals=sorted(Path(tiles).glob('H50Z_030_01_006_*.png'))
    if len(originals)!=18:raise ValueError('Expected all 18 original 006 crops')
    manifest=audited_large_manifest(measurements)
    tracks=[t for f in manifest['frames'] for t in f['tracks'] if t['n_points']>=7]
    if len(tracks)<200:raise ValueError('Insufficient measured track templates')
    rng=np.random.default_rng(seed);records=[]
    for i in range(n):
        path=originals[i%len(originals)];src=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if src is None or src.shape!=(512,512):raise ValueError('Bad authentic H50Z crop')
        xx=int(rng.integers(35,512-35));yy=int(rng.integers(35,512-35))
        patch=src[yy-32:yy+32,xx-32:xx+32].copy()
        template=tracks[(i*131+seed*7)%len(tracks)]
        record=render_one(patch,template,seed,i+8192,'006',path.stem,(xx,yy))
        records.append(record)
    return records,{'source_photo':'006','source_tile_count':len(originals),
        'measurement_source_sha256':manifest['provenance_sha256'],
        'source_image_file_sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in originals],
        'actual_real_photo_track_truth':False,'only_synthetic_overlay_truth':True,
        'independent_physical_run':False}


def metric(y,score,patch_ids,seed):
    y=np.asarray(y,int);score=np.asarray(score,float);patch_ids=np.asarray(patch_ids,int)
    if len(y)!=len(score) or len(y)!=len(patch_ids) or set(np.unique(y))!={0,1}:raise ValueError('Bad injected test')
    auroc=float(roc_auc_score(y,score));ap=float(average_precision_score(y,score))
    rng=np.random.default_rng(seed);n=len(np.unique(patch_ids));reps=[]
    for _ in range(200):
        ind=rng.integers(0,n,n);freq=np.bincount(ind,minlength=n);w=freq[patch_ids]
        idx=w>0
        if len(np.unique(y[idx]))<2:continue
        reps.append(float(roc_auc_score(y[idx],score[idx],sample_weight=w[idx])))
    return {'injected_pixel_auroc':auroc,'injected_pixel_average_precision':ap,
            'paired_by_patch_bootstrap_AUROC_95ci':[float(x) for x in np.quantile(reps,[.025,.975])],
            'physical_track_segmentation_dice':'NOT_MEASURED'}


def run(tiles,measurements,model_dir,out,*,seeds=(20261029,20261030),n=48, historical_unet=None,historical_azunet=None):
    verify_model_source();torch.set_num_threads(2)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    result={'schema':'s32_external_injected_track_vs_scratch_v1','source_photo':'006',
            'historical_S31_training_budget_not_matched':bool(historical_unet is not None),
            'human_labeled_real_track_pixels':0,'real_photo_native_track_accuracy':'NOT_MEASURED',
            'physics_auto_join_allowed':False,'matched_training_arms':True,'results':[]}
    for seed in seeds:
        items,provenance=independent_injections(tiles,measurements,seed,n)
        train_seed={20261029:20261025,20261030:20261026}[seed]
        net={}
        for arch in ('unet','azunet'):
            for mode in ('s31_unweighted_control','s32_confidence'):
                name=f'{arch}_{mode}'
                ck=Path(model_dir)/f'{name}_seed{train_seed}'/'best.pt'
                net[name]=load_ck(ck,arch)
        if historical_unet is not None and historical_azunet is not None:
            net['historical_S31_unet']=load_ck(historical_unet,'unet')
            net['historical_S31_azunet']=load_ck(historical_azunet,'azunet')
        targets=[];groups=[];scores={name:[] for name in list(net)+['local_contrast','steerable_ridge']}
        with torch.no_grad():
            for j,item in enumerate(items):
                fg=item.synthetic_track_mask>0;neg=item.synthetic_scratch_mask>0
                if np.any(fg&neg):raise AssertionError('Overlapping synthetic injected classes')
                known=fg|neg
                targets.extend(fg[known].astype(int).tolist())
                groups.extend([j]*int(known.sum()))
                x=torch.from_numpy((255-item.gray).astype('float32')[None,None]/255.)
                for name,m in net.items():
                    pred=torch.sigmoid(m(x))[0,0].numpy()
                    scores[name].extend(pred[known].tolist())
                for name in ('local_contrast','steerable_ridge'):
                    pred=classical_score(item.gray,name)
                    scores[name].extend(pred[known].tolist())
        record={'injection_seed':seed,'training_seed':train_seed,'images':len(items),
                'positive_injected_pixels':int(sum(targets)),
                'negative_injected_scratch_pixels':int(len(targets)-sum(targets)),
                'provenance':provenance,'methods':{name:metric(targets,a,groups,seed) for name,a in scores.items()}}
        result['results'].append(record)
        out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('S32 CONTROL',seed,{name:round(v['injected_pixel_auroc'],4) for name,v in record['methods'].items()},flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tiles',required=True,type=Path)
    p.add_argument('--measurements',required=True,type=Path)
    p.add_argument('--models',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[20261029,20261030]);p.add_argument('--n',type=int,default=48)
    p.add_argument('--historical-unet',type=Path);p.add_argument('--historical-azunet',type=Path)
    a=p.parse_args();run(a.tiles,a.measurements,a.models,a.out,seeds=tuple(a.seeds),n=a.n,
       historical_unet=a.historical_unet,historical_azunet=a.historical_azunet)
