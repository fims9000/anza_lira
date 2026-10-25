#!/usr/bin/env python3
"""External synthetic-injection control for S31 pseudo-trained image models.

True labels are artificial inserted tracks and scratches; unmodified native
real photo pixels stay UNKNOWN and are excluded from every reported AUC.
"""
from __future__ import annotations
import argparse,json,hashlib,sys
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np,torch
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model
from scripts.research.bubble_chamber.s26_two_photo_baselines import render_one,classical_score
from bubble_chamber.large_archive import audited_large_manifest


def run(measurements,bmp,weights_dir,out,*,n=48,seed=20261029):
    assert n>=16
    torch.set_num_threads(2)
    manifest=audited_large_manifest(measurements)
    tracks=[t for f in manifest['frames'] for t in f['tracks'] if t['n_points']>=7]
    if len(tracks)<200:raise ValueError('Need audited measured track templates')
    im=cv2.imread(str(bmp),cv2.IMREAD_GRAYSCALE)
    if im is None or im.dtype!=np.uint8 or min(im.shape)<1000:raise ValueError('Expected full original H50Z BMP')
    models={}
    for method in ['unet','azunet']:
        ck_path=Path(weights_dir)/method/'best.pt';ck=torch.load(ck_path,weights_only=True,map_location='cpu')
        if ck.get('label_type')!='MACHINE_DRAFT_NOT_HUMAN_EXPERT' or ck.get('kind')!=method:
            raise ValueError('Refuse checkpoint claiming manual truth')
        model=build_model(method,widths=(4,6,8,12),rules=2)
        model.load_state_dict(ck['model_state']);model.eval();models[method]=model
    g=np.random.default_rng(seed)
    samples=[]
    h,w=im.shape
    for k in range(n):
        x=int(g.integers(int(w*.18),int(w*.82)));y=int(g.integers(int(h*.18),int(h*.82)))
        bg=im[y-32:y+32,x-32:x+32].copy()
        track=tracks[(k*131+seed*7)%len(tracks)]
        one=render_one(bg,track,seed,k+8192,'006','TEST_SYNTHETIC_INJECTION',(x,y))
        samples.append(one)
    collected={k:[] for k in ['unet','azunet','local_contrast','steerable_ridge']}
    target=[];image_ids=[]
    with torch.no_grad():
        for idx,e in enumerate(samples):
            fg=e.synthetic_track_mask>0;neg=e.synthetic_scratch_mask>0
            assert not np.any(fg&neg)
            known=fg|neg
            target.extend(fg[known].astype(int).tolist());image_ids.extend([idx]*int(known.sum()))
            x=torch.from_numpy((255-e.gray).astype(np.float32)[None,None]/255.)
            for method,model in models.items():
                prob=torch.sigmoid(model(x))[0,0].numpy()
                collected[method].extend(prob[known].tolist())
            for method in ['local_contrast','steerable_ridge']:
                p=classical_score(e.gray,method)
                collected[method].extend(p[known].tolist())
    target=np.asarray(target,dtype=int);image_ids=np.asarray(image_ids,dtype=int)
    result={'schema':'s31_holdout_injected_track_vs_scratch_ablation_v1',
       'image_source':'authentic H50Z_030_01_006.bmp native photographic background',
       'image_bmp_sha256':hashlib.sha256(Path(bmp).read_bytes()).hexdigest(),
       'track_templates_source':'audited measured VisualMetro film 070',
       'track_templates_source_sha256':manifest['provenance_sha256'],
       'injected_truth_is_synthetic':True,'real_native_photo_pixels_have_no_truth':True,
       'source_photo_test':'006_same_physical_run_as_training_photo_005',
       'n_injected_images':n,'sampled_positive_injected_pixels':int(target.sum()),
       'sampled_negative_scratch_pixels':int(len(target)-target.sum()),'trained_on_physical_track_masks':False,
       'rankings_tested':['unet','azunet','local_contrast','steerable_ridge'],
       'metrics':{},'original_ANZA_source_unchanged':True,'physics_auto_accept':False}
    for method,scores in collected.items():
        a=np.asarray(scores,float)
        if len(a)!=len(target) or not np.isfinite(a).all():raise AssertionError('invalid prediction')
        # AUC only on injected track vs injected scratch pixels, never native background.
        auroc=float(roc_auc_score(target,a));ap=float(average_precision_score(target,a))
        rng=np.random.default_rng(seed)
        boots=[]
        for _ in range(500):
            selected=rng.integers(0,n,n)
            weights=np.bincount(selected,minlength=n)
            inds=np.flatnonzero(weights[image_ids]>0)
            if len(np.unique(target[inds]))!=2:continue
            boots.append(float(roc_auc_score(target[inds],a[inds],sample_weight=weights[image_ids[inds]])))
        result['metrics'][method]={'auroc_synthetic_injected_only':auroc,'average_precision_synthetic_injected_only':ap,
               'auroc_frame_bootstrap_ci95':[float(v) for v in np.quantile(boots,[.025,.975])],
               'correct_3d_particle_identity':'NOT_AVAILABLE','real_native_track_dice':'NOT_AVAILABLE'}
        print('CONTROL',method,'AUROC',round(auroc,4),'AP',round(ap,4),flush=True)
    out=Path(out);out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--measurements',required=True,type=Path);p.add_argument('--bmp',required=True,type=Path)
 p.add_argument('--weights-dir',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--samples',type=int,default=48);p.add_argument('--seed',type=int,default=20261029)
 a=p.parse_args();run(a.measurements,a.bmp,a.weights_dir,a.out,n=a.samples,seed=a.seed)
