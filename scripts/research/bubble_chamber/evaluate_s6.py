#!/usr/bin/env python3
"""S6 reproducible synthetic segmentation comparison, with frozen val thresholds.

Requires the exact original AZUNet/BaselineUNet in the full anza_lira repo.
The only real measurements are sparse 070 track clicks; all image pixels are generated.
"""
from __future__ import annotations
import argparse,json,sys,hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.large_relation_study import frame_partition
from bubble_chamber.synthetic_from_clicks import SyntheticFilmConfig,frame_render_dataset
from bubble_chamber.anza_segmentation_adapter import BubbleSegmentationConfig,build_comparison_models
from bubble_chamber.ridges import frangi_dark_ridges


def aggregate(pred,truth):
    pred=np.asarray(pred,bool); truth=np.asarray(truth,bool)
    if pred.shape!=truth.shape:raise ValueError('prediction/gold shape mismatch')
    return np.array([int((pred&truth).sum()),int((pred&~truth).sum()),int((~pred&truth).sum())],np.int64)


def metrics(c):
    tp,fp,fn=(int(x) for x in c)
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(tp+fp,1),
            'recall':tp/max(tp+fn,1),'dice':2*tp/max(2*tp+fp+fn,1)}


def clustered_counts(pred,gold,ids):
    if not (len(pred)==len(gold)==len(ids)):raise ValueError('sample mismatch')
    counts=defaultdict(lambda:np.zeros(3,np.int64))
    for p,g,f in zip(pred,gold,ids):counts[f]+=aggregate(p,g)
    return dict(counts)


def paired_bootstrap(base,alt,seed=20261024,repeats=2500):
    frames=sorted(base)
    if frames!=sorted(alt) or len(frames)<5:raise ValueError('frame identity mismatch')
    a=np.array([base[f] for f in frames]);b=np.array([alt[f] for f in frames])
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(frames),(repeats,len(frames)))
    x=a[idx].sum(1);y=b[idx].sum(1)
    da=2*x[:,0]/np.maximum(2*x[:,0]+x[:,1]+x[:,2],1)
    db=2*y[:,0]/np.maximum(2*y[:,0]+y[:,1]+y[:,2],1)
    return {'delta_dice':metrics(b.sum(0))['dice']-metrics(a.sum(0))['dice'],
            'ci95':np.quantile(db-da,[.025,.975]).tolist(),
            'frames':len(frames),'bootstrap_repetitions':repeats}


def line_f1(pred,truth,radius=2):
    yy,xx=np.ogrid[-radius:radius+1,-radius:radius+1]
    disk=(xx**2+yy**2)<=radius**2
    hits_pred=hits_gt=npix=ngt=0
    for p,t in zip(pred,truth):
        pp=skeletonize(np.asarray(p,bool));tt=skeletonize(np.asarray(t,bool))
        hits_pred+=int((pp&binary_dilation(tt,structure=disk)).sum())
        hits_gt+=int((tt&binary_dilation(pp,structure=disk)).sum())
        npix+=int(pp.sum());ngt+=int(tt.sum())
    prec=hits_pred/max(npix,1);rec=hits_gt/max(ngt,1)
    return {'precision':prec,'recall':rec,'f1':2*prec*rec/max(prec+rec,1e-12),'radius_px':radius}


def draw(manifest,frames,seed,size,repeats):
    source=set(frames); ims=[];truth=[];ids=[]
    for image,mask,meta in frame_render_dataset(manifest,sorted(source),SyntheticFilmConfig(size=size,sample_seed=seed),repeats=repeats):
        ims.append(image);truth.append(mask.astype(bool));ids.append(meta['source_frame'])
    if not ims:raise ValueError('no rendered data')
    return np.stack(ims),np.stack(truth),ids


def predict(model,images,threshold,batch_size=8):
    predictions=[];model.eval()
    with torch.no_grad():
        for i in range(0,len(images),batch_size):
            x=torch.from_numpy(((255-images[i:i+batch_size]).astype('float32')/255)[:,None])
            predictions.append(torch.sigmoid(model(x)).squeeze(1).numpy()>=threshold)
    return np.concatenate(predictions)


def stress_images(images,seed=51,scale=.43,noise=12):
    rng=np.random.default_rng(seed);out=np.empty_like(images)
    for i,image in enumerate(images):
        x=image.astype(np.float32);median=float(np.median(x))
        out[i]=np.clip(median+(x-median)*scale+rng.normal(0,noise,x.shape),0,255).astype('uint8')
    return out


def main():
    a=argparse.ArgumentParser();a.add_argument('--measurements',type=Path,required=True)
    a.add_argument('--training-dir',type=Path,required=True)
    a.add_argument('--out-dir',type=Path,required=True)
    a.add_argument('--test-views',type=int,choices=(2,3),default=2)
    a.add_argument('--no-frangi',action='store_true')
    args=a.parse_args()
    saved=json.loads((args.training_dir/'metrics.json').read_text())
    manifest=audited_large_manifest(args.measurements)
    if manifest['provenance_sha256']!=saved['data_sha256']:raise RuntimeError('input changed')
    seed=int(saved['training']['seed']);size=64
    split=frame_partition([x['frame_id'] for x in manifest['frames']],seed)
    if any(saved['splits'][k]!=len(v) for k,v in split.items()):raise RuntimeError('different split')
    x,y,ids=draw(manifest,split['test'],seed,size,args.test_views)
    valx,valy,val_ids=draw(manifest,split['val'],seed,size,2)
    assert not (set(ids)&set(val_ids))
    models=build_comparison_models(BubbleSegmentationConfig(layers=(8,16,24,32)))
    row={'schema':'S6_SYNTHETIC_NEURAL_COMPARISON_V1','test_views':args.test_views,
         'test_samples':len(x),'test_frames':len(set(ids)),'source_sha256':manifest['provenance_sha256'],
         'test_label_source':'generated masks from real sparse click geometry; NOT real film masks',
         'model_source':'use original models/azconv.py in full repo',
         'training_sizes':saved['sizes'],'results':{},'paired':{},
         'limitations':['No matched film 070 BMP','No independent film/run holdout','One training seed','No real-photo classification or true false-bridge rate']}
    by={};stress_by={};stressed=stress_images(x)
    for name,model in models.items():
        ck=saved['results'][name]
        w=torch.load(args.training_dir/ck['checkpoint_file'],map_location='cpu',weights_only=True)
        model.load_state_dict(w['weights'])
        t=float(ck['threshold_selected_on_synthetic_validation'])
        clean=predict(model,x,t)
        altered=predict(model,stressed,t)
        by[name]=clustered_counts(clean,y,ids)
        stress_by[name]=clustered_counts(altered,y,ids)
        row['results'][name]={'threshold_val':t,'test':metrics(np.stack(list(by[name].values())).sum(0)),
                              'centerline_f1':line_f1(clean,y),'stress_test':metrics(np.stack(list(stress_by[name].values())).sum(0)),
                              'stress_centerline_f1':line_f1(altered,y)}
    if not args.no_frangi:
        trials=[]
        for fraction in (.04,.08,.12,.18,.25):
            masks=[frangi_dark_ridges(v,response_fraction=fraction,scales=(1.,2.,3.),min_area=3)[0] for v in valx]
            c=clustered_counts(masks,valy,val_ids)
            trials.append((metrics(np.stack(list(c.values())).sum(0))['dice'],fraction))
        threshold=max(trials,key=lambda pair:(pair[0],-abs(pair[1]-.12)))[1]
        masks=[frangi_dark_ridges(v,response_fraction=threshold,scales=(1.,2.,3.),min_area=3)[0] for v in x]
        masks_stress=[frangi_dark_ridges(v,response_fraction=threshold,scales=(1.,2.,3.),min_area=3)[0] for v in stressed]
        by['frangi']=clustered_counts(masks,y,ids)
        stress_by['frangi']=clustered_counts(masks_stress,y,ids)
        row['results']['frangi']={'threshold_val_fraction':threshold,
                                 'test':metrics(np.stack(list(by['frangi'].values())).sum(0)),
                                 'centerline_f1':line_f1(masks,y),
                                 'stress_test':metrics(np.stack(list(stress_by['frangi'].values())).sum(0)),
                                 'stress_centerline_f1':line_f1(masks_stress,y)}
    for name in by:
        if name=='plain_unet':continue
        row['paired'][name+'_minus_plain']=paired_bootstrap(by['plain_unet'],by[name],seed+17)
        row['paired'][name+'_minus_plain_stress']=paired_bootstrap(stress_by['plain_unet'],stress_by[name],seed+171)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'metrics.json').write_text(json.dumps(row,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'samples':len(x),'models':{k:v['test'] for k,v in row['results'].items()}},indent=2))

if __name__=='__main__':main()
