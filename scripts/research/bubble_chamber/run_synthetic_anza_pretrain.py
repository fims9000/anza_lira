#!/usr/bin/env python3
"""Same-split AZUNet vs BaselineUNet on simulated images of real clicked curves.

CAUTION: This does not evaluate real BMP segmentations or provide physical
results. The real expert geometry shapes these ARTIFICIALLY rendered images.
This script MUST be run from the complete anza_lira repository because it
imports its original models.azconv and models.segmentation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.large_relation_study import frame_partition
from bubble_chamber.synthetic_from_clicks import SyntheticFilmConfig, frame_render_dataset


def tensor_dataset(manifest, frame_ids, *, repeats:int, cfg:SyntheticFilmConfig):
    images=[];labels=[];sources=[]
    for image, mask, provenance in frame_render_dataset(manifest,frame_ids,cfg=cfg,repeats=repeats):
        images.append((255-image).astype(np.float32)[None]/255.)
        labels.append(mask.astype(np.float32)[None])
        sources.append(provenance)
    if not images:
        raise ValueError('no eligible tracks for synthetic rendering')
    return TensorDataset(torch.from_numpy(np.stack(images)),torch.from_numpy(np.stack(labels))),sources


def segmentation_loss(logits:torch.Tensor, targets:torch.Tensor):
    if logits.shape != targets.shape:
        raise ValueError('mismatched image/mask prediction')
    positive=targets.sum().clamp_min(1)
    negative=(1-targets).sum().clamp_min(1)
    pos_weight=(negative/positive).clamp(1,15)
    bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets,pos_weight=pos_weight)
    prob=torch.sigmoid(logits)
    soft_dice=1-(2*(prob*targets).sum()+1)/((prob+targets).sum()+1)
    return bce+soft_dice


@torch.no_grad()
def inference(model, loader, device):
    model.eval();probs=[];labels=[]
    for x,y in loader:
        prob=torch.sigmoid(model(x.to(device))).cpu().numpy()
        probs.append(prob);labels.append(y.numpy())
    return np.concatenate(probs).ravel(),np.concatenate(labels).ravel().astype(np.uint8)


def pixel_metrics(prob:np.ndarray,truth:np.ndarray,threshold:float):
    guess=prob>=threshold;gold=truth==1
    tp=int(np.logical_and(guess,gold).sum());fp=int(np.logical_and(guess,~gold).sum());fn=int(np.logical_and(~guess,gold).sum())
    return {'tp':tp,'fp':fp,'fn':fn,
            'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),
            'dice':2*tp/max(2*tp+fp+fn,1)}


def threshold_from_validation(prob:np.ndarray,truth:np.ndarray):
    candidates=np.arange(.1,.91,.05)
    return float(max(candidates,key=lambda t:(pixel_metrics(prob,truth,float(t))['dice'],-abs(t-.5))))


def train_one(model,model_name,train_loader,val_loader,test_loader,*,device,epochs:int,seed:int,out:Path):
    model=model.to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=0.0003,weight_decay=0.0001)
    best=float('inf');history=[];best_state=None;start=time.monotonic()
    for epoch in range(epochs):
        model.train();losses=[]
        for x,y in train_loader:
            optim.zero_grad(set_to_none=True)
            loss=segmentation_loss(model(x.to(device)),y.to(device))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f'{model_name} produced nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optim.step();losses.append(float(loss.detach()))
        model.eval();val_losses=[]
        with torch.no_grad():
            for x,y in val_loader:
                val_losses.append(float(segmentation_loss(model(x.to(device)),y.to(device))))
        dev=float(np.mean(val_losses));history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'val_loss':dev})
        if dev<best:
            best=dev;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model=model.to(device)
    pval,yval=inference(model,val_loader,device)
    thresh=threshold_from_validation(pval,yval)
    ptest,ytest=inference(model,test_loader,device)
    metrics=pixel_metrics(ptest,ytest,thresh)
    checkpoint=out/f'{model_name}_best_synthetic.pt'
    torch.save({'model':model_name,'weights':best_state,'synthetic_only':True,'seed':seed,'threshold_val':thresh},checkpoint)
    return {'name':model_name,'trained_epochs':epochs,'trainable_parameters':sum(x.numel() for x in model.parameters() if x.requires_grad),
            'threshold_selected_on_synthetic_validation':thresh,'synthetic_test':metrics,
            'elapsed_walltime_seconds':round(time.monotonic()-start,2),'history':history,
            'checkpoint_file':checkpoint.name,'physical_bmp_accuracy':'NOT_EVALUATED'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--size',type=int,default=128)
    p.add_argument('--seed',type=int,default=20261024)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1:
        p.error('epochs and batch size must be positive')
    from bubble_chamber.anza_segmentation_adapter import build_comparison_models, BubbleSegmentationConfig
    source=audited_large_manifest(args.measurements)
    splits=frame_partition([x['frame_id'] for x in source['frames']],args.seed)
    cfg=SyntheticFilmConfig(size=args.size,sample_seed=args.seed)
    train,_=tensor_dataset(source,splits['train'],repeats=3,cfg=cfg)
    val,_=tensor_dataset(source,splits['val'],repeats=2,cfg=cfg)
    test,_=tensor_dataset(source,splits['test'],repeats=2,cfg=cfg)
    if args.device=='cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    args.out_dir.mkdir(parents=True,exist_ok=True)
    results={}
    torch.manual_seed(args.seed)
    if device=='cuda': torch.cuda.manual_seed_all(args.seed)
    models=build_comparison_models(BubbleSegmentationConfig(layers=(8,16,24,32)))
    for name,model in models.items():
        torch.manual_seed(args.seed)
        if device=='cuda':torch.cuda.manual_seed_all(args.seed)
        # Identical row order and random sampler seed for both networks.
        train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
        val_loader=DataLoader(val,batch_size=args.batch_size,shuffle=False)
        test_loader=DataLoader(test,batch_size=args.batch_size,shuffle=False)
        results[name]=train_one(model,name,train_loader,val_loader,test_loader,device=device,epochs=args.epochs,seed=args.seed,out=args.out_dir)
        print(name,results[name]['synthetic_test'],flush=True)
    report={'experiment':'S5_TRUE_AZUNET_VS_CONVUNET_ON_SYNTHETIC_FILM_GEOMETRIES',
            'real_bmp_ground_truth_used':False,'dataset_label':'synthetically generated pixel masks from real sparse track clicks',
            'data_sha256':source['provenance_sha256'],'film_ids':source['summary']['distinct_film_ids'],
            'splits':{name:len(ids) for name,ids in splits.items()},
            'sizes':{'train':len(train),'validation':len(val),'test':len(test)},
            'training':{'seed':args.seed,'epochs':args.epochs,'batch_size':args.batch_size,'device':device},
            'results':results,
            'limitations':['No paired BMP images from film 070','No ANZA superiority on real bubble chamber can be claimed',
                           'Frame holdout only; same film/run and same synthetic generator','False crossings and real noise distributions are not replicated']}
    (args.out_dir/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':'SYNTHETIC_TRAINING_COMPLETE_NOT_REAL_VALIDATION','results':{k:v['synthetic_test'] for k,v in results.items()}},indent=2))

if __name__=='__main__': main()
