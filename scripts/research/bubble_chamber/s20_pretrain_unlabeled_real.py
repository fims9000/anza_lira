#!/usr/bin/env python3
"""S20: original AZUNet / BaselineUNet inpainting on REAL H50Z pixels, no labels.

Output checkpoints are feature-pretraining candidates only; all missing-pixel
masks are synthetic corruption positions, NOT actual particle tracks.
"""
from __future__ import annotations

import argparse, json, hashlib, random, time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s20_unlabeled import (discover_real_tiles, UnlabeledPhotoPatches,
    reconstruction_loss, masked_metrics, inpaint_baseline, edge_weights)

SOURCE_BLOBS={'models/azconv.py':'f4217ada00024927506fa8ebe520aa1f68ae5854',
              'models/segmentation.py':'0b971c117f3d56cc588ebade6991c65faceb8a2a'}


def verify_original_sources(root=ROOT):
    from hashlib import sha1
    for p, expected in SOURCE_BLOBS.items():
        f=Path(root)/p
        if not f.is_file():raise FileNotFoundError(f'Missing original ANZA source {p}')
        b=f.read_bytes()
        actual=sha1(b'blob '+str(len(b)).encode('ascii')+b'\0'+b).hexdigest()
        if actual!=expected: raise ValueError(f'Model source changed: {p}: {actual}')


@dataclass(frozen=True)
class Config:
    seed:int=20261025
    epochs:int=6
    batch:int=8
    patch:int=64
    train_per_tile:int=24
    val_per_tile:int=12
    fraction:float=.22
    block:int=6
    widths:tuple[int,int,int,int]=(4,6,8,12)
    rules:int=2
    lr:float=.0005
    threads:int=2
    brightness_jitter_px:float=0.
    contrast_jitter:float=0.
    edge_weight:float=0.


def build_model(name:str,cfg:Config):
    verify_original_sources()
    from models.segmentation import AZUNet,BaselineUNet
    from models.azconv import AZConvConfig
    if name=='azunet':
        return AZUNet(in_channels=1,out_channels=1,num_rules=cfg.rules,widths=cfg.widths,
                      cfg=AZConvConfig(geometry_mode='local_hyperbolic',use_fuzzy=True,use_anisotropy=True))
    if name=='unet':return BaselineUNet(in_channels=1,out_channels=1,widths=cfg.widths)
    raise ValueError('Unknown model')


def dataset_receipt(data:UnlabeledPhotoPatches):
    h=hashlib.sha256()
    for a in (data.x,data.y,data.hole): h.update(a.numpy().tobytes())
    return {'n_patches':len(data),'split_sha256':h.hexdigest(),
            'source_photo_ids':sorted(set(x['photo_id'] for x in data.provenance)),
            'artificial_mask_pixels':int(data.hole.sum()),'expert_labels':0}


def eval_reconstruction(model,data,cfg):
    loader=DataLoader(data,batch_size=cfg.batch,shuffle=False)
    ps,ys,hs=[],[],[];model.eval()
    with torch.no_grad():
        for x,y,h in loader:
            ps.append(torch.sigmoid(model(x)).cpu().numpy())
            ys.append(y.numpy());hs.append(h.numpy())
    return np.concatenate(ps),np.concatenate(ys),np.concatenate(hs)


def save_atomic(path,payload):
    tmp=path.with_suffix('.tmp')
    torch.save(payload,tmp);tmp.replace(path)


def train_one(name:str,cfg:Config,train,val,out:Path,split_receipt:dict):
    torch.manual_seed(cfg.seed);np.random.seed(cfg.seed);random.seed(cfg.seed)
    model=build_model(name,cfg)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=.0001)
    out.mkdir(parents=True,exist_ok=True)
    best=float('inf');history=[];t0=time.monotonic()
    for epoch in range(cfg.epochs):
        model.train();losses=[]
        g=torch.Generator().manual_seed(cfg.seed+epoch)
        for x,y,holes in DataLoader(train,batch_size=cfg.batch,shuffle=True,generator=g):
            opt.zero_grad(set_to_none=True)
            pred=model(x)
            if cfg.edge_weight>0:
                # The hidden true pixel may enter the LOSS, not the model input.
                weights=(1.+cfg.edge_weight*edge_weights(y))[holes]
                err=torch.nn.functional.smooth_l1_loss(torch.sigmoid(pred)[holes], y[holes], beta=.1,reduction='none')
                loss=(weights*err).sum()/weights.sum()
            else:
                loss=reconstruction_loss(pred,y,holes)
            if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite self-supervised loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            losses.append(float(loss.detach()))
        predicted,actual,mask=eval_reconstruction(model,val,cfg)
        metric=masked_metrics(predicted,actual,mask)
        history.append({'epoch':epoch+1,'mean_train_huber':float(np.mean(losses)),
                        'validation_masked_mae':metric['masked_mae']})
        checkpoint={'schema':'s20_real_unlabeled_masked_reconstruction_v1','architecture':name,
                    'weights':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    'widths':cfg.widths,'rules':cfg.rules,'source_blobs':SOURCE_BLOBS,
                    'training_config':asdict(cfg), 'source_split':split_receipt,
                    'validation_proxy_masked_mae':metric['masked_mae'],
                    'checkpoint_epoch':epoch+1,
                    'training_is_unlabeled':True,'human_verified_track_labels':False,
                    'model_is_not_calibrated_for_real_track_segmentation':True}
        save_atomic(out/'last.pt',checkpoint)
        if metric['masked_mae']<best:
            best=metric['masked_mae'];save_atomic(out/'best.pt',checkpoint)
    checkpoint=torch.load(out/'best.pt',weights_only=True,map_location='cpu')
    model.load_state_dict(checkpoint['weights']);ps,ys,hs=eval_reconstruction(model,val,cfg)
    best_metrics=masked_metrics(ps,ys,hs)
    report={'architecture':name,'parameters':sum(p.numel() for p in model.parameters()),
            'best_epoch':checkpoint['checkpoint_epoch'],'history':history,
            'validation_reconstruction':best_metrics,'elapsed_cpu_seconds':round(time.monotonic()-t0,2),
            'checkpoint':'best.pt','checkpoint_sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),
            'segmentation_accuracy':'NOT_MEASURED', 'photo_count_test':0}
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def run(tiles_dir:Path,out:Path,cfg:Config,models=('unet','azunet')):
    if cfg.epochs<1 or cfg.batch<1 or cfg.threads<1 or cfg.rules<1 or any(w<2 for w in cfg.widths) or not 0<=cfg.edge_weight<=10:
        raise ValueError('Bad training configuration')
    verify_original_sources()
    torch.set_num_threads(cfg.threads)
    tiles,groups=discover_real_tiles(tiles_dir)
    train=[t for t in tiles if t.photo_id==groups['train_photo_id']]
    val=[t for t in tiles if t.photo_id==groups['validation_photo_id']]
    trainset=UnlabeledPhotoPatches(train,patch=cfg.patch,per_tile=cfg.train_per_tile,
                                  seed=cfg.seed,fraction=cfg.fraction,block=cfg.block,
                                  brightness_jitter_px=cfg.brightness_jitter_px,
                                  contrast_jitter=cfg.contrast_jitter)
    valset=UnlabeledPhotoPatches(val,patch=cfg.patch,per_tile=cfg.val_per_tile,
                                seed=cfg.seed+112,fraction=cfg.fraction,block=cfg.block)
    assert not set(x['photo_id'] for x in trainset.provenance)&set(x['photo_id'] for x in valset.provenance)
    split_receipt={'original_photo_groups':groups,'train':dataset_receipt(trainset),
                   'validation':dataset_receipt(valset)}
    baseline_preds=[]
    for x,y,holes in valset:
        baseline_preds.append(inpaint_baseline(x[0].numpy(),holes[0].numpy())[None])
    simple_baseline=masked_metrics(np.stack(baseline_preds),valset.y.numpy(),valset.hole.numpy())
    report={'schema':'s20_real_unlabeled_photo_reconstruction_v1',
            'sources':split_receipt,'config':asdict(cfg), 'methods':{},
            'telea_inpainting_proxy':simple_baseline,
            'real_source_pixels_used':True, 'real_track_labels_used':False,
            'track_segmentation_quality':'NOT_MEASURED',
            'independent_physical_run_test':'NOT_AVAILABLE',
            'note':'Model selected by masked reconstruction on source photo 006; that same photo is validation, NOT independent test',
            'original_model_source_blobs':SOURCE_BLOBS}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name in models:
        report['methods'][name]=train_one(name,cfg,trainset,valset,out/name,split_receipt)
        (out/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tiles-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--seed',type=int,default=20261025)
    p.add_argument('--patch',type=int,default=64)
    p.add_argument('--train-per-tile',type=int,default=24)
    p.add_argument('--val-per-tile',type=int,default=12)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--widths',type=int,nargs=4,default=[4,6,8,12])
    p.add_argument('--rules',type=int,default=2)
    p.add_argument('--models',nargs='+',choices=['azunet','unet'],default=['unet','azunet'])
    p.add_argument('--brightness-jitter-px',type=float,default=0.)
    p.add_argument('--contrast-jitter',type=float,default=0.)
    p.add_argument('--edge-weight',type=float,default=0.)
    a=p.parse_args()
    cfg=Config(seed=a.seed,epochs=a.epochs,patch=a.patch,train_per_tile=a.train_per_tile,
               val_per_tile=a.val_per_tile,batch=a.batch,widths=tuple(a.widths),rules=a.rules,
               brightness_jitter_px=a.brightness_jitter_px,contrast_jitter=a.contrast_jitter,
               edge_weight=a.edge_weight)
    result=run(a.tiles_dir,a.out_dir,cfg,tuple(a.models))
    print(json.dumps({'photo_grouped_val':result['sources']['validation']['source_photo_ids'],
                      'unlabeled_pretext':True,
                      'methods':{name:m['validation_reconstruction'] for name,m in result['methods'].items()},
                      'classical_telea':result['telea_inpainting_proxy'],
                      'real_track_segmentation':'NOT_MEASURED'},indent=2))
if __name__=='__main__':main()
