#!/usr/bin/env python3
"""S7: exact-source AZConv2d vs ordinary Conv2d on generated track pixels.

Only SOURCE track geometry is human-measured for --measurements. Pixel images,
widths, noise, occlusions and target masks are SIMULATED. A success here is
never real bubble chamber performance. Does not modify original models/.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Exact git blob IDs recorded against user branch cca89a3. These are source
# fingerprints, NOT trained-weight hashes or accuracy certificates.
SOURCE_BLOBS = {
    'models/azconv.py': 'f4217ada00024927506fa8ebe520aa1f68ae5854',
    'models/segmentation.py': '0b971c117f3d56cc588ebade6991c65faceb8a2a',
}


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()


def verify_sources(root: Path = REPO_ROOT) -> dict[str, str]:
    actual = {}
    for relative, expected in SOURCE_BLOBS.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f'Original source missing: {path}')
        actual[relative] = blob_sha(path.read_bytes())
        if actual[relative] != expected:
            raise ValueError(f'Original model source mismatch: {relative}; expected={expected}, got={actual[relative]}')
    return actual


@dataclass(frozen=True)
class RunConfig:
    source: str = 'procedural'  # or visualmetro
    seed: int = 20261024
    data_seed: int | None = None  # source data fixed independently of model initialization
    size: int = 64
    epochs: int = 2
    batch_size: int = 4
    widths: tuple[int, int, int, int] = (8, 16, 24, 32)
    num_rules: int = 2
    cpu_threads: int = 2
    train_limit: int = 48
    val_limit: int = 16
    test_limit: int = 24
    steps_per_epoch: int = 0  # 0 means all
    learning_rate: float = 0.0003
    stress_contrast: float = .25
    stress_noise_std: float = .08
    az_residual: bool = False
    robust_train: bool = False
    robust_probability: float = 0.8
    robust_scale_min: float = 0.15
    robust_scale_max: float = 1.0
    robust_noise_max: float = 0.08


def effective_data_seed(cfg: RunConfig) -> int:
    return cfg.seed if cfg.data_seed is None else int(cfg.data_seed)


class TrackDataset(Dataset):
    def __init__(self, examples: list[tuple[np.ndarray, np.ndarray, str]]):
        if not examples:
            raise ValueError('Empty dataset')
        self.samples = examples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        x, y, _ = self.samples[index]
        return torch.from_numpy(x.astype(np.float32)[None]), torch.from_numpy(y.astype(np.float32)[None]), index
    def groups(self):
        return [g for _, _, g in self.samples]


def procedural_examples(n: int, size: int, *, seed: int, split: str):
    """Procedural *software smoke* control. NOT based on physical photographs."""
    if n < 1 or size < 32 or size % 16:
        raise ValueError('invalid synthetic dataset parameters')
    result=[]
    for j in range(n):
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'{seed}:{split}:{j}'.encode()).digest()[:8], 'little'))
        image=np.full((size,size),rng.uniform(185,235),dtype=np.float32)
        image+=rng.normal(0,rng.uniform(2,8),(size,size))
        yy,xx=np.ogrid[:size,:size]
        image+=rng.uniform(-18,18)*xx/size+rng.uniform(-18,18)*yy/size
        mask=np.zeros((size,size),np.uint8)
        line_count=int(rng.integers(1,4))
        for line in range(line_count):
            xs=np.linspace(3,size-4,20)
            offset=float(rng.uniform(size*.20,size*.78))
            slope=float(rng.uniform(-.6,.6))
            curve=float(rng.uniform(-.006,.006))
            ys=offset+slope*(xs-size/2)+curve*(xs-size/2)**2
            coordinates=np.round(np.stack([xs,ys],axis=1)).astype(np.int32)
            cv2.polylines(mask,[coordinates],False,1,int(rng.integers(1,3)))
        contrast=float(rng.uniform(40,95));image-=mask*contrast
        for _ in range(int(rng.integers(0,4))):
            a=(int(rng.integers(size)),int(rng.integers(size)))
            b=(int(np.clip(a[0]+rng.integers(-10,11),0,size-1)),int(np.clip(a[1]+rng.integers(-10,11),0,size-1)))
            cv2.line(image,a,b,color=float(rng.uniform(50,100)),thickness=1)  # distractor
        image=cv2.GaussianBlur(image.astype(np.float32),(3,3),float(rng.uniform(.4,1)))
        # Inverted grayscale: higher number means darker film features.
        result.append(((255-np.clip(image,0,255))/255,mask.astype(np.float32),f'{split}:{j}'))
    return result


def measured_examples(root: Path, cfg: RunConfig):
    # Deferred imports permit procedural smoke without a measured archive.
    sys.path.insert(0, str(REPO_ROOT))
    from bubble_chamber.large_archive import audited_large_manifest
    from bubble_chamber.large_relation_study import frame_partition
    from bubble_chamber.synthetic_from_clicks import SyntheticFilmConfig, frame_render_dataset
    manifest=audited_large_manifest(root)
    frames=[f['frame_id'] for f in manifest['frames']]
    splits=frame_partition(frames,effective_data_seed(cfg))
    limits={'train':cfg.train_limit,'val':cfg.val_limit,'test':cfg.test_limit}
    data={}
    for name, frame_ids in splits.items():
        rows=[]
        cap=limits[name]
        # The renderer yields many contiguous examples per frame. Limit frame
        # contributions BEFORE the global cap to prevent first-frame test bias.
        per_frame_cap=max(1, int(np.ceil(cap / len(frame_ids)))) if cap else 0
        seen_per_frame={}
        for im,mask,meta in frame_render_dataset(manifest,frame_ids,
                                                 cfg=SyntheticFilmConfig(size=cfg.size,sample_seed=effective_data_seed(cfg)),repeats=3):
            source_frame=str(meta['source_frame'])
            used=seen_per_frame.get(source_frame,0)
            if cap and used >= per_frame_cap:continue
            seen_per_frame[source_frame]=used+1
            rows.append(((255-im).astype(np.float32)/255., mask.astype(np.float32), source_frame))
            if cap and len(rows)>=cap:break
        if not rows:raise ValueError(f'No eligible examples in {name} split')
        data[name]=rows
    for a in ('train','val','test'):
        for b in ('train','val','test'):
            if a<b and set(g for _,_,g in data[a]) & set(g for _,_,g in data[b]):
                raise ValueError(f'Leaked frame between {a} and {b}')
    return data,{'source_kind':'measured_track_shapes_synthetic_pixel_images',
                 'audit_sha256':manifest['provenance_sha256'],'frame_split_counts':{k:len(v) for k,v in splits.items()},
                 'film_ids':manifest['summary']['distinct_film_ids'],
                 'real_pixel_labels':False}


def make_datasets(cfg: RunConfig, measurements: Path | None):
    if cfg.source=='visualmetro':
        if measurements is None:raise ValueError('--measurements required')
        return measured_examples(measurements,cfg)
    if cfg.source!='procedural':raise ValueError('Unknown source')
    d={name:procedural_examples(getattr(cfg,f'{name}_limit'),cfg.size,seed=effective_data_seed(cfg),split=name)
       for name in ('train','val','test')}
    return d,{'source_kind':'fully_procedural_synthetic_software_smoke',
              'film_ids':[],'real_pixel_labels':False,'frame_split_counts':{k:len(v) for k,v in d.items()}}


def masked_loss(logits, target):
    if logits.shape!=target.shape:raise ValueError('target shape mismatch')
    pos=target.sum().clamp_min(1)
    neg=(1-target).sum().clamp_min(1)
    weight=(neg/pos).clamp(1,15)
    bce=nn.functional.binary_cross_entropy_with_logits(logits,target,pos_weight=weight)
    prob=logits.sigmoid()
    dice=1-(2*(prob*target).sum()+1)/((prob+target).sum()+1)
    return bce+dice


def metrics_counts(pred: np.ndarray, labels: np.ndarray):
    p=pred.astype(bool);y=labels.astype(bool)
    return (int(np.logical_and(p,y).sum()),int(np.logical_and(p,~y).sum()),int(np.logical_and(~p,y).sum()))


def metric_from_counts(counts):
    tp,fp,fn=map(int,counts)
    return {'tp':tp,'fp':fp,'fn':fn,'dice':2*tp/max(2*tp+fp+fn,1),
            'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1)}


def select_threshold(probs: np.ndarray, labels: np.ndarray):
    valid=np.arange(.1,.901,.05)
    scores=[metric_from_counts(metrics_counts(probs>=t,labels>=.5))['dice'] for t in valid]
    index=int(np.argmax(scores))
    return float(round(float(valid[index]),4))


def apply_stress(x: torch.Tensor, indices: torch.Tensor, cfg: RunConfig):
    result=x*cfg.stress_contrast
    for n,idx in enumerate(indices.tolist()):
        generator=torch.Generator(device='cpu').manual_seed(effective_data_seed(cfg)*37+int(idx))
        noise=torch.randn(x[n].shape,generator=generator,dtype=x.dtype)*cfg.stress_noise_std
        result[n]=(result[n]+noise).clamp(0,1)
    return result


def robust_augment(x: torch.Tensor, indices: torch.Tensor, cfg: RunConfig, epoch: int) -> torch.Tensor:
    """Deterministic train-only contrast/noise; same image, epoch, seed for both models."""
    if not 0.0 <= cfg.robust_probability <= 1.0 or not (0 < cfg.robust_scale_min <= cfg.robust_scale_max <= 1) or cfg.robust_noise_max < 0:
        raise ValueError('Invalid robust augmentation parameters')
    if x.shape[0] != len(indices):raise ValueError('batch/index mismatch')
    result=x.clone()
    for n,index in enumerate(indices.tolist()):
        g=torch.Generator(device='cpu').manual_seed(effective_data_seed(cfg)*100007+int(epoch)*4099+int(index)*13)
        if float(torch.rand((),generator=g)) >= cfg.robust_probability:continue
        scale=cfg.robust_scale_min+(cfg.robust_scale_max-cfg.robust_scale_min)*float(torch.rand((),generator=g))
        sigma=cfg.robust_noise_max*float(torch.rand((),generator=g))
        noise=torch.randn(result[n].shape,generator=g,dtype=result.dtype)*sigma
        result[n]=(result[n]*scale+noise).clamp(0,1)
    return result


def robust_validation_threshold(clean_prob: np.ndarray, stressed_prob: np.ndarray, labels: np.ndarray) -> float:
    """Use validation ONLY for the common clean/stress threshold."""
    if clean_prob.shape!=stressed_prob.shape or clean_prob.shape!=labels.shape:
        raise ValueError('Validation arrays must agree')
    candidates=np.arange(0.05,0.951,0.05)
    best=None
    for t in candidates:
        normal=metric_from_counts(metrics_counts(clean_prob>=t,labels>=.5))['dice']
        stress=metric_from_counts(metrics_counts(stressed_prob>=t,labels>=.5))['dice']
        key=(min(normal,stress),0.5*(normal+stress),-abs(float(t)-0.5))
        if best is None or key>best[0]:best=(key,float(t))
    return float(round(best[1],4))


@torch.no_grad()
def predict_probabilities(model, data: TrackDataset, cfg: RunConfig, *, stress: bool=False, device='cpu'):
    model.eval()
    loader=DataLoader(data,batch_size=cfg.batch_size,shuffle=False)
    probs=[];labels=[]
    for x,y,idx in loader:
        if stress:x=apply_stress(x,idx,cfg)
        z=model(x.to(device))
        if isinstance(z,dict):z=z['main']
        probs.extend(z.sigmoid().cpu().numpy()[:,0])
        labels.extend(y.numpy()[:,0])
    return np.stack(probs),np.stack(labels)


def paired_bootstrap(results_a,results_b,groups,*,seed:int,repeats=1000):
    """Paired bootstrap by original frame, never by pixel/patch."""
    pa, pb, ya = results_a[0],results_b[0],results_a[1]
    if not np.array_equal(ya,results_b[1]):raise ValueError('Different label sets')
    if len(groups)!=len(ya):raise ValueError('Missing group provenance')
    unique=sorted(set(groups));rng=np.random.default_rng(seed)
    idx_per_group={g:np.asarray([i for i,x in enumerate(groups) if x==g],int) for g in unique}
    def score(probs,indices):
        return metric_from_counts(metrics_counts(probs[indices],ya[indices]))['dice']
    ds=[]
    for _ in range(repeats):
        chosen=rng.choice(unique,size=len(unique),replace=True)
        inds=np.concatenate([idx_per_group[g] for g in chosen]);ds.append(score(pa,inds)-score(pb,inds))
    return {'mean_delta_dice':float(np.mean(ds)),'ci95':[float(v) for v in np.quantile(ds,[.025,.975])],
            'bootstrap_unit':'source_frame','unique_groups':len(unique),'bootstrap_repeats':repeats}


def build_models(cfg:RunConfig):
    # Imports strictly AFTER original source integrity gate.
    from models.segmentation import AZUNet, BaselineUNet
    from models.azconv import AZConvConfig
    torch.manual_seed(cfg.seed)  # independent, reproducible model initialization
    az=AZUNet(in_channels=1,out_channels=1, num_rules=cfg.num_rules, widths=cfg.widths,
              cfg=AZConvConfig(geometry_mode='local_hyperbolic',use_fuzzy=True,use_anisotropy=True,learn_directions=True,
                               use_input_residual=cfg.az_residual,residual_init=0.5 if cfg.az_residual else 0.0))
    torch.manual_seed(cfg.seed)
    base=BaselineUNet(in_channels=1,out_channels=1,widths=cfg.widths)
    return {'plain_unet':base,'original_azunet':az}


def save_atomic(path:Path,payload):
    tmp=path.with_suffix('.tmp')
    torch.save(payload,tmp)
    tmp.replace(path)


def split_fingerprints(sets):
    """Detect input changes before resuming a source-locked experiment."""
    result = {}
    for split, rows in sets.items():
        digest = hashlib.sha256()
        for x, y, group in rows:
            digest.update(np.ascontiguousarray(x).tobytes())
            digest.update(np.ascontiguousarray(y).tobytes())
            digest.update(str(group).encode('utf-8'))
        result[split] = {'sha256': digest.hexdigest(), 'examples': len(rows),
                         'source_groups': len({g for _, _, g in rows})}
    return result


def train_one(name, model, sets, cfg:RunConfig, outdir:Path, *, device='cpu',
              resume: bool=False, data_fingerprints=None):
    """Train or resume an identical run without touching held-out-test labels.

    Only best validation loss picks weights. Only validation data chooses a
    probability threshold. The saved RNG and optimizer permit exact resumes
    for deterministic models, while data checksums prevent dataset drift.
    """
    name_dir=outdir/name;name_dir.mkdir(parents=True,exist_ok=True)
    model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=.0001)
    best=float('inf');best_epoch=None;history=[];start_epoch=0
    train=TrackDataset(sets['train']);valid=TrackDataset(sets['val']);test=TrackDataset(sets['test'])
    hashes = data_fingerprints if data_fingerprints is not None else split_fingerprints(sets)
    if resume:
        latest = name_dir/'last.pt'
        best_file = name_dir/'best.pt'
        if not latest.is_file() or not best_file.is_file():
            raise FileNotFoundError(f'{name}: cannot resume without last.pt and best.pt')
        previous = torch.load(latest, map_location='cpu', weights_only=False)
        if previous['model'] != name or previous['source_blobs'] != SOURCE_BLOBS:
            raise ValueError(f'{name}: model/source changed since checkpoint')
        previous_cfg = dict(previous['config']); new_cfg = asdict(cfg)
        # Extending epochs is permitted. All learning and data parameters are fixed.
        previous_cfg.pop('epochs', None); new_cfg.pop('epochs', None)
        if previous_cfg != new_cfg:
            raise ValueError(f'{name}: training configuration changed on resume')
        if previous.get('data_fingerprints') != hashes:
            raise ValueError(f'{name}: training/validation/test data changed on resume')
        model.load_state_dict(previous['state_dict'])
        optimizer.load_state_dict(previous['optimizer'])
        history = list(previous['history'])
        start_epoch = int(previous['epoch']) + 1
        if start_epoch > cfg.epochs or len(history) != start_epoch:
            raise ValueError(f'{name}: invalid checkpoint epoch/history')
        best_payload = torch.load(best_file, map_location='cpu', weights_only=False)
        best_epoch = int(best_payload['epoch'])
        best = float(best_payload['history'][-1]['val_loss'])
        if 'torch_rng_state' in previous:
            torch.set_rng_state(previous['torch_rng_state'])
    started=time.monotonic()
    for ep in range(start_epoch, cfg.epochs):
        model.train();losses=[]
        loader=DataLoader(train,batch_size=cfg.batch_size,shuffle=True,
                          generator=torch.Generator().manual_seed(cfg.seed+ep))
        for step,(x,y,idx) in enumerate(loader):
            if cfg.steps_per_epoch and step>=cfg.steps_per_epoch:break
            if cfg.robust_train:x=robust_augment(x,idx,cfg,ep)
            optimizer.zero_grad(set_to_none=True)
            loss=masked_loss(model(x.to(device)),y.to(device))
            if not torch.isfinite(loss):raise RuntimeError(f'{name}: non-finite loss at epoch {ep} step {step}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step();losses.append(float(loss.detach().cpu()))
        if not losses:raise RuntimeError(f'{name}: no training batches')
        model.eval();val_losses=[]
        with torch.no_grad():
            for x,y,_ in DataLoader(valid,batch_size=cfg.batch_size):
                loss=masked_loss(model(x.to(device)),y.to(device))
                val_losses.append(float(loss.detach().cpu()))
        val_loss=float(np.mean(val_losses))
        history.append({'epoch':ep,'train_loss':float(np.mean(losses)),
                        'val_loss':val_loss,'train_steps':len(losses)})
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        payload={'model':name,'epoch':ep,'state_dict':state,
                 'optimizer':optimizer.state_dict(), 'torch_rng_state':torch.get_rng_state(),
                 'history':history,'config':asdict(cfg),'source_blobs':SOURCE_BLOBS,
                 'data_fingerprints':hashes,'synthetic_labels_only':True}
        save_atomic(name_dir/'last.pt',payload)
        if val_loss<best:
            best=val_loss;best_epoch=ep
            save_atomic(name_dir/'best.pt',payload)
        print(f'S7 {name} epoch {ep+1}/{cfg.epochs} train={np.mean(losses):.4f} val={val_loss:.4f}',flush=True)
    best_payload=torch.load(name_dir/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(best_payload['state_dict']);model=model.to(device)
    pv,yv=predict_probabilities(model,valid,cfg,device=device)
    if cfg.robust_train:
        pv_stress,yv_stress=predict_probabilities(model,valid,cfg,stress=True,device=device)
        if not np.array_equal(yv,yv_stress):raise ValueError('Validation target mismatch')
        threshold=robust_validation_threshold(pv,pv_stress,yv)
    else:threshold=select_threshold(pv,yv)
    pt,yt=predict_probabilities(model,test,cfg,device=device)
    ps,ys=predict_probabilities(model,test,cfg,stress=True,device=device)
    assert np.array_equal(yt,ys)
    clean=metric_from_counts(metrics_counts(pt>=threshold,yt>=.5))
    stress=metric_from_counts(metrics_counts(ps>=threshold,ys>=.5))
    from bubble_chamber.neural_metrics import aggregate_topology
    clean_topology=aggregate_topology(pt>=threshold,yt>=.5)
    stress_topology=aggregate_topology(ps>=threshold,ys>=.5)
    save_atomic(name_dir/'best_final.pt',{**best_payload,'validation_selected_threshold':threshold,
                                          'clean_test':clean,'stress_test':stress,
                                          'clean_topology':clean_topology,'stress_topology':stress_topology,
                                          'last_trained_epoch':cfg.epochs-1})
    result={'trained_epochs':cfg.epochs,'best_epoch':best_epoch,
            'parameters':sum(p.numel() for p in model.parameters()),
            'test_clean':clean,'test_stress':stress,
            'test_clean_topology':clean_topology,
            'test_stress_topology':stress_topology,
            'selected_validation_threshold':threshold,
            'elapsed_seconds_this_invocation':round(time.monotonic()-started,3),
            'checkpoint':'best_final.pt','history':history,
            'resumed_from_epoch':start_epoch if resume else None,
            'data_fingerprints':hashes}
    (name_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result,(pt>=threshold,yt),(ps>=threshold,ys)


def run(cfg:RunConfig,*,measurements:Path|None,out_dir:Path, resume:bool=False):
    verify_sources(REPO_ROOT)
    if cfg.size<32 or cfg.size%16 or cfg.batch_size<1 or cfg.epochs<1:
        raise ValueError('Invalid run size/epochs/batch')
    out_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(cfg.cpu_threads)
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    sets,source=make_datasets(cfg,measurements)
    source['synthetic_masks_only']=True
    # Preserve generated dataset hashes before model training, exclude pixel arrays from reports.
    fingerprints=split_fingerprints(sets)
    for split,meta in fingerprints.items():
        source[f'{split}_examples']={'n':meta['examples'],'sha256':meta['sha256'],'frames':meta['source_groups']}
    (out_dir/'dataset_manifest.json').write_text(json.dumps(source,indent=2)+'\n')
    models=build_models(cfg)
    results={};predictions={}
    for name,m in models.items():
        torch.manual_seed(cfg.seed)
        result,clean,stress=train_one(name,m,sets,cfg,out_dir,resume=resume,data_fingerprints=fingerprints)
        results[name]=result;predictions[name]={'clean':clean,'stress':stress}
    paired={}
    groups=TrackDataset(sets['test']).groups()
    for condition in ('clean','stress'):
        a=predictions['original_azunet'][condition]
        b=predictions['plain_unet'][condition]
        paired[condition]=paired_bootstrap(a,b,groups,seed=effective_data_seed(cfg)+233)
    # S10 fixed-source evaluation: only expert frame identifiers and sufficient
    # confusion counts. Never save source BMP images or synthetic masks here.
    if cfg.data_seed is not None:
        frame_counts={'data_seed':effective_data_seed(cfg),'model_seed':cfg.seed,
                      'test_groups':groups,'conditions':{}}
        for condition in ('clean','stress'):
            frame_counts['conditions'][condition]={}
            for model_name in ('plain_unet','original_azunet'):
                predicted,labels=predictions[model_name][condition]
                frame_counts['conditions'][condition][model_name]=[
                    list(metrics_counts(x,y)) for x,y in zip(predicted,labels)]
        (out_dir/'s10_test_frame_counts.json').write_text(json.dumps(frame_counts,indent=2)+'\n')
    report={'experiment':'S7_ORIGINAL_LOCKED_AZUNET_VS_UNET','original_source_git_blobs':SOURCE_BLOBS,
            'source_data':source,'config':asdict(cfg),'results':results,'paired':paired,
            'scientific_claim':'synthetic pixels generated from measured geometry only; not real film superiority',
            'source_port_parity':'UNVERIFIED: no retained S6 local port/checkpoint to compare',
            'frame_holdout': True,
            'film_holdout':False if cfg.source=='visualmetro' else None,
            'source_image_pixels_are_real': False, 'resume_requested':resume}
    (out_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',choices=['procedural','visualmetro'],default='procedural')
    parser.add_argument('--measurements',type=Path)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--size',type=int,default=64)
    parser.add_argument('--seed',type=int,default=20261024)
    parser.add_argument('--data-seed',type=int,default=None,help='Lock source frames/rendering/stress independently of model initialization')
    parser.add_argument('--widths',type=int,nargs=4,default=[8,16,24,32])
    parser.add_argument('--rules',type=int,default=2)
    parser.add_argument('--train-limit',type=int,default=48)
    parser.add_argument('--val-limit',type=int,default=16)
    parser.add_argument('--test-limit',type=int,default=24)
    parser.add_argument('--steps-per-epoch',type=int,default=0)
    parser.add_argument('--cpu-threads',type=int,default=2)
    parser.add_argument('--resume',action='store_true',help='Continue with matching last.pt and best.pt checkpoints')
    parser.add_argument('--az-residual',action='store_true',help='Use original AZConvConfig residual branch')
    parser.add_argument('--robust-train',action='store_true',help='Deterministic train-only contrast/noise and robust validation threshold')
    parser.add_argument('--robust-probability',type=float,default=0.8)
    parser.add_argument('--robust-scale-min',type=float,default=0.15)
    parser.add_argument('--robust-scale-max',type=float,default=1.0)
    parser.add_argument('--robust-noise-max',type=float,default=0.08)
    args=parser.parse_args()
    cfg=RunConfig(source=args.source,seed=args.seed,data_seed=args.data_seed,size=args.size,epochs=args.epochs,batch_size=args.batch_size,
                  widths=tuple(args.widths),num_rules=args.rules,train_limit=args.train_limit,val_limit=args.val_limit,
                  test_limit=args.test_limit,steps_per_epoch=args.steps_per_epoch,cpu_threads=args.cpu_threads,
                  az_residual=args.az_residual,robust_train=args.robust_train,
                  robust_probability=args.robust_probability,robust_scale_min=args.robust_scale_min,
                  robust_scale_max=args.robust_scale_max,robust_noise_max=args.robust_noise_max)
    report=run(cfg,measurements=args.measurements,out_dir=args.out_dir,resume=args.resume)
    print(json.dumps({'status':'S7_COMPLETE_SYNTHETIC_ONLY','clean_delta':report['paired']['clean'],
                      'stress_delta':report['paired']['stress']},indent=2),flush=True)

if __name__=='__main__':main()
