#!/usr/bin/env python3
"""S33: matched photometric normalization comparison on H50Z machine masks.

The source is the original 005 BMP's 512px crops; 006 NEVER trains and is
NEVER used for checkpoint selection. All checkpoints are fixed last epoch.
The targets 0/1/255 are fallible machine drafts, NOT physicist track truth.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from bubble_chamber.s32_confidence import balanced_weak_windows, confidence_bce, patch_coverage
from bubble_chamber.s33_photometric_domain import image_input, input_domain_audit, photometric_augment
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model, verify_model_source


RESEARCH_ARMS=('raw','quantile','local')


def _sha(file:Path)->str:
    return sha256(Path(file).read_bytes()).hexdigest()


def load_source_weak_masks(manifest:Path,tiles:Path):
    """Fail closed if source bitmap or 0/1/255 weak mark provenance changes."""
    manifest=Path(manifest).resolve();tiles=Path(tiles).resolve()
    x=json.loads(manifest.read_text(encoding='utf8'))
    if (x.get('schema')!='s31_weak_training_masks_source_locked_v1' or
        x.get('human_certified_track_pixels')!=0 or
        x.get('physics_auto_join_allowed') is not False or
        len(x.get('entries',[]))!=39):
        raise ValueError('Wrong S31 weak photo protocol; NOT human-certified data')
    result={'005':[],'006':[]};seen=set()
    for row in x['entries']:
        cid=row['crop_id'];photo=row['photo_id']
        if photo not in result or cid in seen or not cid.startswith('H50Z_030_01_'+photo+'_'):
            raise ValueError('Invalid photo identity or duplicate crop')
        seen.add(cid)
        def safe(p,root):
            q=(root/p).resolve()
            if not q.is_relative_to(root) or not q.is_file():
                raise ValueError('Unsafe, missing or changed input '+str(p))
            return q
        src=safe(cid+'.png',tiles)
        mask=safe(row['mask'],manifest.parent)
        conf=safe(row['weight'],manifest.parent)
        if _sha(src)!=row['image_sha256'] or _sha(mask)!=row['mask_sha256'] or _sha(conf)!=row['weight_sha256']:
            raise ValueError('Source SHA-256 mismatch, cannot silently retrain')
        im=cv2.imread(str(src),cv2.IMREAD_GRAYSCALE)
        mk=cv2.imread(str(mask),cv2.IMREAD_UNCHANGED)
        wt=np.load(conf,allow_pickle=False).astype('float32')
        if any(q.shape!=(512,512) for q in (im,mk,wt)) or mk.dtype!=np.uint8 or im.dtype!=np.uint8:
            raise ValueError('Source crop or mask dimensions/dtype mismatch')
        if set(np.unique(mk))-{0,1,255} or not np.isfinite(wt).all() or np.any(wt[mk==255]!=0):
            raise ValueError('Unknown photo pixels must have ZERO weight')
        result[photo].append((im,mk,wt,row))
    if len(result['005'])!=21 or len(result['006'])!=18:
        raise ValueError('Expected exact photo-disjoint split 21 train / 18 test')
    return result, {'weak_manifest_sha256':_sha(manifest),'photos':['005','006'],'physically_certified_track_pixels':0}


def make_batch(patches,indices,*,seed,epoch,mode):
    xs=[];ys=[];ws=[]
    for i in indices:
        gray,mark,confidence,*_=patches[int(i)]
        # Identical spatial transformations across every architecture/mode.
        rot=int(np.random.default_rng(seed+int(i)*13+epoch*10001).integers(0,4))
        gray=np.rot90(gray,rot).copy();mark=np.rot90(mark,rot).copy();confidence=np.rot90(confidence,rot).copy()
        if mode=='raw_aug':
            gray=photometric_augment(gray,seed=seed*10007+epoch*7919+int(i))
        x=image_input(gray,'raw' if mode=='raw_aug' else mode)
        xs.append(x);ys.append(mark);ws.append(confidence)
    return (torch.from_numpy(np.stack(xs)[:,None]),
            torch.from_numpy(np.stack(ys).astype('int64')[:,None]),
            torch.from_numpy(np.stack(ws).astype('float32')[:,None]))


def train_ablation(manifest:Path,tiles:Path,out:Path,*,seeds=(20261025,20261026),
                   architectures=('azunet','unet'),modes=RESEARCH_ARMS,
                   epochs=2,positives_per_crop=5,negatives_per_crop=5,batch_size=4,
                   train_photo='005'):
    verify_model_source()
    if not 1<=epochs<=10 or not 1<=batch_size<=8 or not seeds:
        raise ValueError('Invalid pre-registered budget')
    if any(x not in ('azunet','unet') for x in architectures):raise ValueError('Invalid architecture')
    if any(x not in RESEARCH_ARMS+('raw_aug',) for x in modes):raise ValueError('Invalid preprocessing arm')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    if train_photo not in {'005','006'}:
        raise ValueError('Only the two authenticated source photos may train')
    test_photo='006' if train_photo=='005' else '005'
    groups,src=load_source_weak_masks(manifest,tiles)
    audit=input_domain_audit({photo:[t[0] for t in items] for photo,items in groups.items()})
    receipt={'schema':'s33_source_locked_photo_normalization_v1',
       'data':src,'source_photo_train':train_photo,'source_photo_test':test_photo,
       'test_labels_real_approved':False,'source_photos_from_same_run':True,
       'original_AZConv_code_unchanged':True,'photo_006_used_to_select_epoch':False,
       'train_epochs_fixed_not_chosen_by_test':epochs,
       'machine_0_1_255_pseudolabels_only':True,
       'domain_audit_descriptive_not_test_selection':audit,
       'modes':list(modes),'seeds':list(seeds),'results':[],
       'physics_links_authorized':False}
    started=time.monotonic()
    for seed in seeds:
        src_train=balanced_weak_windows(groups[train_photo],seed=seed,
           positives_per_crop=positives_per_crop,negatives_per_crop=negatives_per_crop)
        if not any(x[4]==1 for x in src_train) or not any(x[4]==0 for x in src_train):
            raise ValueError('No positive/assumed-negative machine marks in training data')
        # The 006 masks are not loaded into gradient updates or epoch selection.
        for arch in architectures:
            for mode in modes:
                torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
                model=build_model(arch,widths=(4,6,8,12),rules=2)
                opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001)
                history=[];t0=time.monotonic()
                for epoch in range(epochs):
                    model.train();losses=[]
                    permutation=np.random.default_rng(seed+epoch).permutation(len(src_train))
                    for k in range(0,len(permutation),batch_size):
                        x,m,w=make_batch(src_train,permutation[k:k+batch_size],
                                         seed=seed,epoch=epoch,mode=mode)
                        logits=model(x)
                        loss=confidence_bce(logits,m,w)
                        if loss is None or not bool(torch.isfinite(loss)):
                            raise RuntimeError('Missing or nonfinite weak confidence objective')
                        opt.zero_grad(set_to_none=True);loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
                        losses.append(float(loss.detach()))
                    history.append({'epoch':epoch+1,'machine_train_confidence_BCE':float(np.mean(losses))})
                save=out/f'{arch}_{mode}_seed{seed}.pt'
                state={'schema':'s33_machine_draft_photo_normalized_v1','architecture':arch,'mode':mode,
                       'seed':seed,'epoch':epochs,'photo_train':train_photo,'test_photo_NOT_used_for_training':test_photo,
                       'label_type':'MACHINE_DRAFT_NOT_HUMAN_EXPERT','source_weak_manifest_sha256':src['weak_manifest_sha256'],
                       'physical_auto_joins_authorized':False,'model_state':model.state_dict()}
                torch.save(state,save)
                rec={'architecture':arch,'mode':mode,'seed':seed,'epochs':epochs,
                     'train_windows':len(src_train),'train_coverage':patch_coverage(src_train),
                     'history':history,'checkpoint_file':save.name,'checkpoint_sha256':_sha(save),
                     'elapsed_sec':round(time.monotonic()-t0,2),'real_photo_pixel_accuracy':'NOT_MEASURED'}
                receipt['results'].append(rec)
                (out/'S33_training_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
                print('S33',seed,arch,mode,'train loss',round(history[-1]['machine_train_confidence_BCE'],4),'seconds',rec['elapsed_sec'],flush=True)
    receipt['elapsed_sec']=round(time.monotonic()-started,2)
    (out/'S33_training_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--tiles',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[20261025,20261026])
    p.add_argument('--architectures',nargs='+',default=['azunet','unet'])
    p.add_argument('--modes',nargs='+',default=list(RESEARCH_ARMS))
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--positives-per-crop',type=int,default=5)
    p.add_argument('--negatives-per-crop',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--train-photo',choices=['005','006'],default='005')
    a=p.parse_args()
    train_ablation(a.manifest,a.tiles,a.out,seeds=tuple(a.seeds),architectures=tuple(a.architectures),
                   modes=tuple(a.modes),epochs=a.epochs,positives_per_crop=a.positives_per_crop,
                   negatives_per_crop=a.negatives_per_crop,batch_size=a.batch_size,train_photo=a.train_photo)
