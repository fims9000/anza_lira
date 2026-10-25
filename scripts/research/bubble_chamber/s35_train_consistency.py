#!/usr/bin/env python3
"""S35 fixed-budget two-view photo consistency on S31 machine-only H50Z marks.

Only 005 or 006 is used for gradient updates per direction. Original AZConv
and BaselineUNet files are verified and are never modified. Models are frozen
at the preregistered last epoch; the other photograph does not choose epochs.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from bubble_chamber.s32_confidence import balanced_weak_windows,patch_coverage
from bubble_chamber.s35_photometric_consistency import S35Protocol,paired_exposure_views,consistency_loss
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model,verify_model_source
from scripts.research.bubble_chamber.s33_train_photometric import load_source_weak_masks


ARM_BY_ARCH={'azunet':'quantile','unet':'local'}


def _sha(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def make_matched_views(patches,indices,seed,epoch,mode):
    one=[];two=[];targets=[];conf=[]
    for row in indices:
        j=int(row);gray,mask,weight,*_=patches[j]
        # EXACT same geometric augmented targets, all arms.
        rot=int(np.random.default_rng(seed+j*13+epoch*10001).integers(0,4))
        gray=np.rot90(gray,rot).copy();mask=np.rot90(mask,rot).copy();weight=np.rot90(weight,rot).copy()
        a,b=paired_exposure_views(gray,mode,seed=seed*10007+epoch*7919+j)
        one.append(a);two.append(b);targets.append(mask);conf.append(weight)
    return (torch.from_numpy(np.stack(one)[:,None].astype(np.float32)),
            torch.from_numpy(np.stack(two)[:,None].astype(np.float32)),
            torch.from_numpy(np.stack(targets)[:,None].astype(np.int64)),
            torch.from_numpy(np.stack(conf)[:,None].astype(np.float32)))


def train(manifest:Path,tiles:Path,out:Path,*,train_photo:str,protocol:S35Protocol=S35Protocol(),
          architectures=('azunet','unet'),branches=('matched_two_view','photo_consistency')):
    protocol.check();verify_model_source();torch.set_num_threads(2)
    if train_photo not in ('005','006'):raise ValueError('Bad training photo')
    if set(architectures)-set(ARM_BY_ARCH) or set(branches)-{'matched_two_view','photo_consistency'}:
        raise ValueError('Unknown experimental arm')
    test_photo='006' if train_photo=='005' else '005';out=Path(out);out.mkdir(parents=True,exist_ok=True)
    groups,src=load_source_weak_masks(manifest,tiles)
    receipt={'schema':'s35_predeclared_photo_consistency_machine_only_v1',
       'protocol':asdict(protocol),'source':src,'trained_photo':train_photo,
       'reserved_test_photo':test_photo,'heldout_photo_in_optimizer':False,
       'test_used_for_epoch_selection':False,'original_AZConv_modified':False,
       'human_certified_track_labels':0,'physics_auto_join_authorized':False,
       'same_physical_run':True,'mode_by_arch':ARM_BY_ARCH,'results':[]}
    completed={}
    prior=out/'S35_training_receipt.json'
    if prior.is_file():
        stored=json.loads(prior.read_text())
        if (stored.get('schema')!=receipt['schema'] or stored.get('protocol')!=json.loads(json.dumps(receipt['protocol']))
            or stored.get('source')!=receipt['source'] or stored.get('trained_photo')!=train_photo):
            raise ValueError('Refusing to resume a run with mismatched scientific provenance')
        completed={(r['seed'],r['architecture'],r['branch']):r for r in stored['results']}
    for seed in protocol.train_seeds:
        patches=balanced_weak_windows(groups[train_photo],seed=seed,
            positives_per_crop=protocol.positives_per_crop,negatives_per_crop=protocol.negatives_per_crop)
        if not patches or not any(p[4]==0 for p in patches) or not any(p[4]==1 for p in patches):
            raise ValueError('Missing provisional classes')
        for arch in architectures:
            for branch in branches:
                key=(seed,arch,branch)
                if key in completed:
                    old=completed[key]
                    checkpoint=out/old['checkpoint_file']
                    if not checkpoint.is_file() or _sha(checkpoint)!=old['checkpoint_sha256']:
                        raise ValueError('Incomplete or modified checkpoint during resume')
                    meta=torch.load(checkpoint,map_location='cpu',weights_only=True)
                    if (meta.get('architecture')!=arch or meta.get('seed')!=seed or
                        meta.get('train_photo')!=train_photo or meta.get('branch')!=branch or
                        meta.get('labels_are_human_certified') is not False):
                        raise ValueError('Unsafe resumed model file')
                    receipt['results'].append(old)
                    print('S35 RESUMED',train_photo,seed,arch,branch,flush=True)
                    continue
                t0=time.monotonic();torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
                model=build_model(arch,widths=protocol.model_widths,rules=2)
                opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001)
                history=[]
                for epoch in range(protocol.epochs):
                    model.train();all_loss=[];all_sup=[];all_cons=[];unknown_count=0
                    perm=np.random.default_rng(seed+epoch).permutation(len(patches))
                    for i in range(0,len(perm),protocol.batch_size):
                        x1,x2,m,w=make_matched_views(patches,perm[i:i+protocol.batch_size],seed,epoch,ARM_BY_ARCH[arch])
                        p1=model(x1);p2=model(x2)
                        # Identical BCE supervision and two forward passes for controls.
                        loss,diag=consistency_loss(p1,p2,m,w,alpha=(protocol.lambda_consistency if epoch>=1 else 0.),
                            confidence_margin=protocol.soft_confidence_margin,
                            include_consistency=(branch=='photo_consistency' and epoch>=1))
                        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
                        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        opt.step();all_loss.append(float(loss.detach()));all_sup.append(diag['supervised']);
                        all_cons.append(diag['consistency']);unknown_count+=diag['consistency_unknown_pixels']
                    history.append({'epoch':epoch+1,'train_loss':float(np.mean(all_loss)),
                        'labelled_weak_supervised_BCE':float(np.mean(all_sup)),
                        'unknown_photometric_consistency_MSE':float(np.mean(all_cons)),
                        'unknown_consistency_pixel_visits':unknown_count})
                file=out/f'{arch}_{branch}_train{train_photo}_seed{seed}.pt'
                torch.save({'schema':'s35_machine_only_photo_view_consistency_v1',
                    'architecture':arch,'mode':ARM_BY_ARCH[arch], 'train_photo':train_photo,
                    'test_photo_not_used_for_training':test_photo,'seed':seed,
                    'branch':branch,'last_epoch':protocol.epochs,
                    'source_weak_manifest_sha256':src['weak_manifest_sha256'],
                    'labels_are_human_certified':False,'physical_auto_joins_authorized':False,
                    'model_state':model.state_dict()},file)
                rec={'architecture':arch,'branch':branch,'seed':seed,'train_photo':train_photo,
                     'mode':ARM_BY_ARCH[arch],'train_windows':len(patches),
                     'machine_mark_counts':patch_coverage(patches),'epochs':history,
                     'checkpoint_file':file.name,'checkpoint_sha256':_sha(file),
                     'seconds':round(time.monotonic()-t0,2)}
                receipt['results'].append(rec)
                (out/'S35_training_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
                print('S35',train_photo,seed,arch,branch,round(history[-1]['train_loss'],5),'seconds',rec['seconds'],flush=True)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--tiles',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--train-photo',choices=('005','006'),required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[20261025,20261026])
    a=p.parse_args();cfg=S35Protocol(train_seeds=tuple(a.seeds))
    train(a.manifest,a.tiles,a.out,train_photo=a.train_photo,protocol=cfg)
