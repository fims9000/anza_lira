#!/usr/bin/env python3
"""S34 preregistered ablation: source-locked original AZUNet plus axial loss.

Original AZConv source unchanged. Geometry supervision is a 0/1/255 MACHINE
MASK, not a true human segmented or physically certified track.
"""
from __future__ import annotations
import argparse, json,sys,random,time
from pathlib import Path
from hashlib import sha256
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s32_confidence import balanced_weak_windows,confidence_bce
from scripts.research.bubble_chamber.s33_train_photometric import load_source_weak_masks,make_batch
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model,verify_model_source

FIXED_AXIS_WEIGHT=.03
FIXED_ITERATIONS=5
SOURCE_INPUT_MODE='local'


def fit(manifest,tiles,out,*,seeds=(20261025,20261026),epochs=2,positives_per_crop=5,negatives_per_crop=5,batch_size=4):
    verify_model_source()
    if epochs!=2 or (positives_per_crop,negatives_per_crop,batch_size)!=(5,5,4):
        raise ValueError('S34 preregistered budget frozen, not tuneable on test')
    torch.set_num_threads(2)
    groups,receipt=load_source_weak_masks(manifest,tiles)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    summary={'schema':'s34_azconv_axial_consistency_on_machine_centerlines_v1',
             'model_original_AZConv_code_modified':False,
             'source_manifest_sha256':sha256(Path(manifest).read_bytes()).hexdigest(),
             'train_photo':'005','test_photo':'006','physical_run_count':1,
             'lambda_axis_pre_registered':FIXED_AXIS_WEIGHT,'iterations':FIXED_ITERATIONS,
             'axis_truth':'PROVISIONAL_MACHINE_CENTERLINES_NOT_HUMAN',
             'physics_automatic_joins_authorized':False,
             'test_injection_seeds_reserved':[20261033,20261034,20261035,20261036],
             'results':[]}
    for seed in seeds:
        train=balanced_weak_windows(groups['005'],seed=seed,positives_per_crop=5,negatives_per_crop=5)
        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        model=build_model('azunet',widths=(4,6,8,12),rules=2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001)
        t0=time.monotonic();hist=[]
        for epoch in range(epochs):
            model.train();bces=[];axes=[];allloss=[]
            order=np.random.default_rng(seed+epoch).permutation(len(train))
            for j in range(0,len(train),batch_size):
                image,mask,w=make_batch(train,order[j:j+batch_size],seed=seed,epoch=epoch,mode=SOURCE_INPUT_MODE)
                logits=model(image)
                bce=confidence_bce(logits,mask,w)
                axis=model.axis_alignment_loss((mask==1).float(),(mask!=255).float(),num_iters=FIXED_ITERATIONS)
                if bce is None or not torch.isfinite(axis):raise RuntimeError('Non-finite or empty axis training objective')
                loss=bce+FIXED_AXIS_WEIGHT*axis
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
                bces.append(float(bce.detach()));axes.append(float(axis.detach()));allloss.append(float(loss.detach()))
            hist.append({'epoch':epoch+1,'machine_BCE':float(np.mean(bces)),
                         'machine_axis_loss':float(np.mean(axes)),
                         'combined_loss':float(np.mean(allloss))})
        ck={'schema':'s34_machine_only_AZUNet_axial_ablation_v1',
            'architecture':'azunet','mode':'local','seed':seed,'epoch':epochs,
            'source_manifest_sha256':summary['source_manifest_sha256'],
            'pixel_labels_are_human_certified':False,'axial_target_is_machine_derived':True,
            'train_photo':'005','test_photo_used_for_selection':False,
            'physical_automatic_join_authorized':False,'model_state':model.state_dict()}
        file=out/f'azunet_local_axis_seed{seed}.pt';torch.save(ck,file)
        summary['results'].append({'seed':seed,'history':hist,'train_patches':len(train),
             'checkpoint':file.name,'checkpoint_sha256':sha256(file.read_bytes()).hexdigest(),
             'training_seconds':round(time.monotonic()-t0,2)})
        print('S34 AXIS',seed,hist[-1],flush=True)
    (out/'S34_training_receipt.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--manifest',type=Path,required=True)
 p.add_argument('--tiles',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 fit(a.manifest,a.tiles,a.out)
