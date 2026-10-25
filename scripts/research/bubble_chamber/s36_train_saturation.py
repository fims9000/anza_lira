#!/usr/bin/env python3
"""S36 train AZUNet with fit-free exposure saturation mixing, same budget as S35.

Machine S31 weak marks only; train on one source photo, never other photo's
statistics/labels. Quantile AZUNet baseline is the matched-compute S35 model.
"""
from __future__ import annotations
import argparse,json,sys,time,random
from pathlib import Path
from hashlib import sha256
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s32_confidence import balanced_weak_windows,confidence_bce,patch_coverage
from bubble_chamber.s36_saturation_normalization import matched_s36_photo_views,S36_TEST_FORWARD_SEEDS,S36_TEST_REVERSE_SEEDS
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model,verify_model_source
from scripts.research.bubble_chamber.s33_train_photometric import load_source_weak_masks

TRAIN_SEEDS=(20261025,20261026)


def train(manifest,tiles,out,train_photo):
    if train_photo not in ('005','006'):raise ValueError('Bad training photo')
    verify_model_source();torch.set_num_threads(2)
    groups,prov=load_source_weak_masks(manifest,tiles)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    receipt={'schema':'s36_saturation_fit_free_matched_original_azunet_machine_only_v1',
     'source':prov,'train_photo':train_photo,'holdout_photo':'006' if train_photo=='005' else '005',
     'train_seeds':list(TRAIN_SEEDS),'heldout_seeds':list(S36_TEST_FORWARD_SEEDS if train_photo=='005' else S36_TEST_REVERSE_SEEDS),
     'hyperparameters':{'epochs':2,'batch_size':4,'positive_windows_per_crop':5,'negative_windows_per_crop':5,'adamw_lr':.0005,'decay':.0001},
     'human_certified_tracks':0,'independent_physical_run':False,'physical_auto_join':False,'results':[]}
    for seed in TRAIN_SEEDS:
        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        model=build_model('azunet',widths=(4,6,8,12),rules=2)
        optim=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001)
        rows=balanced_weak_windows(groups[train_photo],seed=seed,positives_per_crop=5,negatives_per_crop=5)
        hist=[];t0=time.monotonic()
        for epoch in range(2):
            model.train();losses=[]
            perm=np.random.default_rng(seed+epoch).permutation(len(rows))
            for i in range(0,len(perm),4):
                a=[];b=[];ys=[];ws=[]
                for ix in perm[i:i+4]:
                    j=int(ix);im,y,w,*_=rows[j]
                    rot=int(np.random.default_rng(seed+j*13+epoch*10001).integers(0,4))
                    im=np.rot90(im,rot).copy();y=np.rot90(y,rot).copy();w=np.rot90(w,rot).copy()
                    p,q=matched_s36_photo_views(im,seed=seed*10007+epoch*7919+j)
                    a.append(p);b.append(q);ys.append(y);ws.append(w)
                x=torch.from_numpy(np.stack(a)[:,None]);x2=torch.from_numpy(np.stack(b)[:,None]);
                y=torch.from_numpy(np.stack(ys)[:,None].astype(np.int64))
                w=torch.from_numpy(np.stack(ws)[:,None].astype(np.float32))
                loss1=confidence_bce(model(x),y,w);loss2=confidence_bce(model(x2),y,w)
                if loss1 is None or loss2 is None:raise ValueError('No pseudo labels')
                loss=(loss1+loss2)/2
                if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
                optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
                losses.append(float(loss.detach()))
            hist.append({'epoch':epoch+1,'mean_machine_pseudo_label_BCE':float(np.mean(losses))})
        ck=out/f'azunet_saturation_blend_train{train_photo}_seed{seed}.pt'
        torch.save({'schema':'s36_machine_only_azunet_saturation_blend_v1',
             'architecture':'azunet','mode':'saturation_blend','source_photo':train_photo,
             'test_photo_unused':'006' if train_photo=='005' else '005',
             'train_seed':seed,'epoch':2,'human_physical_track_truth':False,'physical_auto_join_authorized':False,
             'source_manifest_sha256':prov['weak_manifest_sha256'],'model_state':model.state_dict()},ck)
        receipt['results'].append({'seed':seed,'checkpoint':ck.name,'checkpoint_sha256':sha256(ck.read_bytes()).hexdigest(),
           'loss_history':hist,'pseudo_window_count':len(rows),'pseudo_source_coverage':patch_coverage(rows),'seconds':time.monotonic()-t0})
        (out/'S36_training_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
        print('S36 TRAIN',train_photo,seed,round(hist[-1]['mean_machine_pseudo_label_BCE'],4),flush=True)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--tiles',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--train-photo',choices=('005','006'),required=True)
    a=p.parse_args();train(a.manifest,a.tiles,a.out,a.train_photo)
