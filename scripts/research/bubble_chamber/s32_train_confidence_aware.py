#!/usr/bin/env python3
"""S32: matched original ANZA and U-Net comparison of S31 unweighted vs weighted pseudo labels.

No expert labels exist. Validation on photo 006 is pseudo-label *agreement*;
physical accuracy requires independent human-certified masks and film runs.
"""
from __future__ import annotations
import argparse, json, random, sys, hashlib, time
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model,verify_model_source
from scripts.research.bubble_chamber.s31_train_provisional_anza import load_weak_data,split_bce
from bubble_chamber.s32_confidence import confidence_bce,balanced_weak_windows,patch_coverage


def training_loss(logits,mark,weights,mode):
    if mode == 's31_unweighted_control':return split_bce(logits,mark,weights)
    if mode == 's32_confidence':return confidence_bce(logits,mark,weights)
    raise ValueError('Unknown experimental arm')


def score_patches(model,patches):
    out={'pseudo_positive':0,'pseudo_positive_detected':0,'assumed_negative':0,
         'assumed_negative_predicted_track':0,'known_bce_sum':0.,'known_bce_n':0}
    model.eval()
    with torch.no_grad():
        for image,mark,weights,_,_ in patches:
            x=torch.from_numpy((255-image).astype(np.float32)[None,None]/255.)
            pred=model(x)[0,0]
            p=torch.sigmoid(pred)
            pos=torch.from_numpy(mark==1);neg=torch.from_numpy(mark==0)
            out['pseudo_positive']+=int(pos.sum());out['pseudo_positive_detected']+=int((p[pos]>=.5).sum())
            out['assumed_negative']+=int(neg.sum());out['assumed_negative_predicted_track']+=int((p[neg]>=.5).sum())
            known=pos|neg
            if bool(known.any()):
                yy=torch.from_numpy(np.where(mark==1,.92,.08).astype(np.float32))
                out['known_bce_sum']+=float(F.binary_cross_entropy_with_logits(pred[known],yy[known]))
                out['known_bce_n']+=1
    return {'pseudo_positive_fraction':out['pseudo_positive_detected']/max(out['pseudo_positive'],1),
            'assumed_negative_fraction_predicted_track':out['assumed_negative_predicted_track']/max(out['assumed_negative'],1),
            'pseudo_unweighted_BCE_common_metric':out['known_bce_sum']/max(out['known_bce_n'],1),
            'pseudo_positive_pixels_evaluated':out['pseudo_positive'],
            'assumed_negative_pixels_evaluated':out['assumed_negative'],
            'human_track_quality':'NOT_MEASURED','physical_false_joins':'NOT_MEASURED'}


def train(manifest,tiles,out,*,epochs=3,seeds=(20261025,20261026),
          positives_per_crop=5,negatives_per_crop=5):
    verify_model_source()
    if not 1<=epochs<=15 or len(set(seeds))!=len(seeds):raise ValueError('Invalid experimental budget')
    torch.set_num_threads(2)
    groups,source=load_weak_data(Path(manifest),Path(tiles))
    manifest_sha=hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    receipt={'schema':'s32_confidence_aware_real_photo_pseudo_ablation_v1',
             'source_weak_manifest_sha256':manifest_sha,'machine_marks_are_human_certified':False,
             'human_certified_track_pixels':0,'physical_automatic_joins_authorized':False,
             'train_original_photo':'005','validation_original_photo':'006',
             'same_physical_run':True,'models_original_source_unchanged':True,
             'epochs':epochs,'training_labels':'machine_0_1_255_only',
             'confidence_modes':['s31_unweighted_control','s32_confidence'],
             'results':[]}
    for seed in seeds:
        train_p=balanced_weak_windows(groups['005'],positives_per_crop=positives_per_crop,
                negatives_per_crop=negatives_per_crop,seed=seed)
        val_p=balanced_weak_windows(groups['006'],positives_per_crop=positives_per_crop,
                negatives_per_crop=negatives_per_crop,seed=seed+999)
        if len(train_p)<10 or len(val_p)<4 or not any(q[4]==0 for q in train_p+val_p):
            raise ValueError('Insufficient pseudo class coverage')
        source_summary={'seed':seed,'train':patch_coverage(train_p),'val':patch_coverage(val_p)}
        for architecture in ('unet','azunet'):
            for mode in receipt['confidence_modes']:
                torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
                model=build_model(architecture,widths=(4,6,8,12),rules=2)
                opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
                hist=[];best=None;start=time.monotonic();dest=out/f'{architecture}_{mode}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
                for epoch in range(epochs):
                    model.train();losses=[]
                    for i in np.random.default_rng(seed+epoch).permutation(len(train_p)):
                        image,mark,weights,_,_ =train_p[int(i)]
                        rot=int(np.random.default_rng(seed+epoch*10001+int(i)).integers(0,4))
                        image=np.rot90(image,rot).copy();mark=np.rot90(mark,rot).copy();weights=np.rot90(weights,rot).copy()
                        x=torch.from_numpy((255-image).astype('float32')[None,None]/255.)
                        m=torch.from_numpy(mark.astype('int64'))[None,None]
                        w=torch.from_numpy(weights.astype('float32'))[None,None]
                        logits=model(x);loss=training_loss(logits,m,w,mode)
                        if loss is None or not bool(torch.isfinite(loss)):raise ValueError('Invalid pseudo training loss')
                        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                        losses.append(float(loss.detach()))
                    val=score_patches(model,val_p)
                    hist.append({'epoch':epoch+1,'train_machine_loss':float(np.mean(losses)),
                                 'val_pseudo_unweighted_common_BCE':val['pseudo_unweighted_BCE_common_metric']})
                    if best is None or val['pseudo_unweighted_BCE_common_metric']<best[0]:
                        best=(val['pseudo_unweighted_BCE_common_metric'],epoch+1)
                        ck={'schema':'s32_machine_only_original_model_v1','architecture':architecture,
                            'seed':seed,'epoch':epoch+1,'model_state':model.state_dict(),
                            'source_manifest_sha256':manifest_sha,'label_type':'MACHINE_DRAFT_NOT_HUMAN_EXPERT',
                            'kind':architecture,'source_photo':'005','selection_metric':'pseudo_BCE_on_photo_006_not_real_accuracy',
                            'physical_auto_joins_authorized':False}
                        torch.save(ck,dest/'best.pt')
                selected=torch.load(dest/'best.pt',map_location='cpu',weights_only=True)
                model.load_state_dict(selected['model_state'])
                detail={'seed':seed,'architecture':architecture,'mode':mode,
                        'train_patch_coverage':source_summary['train'],
                        'val_patch_coverage':source_summary['val'],
                        'history':hist,'best_epoch':best[1],
                        'validation_proxy':score_patches(model,val_p),
                        'checkpoint_sha256':hashlib.sha256((dest/'best.pt').read_bytes()).hexdigest(),
                        'checkpoint_relative_path':f'{dest.name}/best.pt',
                        'training_seconds':round(time.monotonic()-start,2)}
                receipt['results'].append(detail)
                (out/'S32_ablation_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
                print('S32 TRAIN',seed,architecture,mode,detail['validation_proxy'],flush=True)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--tiles',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--seeds',type=int,nargs='+',default=[20261025,20261026]);
    p.add_argument('--positive-windows',type=int,default=5);p.add_argument('--negative-windows',type=int,default=5)
    a=p.parse_args()
    train(a.manifest,a.tiles,a.out,epochs=a.epochs,seeds=tuple(a.seeds),
          positives_per_crop=a.positive_windows,negatives_per_crop=a.negative_windows)
