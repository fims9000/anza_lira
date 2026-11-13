#!/usr/bin/env python3
"""V26: train actual original AZConv2d vs matched ordinary conv on real CRACKS.

ANZA is a fresh *small* neural detector using repository AZConv2d; NOT the
historical saved ANZA checkpoint. LIRA downstream rule is deterministic V25,
NOT the historical trained LIRA network. Expert masks appear only after
all decisions. V26 is post-V23/25 exploratory on one F3 volume.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F

from models.azconv import AZConv2d, AZConvConfig
from scripts.cracks_real_auto_run_v23 import (
    TRAIN_IDS,CAL_IDS,SEED,BLUE,ORANGE,zip_index,load_image,load_mask,
    consensus_training_mask,_train,evaluate_mask,candidate_reachability,
    _bootstrap,sha_file,_overlay,
)
from scripts.cracks_real_auto_core_v23 import anisotropic_features,run_pipeline
from scripts.cracks_real_region_v25 import run_regions

METHODS=('hgb','conv','az')
POLICIES=('raw','context','region_hybrid')
THRESHOLDS=(.15,.25,.35,.45,.55,.65,.75,.85)
PATCH=80
BATCH=4
TRAIN_STEPS=180
TRAIN_LR=.002

class CompactDetector(nn.Module):
    """Only first spatial operator differs: exact repo AZConv2d vs Conv2d."""
    def __init__(self,variant):
        super().__init__()
        if variant not in ('conv','az'):raise ValueError(variant)
        if variant=='az':
            self.stem=AZConv2d(
                in_channels=1,out_channels=8,kernel_size=3,num_rules=2,
                cfg=AZConvConfig(geometry_mode='local_hyperbolic'))
        else:
            self.stem=nn.Conv2d(1,8,kernel_size=3,padding=1)
        self.norm=nn.GroupNorm(2,8)
        self.body=nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8,8,3,padding=2,dilation=2),
            nn.ReLU(),
            nn.Conv2d(8,8,3,padding=4,dilation=4),
            nn.ReLU(),
            nn.Conv2d(8,1,1))
    def forward(self,x):
        return self.body(self.norm(self.stem(x)))

def normalize_image(image):
    """Frozen linear intensity normalization independent of expert pixels."""
    return np.clip((np.asarray(image,dtype=np.float32)-.5)/.25,-3.,3.)

def crowd_records(zi,za,images,ann,section_ids):
    records=[]
    for sid in section_ids:
        if sid not in images or sid not in ann:continue
        if 'expert' in ann[sid]:
            continue
        masks=consensus_training_mask(za,ann[sid])
        if masks is None:continue
        p,n=masks
        if int(p.sum())<15 or int(n.sum())<30:continue
        im=normalize_image(load_image(zi,images[sid]))
        if im.shape!=p.shape or p.shape!=n.shape:
            raise ValueError('Image/annotation shape mismatch on '+str(sid))
        records.append({
            'section':sid,'image':im,'positive':p,'negative':n,
            'pos_idx':np.argwhere(p),'neg_idx':np.argwhere(n)})
    return records

def sample_batch(records,rng,batch=BATCH,patch=PATCH):
    xs=np.zeros((batch,1,patch,patch),np.float32)
    targets=np.zeros_like(xs)
    known=np.zeros_like(xs)
    for j in range(batch):
        rec=records[int(rng.integers(len(records)))]
        # Explicitly alternate classes across four patches so no training
        # batch can contain only BLUE or only ORANGE; both models get
        # exactly the same deterministic patch sequence and budgets.
        selection=rec['pos_idx'] if j%2==0 else rec['neg_idx']
        if not len(selection):
            selection=rec['pos_idx'] if len(rec['pos_idx']) else rec['neg_idx']
        center=selection[int(rng.integers(len(selection)))]
        cy=int(center[0])+int(rng.integers(-11,12))
        cx=int(center[1])+int(rng.integers(-11,12))
        h,w=rec['image'].shape
        top=int(np.clip(cy-patch//2,0,h-patch))
        left=int(np.clip(cx-patch//2,0,w-patch))
        ys=slice(top,top+patch);xx=slice(left,left+patch)
        xs[j,0]=rec['image'][ys,xx]
        pos=rec['positive'][ys,xx]
        neg=rec['negative'][ys,xx]
        targets[j,0]=pos.astype(np.float32)
        known[j,0]=(pos|neg).astype(np.float32)
    return torch.from_numpy(xs),torch.from_numpy(targets),torch.from_numpy(known)

def masked_binary_loss(logits,target,known):
    """Macro-balanced on actual explicit BLUE and ORANGE pixels per batch."""
    per_pixel=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    positive=(known>0)&(target>.5)
    negative=(known>0)&(target<=.5)
    counts=(int(positive.sum()),int(negative.sum()))
    if min(counts)<1:raise ValueError('Batch lacks explicit class')
    return .5*(per_pixel[positive].mean()+per_pixel[negative].mean())

def dense_probs(model,image):
    x=torch.from_numpy(normalize_image(image)[None,None])
    model.eval()
    with torch.inference_mode():
        p=torch.sigmoid(model(x))[0,0].cpu().numpy()
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Nonfinite/invalid segmentation probability')
    return p.astype(np.float32)

def fit_model(variant,records,out):
    torch.manual_seed(SEED)
    random.seed(SEED)
    rng=np.random.default_rng(SEED)
    model=CompactDetector(variant)
    optimizer=torch.optim.Adam(model.parameters(),lr=TRAIN_LR,weight_decay=1.e-5)
    record=[]
    model.train()
    for step in range(TRAIN_STEPS):
        x,y,m=sample_batch(records,rng)
        optimizer.zero_grad(set_to_none=True)
        logits=model(x)
        try:
            loss=masked_binary_loss(logits,y,m)
        except ValueError:
            # An occasional class-empty crop may occur due to sparse labels.
            # Deterministically resample; do not weaken the class requirement.
            x,y,m=sample_batch(records,rng)
            logits=model(x)
            loss=masked_binary_loss(logits,y,m)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite training objective for '+variant)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),max_norm=2.)
        optimizer.step()
        if step%15==0 or step==TRAIN_STEPS-1:
            record.append({'model':variant,'step':step,'loss':float(loss.detach()),
                           'N_known':int((m>0).sum())})
            if step%45==0: print('V26 TRAIN',variant,step,'loss',round(float(loss.detach()),5),flush=True)
    model.eval()
    ck=out/f'{variant}_model.pt'
    torch.save({'model':variant,'steps':TRAIN_STEPS,'seed':SEED,
                'state_dict':model.state_dict()},ck)
    return model,record,sha_file(ck)

def calibrate(model,cal_records):
    hist=[]
    logits={}
    for rec in cal_records:
        prob=dense_probs(model,load_original_record_image(rec))
        mask=rec['positive']|rec['negative']
        logits[rec['section']]=(prob[mask],rec['positive'][mask])
    if len(logits)<3:raise RuntimeError('Insufficient crowd CAL sections')
    for threshold in THRESHOLDS:
        tp=fp=fn=0
        for prediction,truth in logits.values():
            chosen=prediction>=threshold
            tp+=int(np.sum(chosen & truth))
            fp+=int(np.sum(chosen & ~truth))
            fn+=int(np.sum(~chosen & truth))
        precision=tp/(tp+fp) if (tp+fp) else 0.
        recall=tp/(tp+fn) if (tp+fn) else 0.
        f1=2*precision*recall/(precision+recall) if (precision+recall) else 0.
        hist.append({'threshold':threshold,'precision':precision,
                     'recall':recall,'f1':f1,'tp':tp,'fp':fp,'fn':fn})
    acceptable=[row for row in hist if row['precision']>=.60]
    choice=max(acceptable or hist,key=lambda row:(row['f1'],row['threshold']))
    return float(choice['threshold']),hist

def load_original_record_image(record):
    # record.image is already normalized: invert without access to expert labels.
    return np.clip(record['image']*.25+.5,0.,1.).astype(np.float32)

def infer_variants(prob,threshold):
    # Both methods act on the same predicted probability map.
    raw=run_pipeline(prob,threshold=threshold,strategy='context')
    link=run_pipeline(prob,threshold=threshold,strategy='context')
    region=run_regions(prob,threshold,mode='region_hybrid')
    return {'raw':raw,'context':link,'region_hybrid':region}

def experiment(images_path,annotations_path,out,max_expert=40):
    torch.set_num_threads(2)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(images_path) as zi,zipfile.ZipFile(annotations_path) as za:
        images,ann=zip_index(zi,za)
        training=crowd_records(zi,za,images,ann,TRAIN_IDS)
        calibration=crowd_records(zi,za,images,ann,CAL_IDS)
        if len(training)<20 or len(calibration)<4:raise RuntimeError('Too few crowd masks')
        if set(r['section'] for r in training)&set(r['section'] for r in calibration):
            raise RuntimeError('Crowd TRAIN and CAL overlap')
        hgb,hgb_tau,hgb_receipt=_train(zi,za,images,ann,out=out)
        models={}
        training_log=[]
        train_sha={}
        tau={'hgb':hgb_tau}
        curves={'hgb':hgb_receipt['calibration_curves']}
        for variant in ('conv','az'):
            net,fit_log,sha=fit_model(variant,training,out)
            models[variant]=net;training_log.extend(fit_log);train_sha[variant]=sha
            tau[variant],curves[variant]=calibrate(net,calibration)
            print('V26 CAL',variant,'tau',tau[variant],
                  'last_train_loss',fit_log[-1]['loss'],flush=True)
        expert_ids=sorted(sid for sid in set(images)&set(ann)
                          if 'expert' in ann[sid])[:max_expert]
        if len(expert_ids)<10:raise RuntimeError('Too few CRACKS expert sections')
        if set(expert_ids)&set(r['section'] for r in training+calibration):
            raise RuntimeError('Crowd train/cal leaked expert section')
        rows=[]
        for idx,sid in enumerate(expert_ids):
            image=load_image(zi,images[sid])
            feats=anisotropic_features(image)
            prob_hgb=hgb.predict_proba(feats.reshape(-1,feats.shape[-1]))[:,1].reshape(image.shape).astype(np.float32)
            inferred={}
            for variant in METHODS:
                prob=prob_hgb if variant=='hgb' else dense_probs(models[variant],image)
                inferred[variant]=infer_variants(prob,tau[variant])
                if idx<3:
                    np.savez_compressed(out/f'{variant}_section_{sid:03d}.npz',
                       probability=prob.astype(np.float16),
                       raw_mask=inferred[variant]['raw']['raw_mask'].astype(np.uint8),
                       hybrid_mask=inferred[variant]['region_hybrid']['completed_mask'].astype(np.uint8))
            # Strict delayed expert-mask access; no infer_variant accepts expert.
            expert=load_mask(za,ann[sid]['expert'])
            row={'section':sid,'split':'V26_POSTHOC_EXPLORATORY'}
            for variant in METHODS:
                for policy in POLICIES:
                    key=f'{variant}_{policy}'
                    result=inferred[variant][policy]
                    mask=result['raw_mask'] if policy=='raw' else result['completed_mask']
                    metrics=evaluate_mask(mask,expert)
                    for name,value in metrics.items():row[f'{key}_{name}']=value
                    row[f'{key}_accepted_links']=0 if policy=='raw' else len(result['selected'])
                    if policy=='raw' or policy=='region_hybrid':
                        audit=candidate_reachability(result,expert)
                        for name,value in audit.items():row[f'{key}_candidate_{name}']=value
            rows.append(row)
            if idx<3:
                for variant in METHODS:
                    _overlay(image,inferred[variant]['raw']['raw_mask'],
                             inferred[variant]['region_hybrid']['completed_mask'],
                             out/f'{variant}_example_{sid:03d}.png')
            print('V26 REAL',sid,'links',[(v,rows[-1][f'{v}_region_hybrid_connected_gold_anchor_pairs']) for v in METHODS],
                  flush=True)
    with (out/'train_loss.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=('model','step','loss','N_known'))
        w.writeheader();w.writerows(training_log)
    csv_path=out/'per_section_v26.csv'
    with csv_path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    aggregate={}
    for variant in METHODS:
        aggregate[variant]={}
        for policy in POLICIES:
            prefix=f'{variant}_{policy}_'
            nums=lambda key:sum(int(r[prefix+key]) for r in rows)
            tp=nums('expert_blue_recalled_tol2')
            npos=nums('N_expert_blue')
            fp=nums('explicit_orange_false_pixels')
            nneg=nums('N_explicit_orange')
            total=nums('eligible_paths')
            connected=nums('connected_gold_anchor_pairs')
            detail={
                'N_sections':len(rows),'N_positive_pixels':npos,
                'N_explicit_negative_pixels':nneg,
                'N_eligible_expert_paths':total,
                'blue_recall_tol2':tp/npos if npos else None,
                'false_orange_fraction':fp/nneg if nneg else None,
                'connected_gold_anchor_pairs':connected,
                'connectivity_rate':connected/total if total else None,
                'accepted_links':nums('accepted_links')
            }
            if policy in ('raw','region_hybrid'):
                detail['direct_candidate_upper_bound']=sum(int(r[prefix+'candidate_direct_candidate_could_connect']) for r in rows)
                detail['N_gold_disconnected_with_both_anchors']=sum(int(r[prefix+'candidate_unconnected_but_both_visible']) for r in rows)
            aggregate[variant][policy]=detail
    paired={}
    for target in ('region_hybrid','raw'):
        for comparator in ('conv','hgb'):
            paired[f'az_minus_{comparator}_{target}']=_bootstrap(
                rows,'az_'+target,comparator+'_'+target)
    report={
        'status':'EXECUTED_EXPLORATORY',
        'protocol':'CRACKS_V26_TRUE_REPOSITORY_AZCONV_VS_STANDARD_CONV',
        'source':'CRACKS v2 doi:10.5281/zenodo.13926822',
        'original_historical_ANZA_checkpoint':False,
        'actual_retrained_repository_AZConv2d':True,
        'original_historical_LIRA_model_checkpoint':False,
        'uses_physical_fault_identity_pairs':False,
        'source_is_same_previous_opened_F3_expert_volume':True,
        'trained_only_on_nonexpert_crowd':True,
        'expert_masks_read_only_after_all_model_inferences':True,
        'train_sections':[r['section'] for r in training],
        'calibration_sections':[r['section'] for r in calibration],
        'N_train_steps_per_neural_model':TRAIN_STEPS,
        'neural_seed':SEED,'thresholds':tau,'crowd_calibration':curves,
        'checkpoint_sha256':train_sha,
        'hgb_baseline_receipt':hgb_receipt,
        'methods':aggregate,
        'paired_section_cluster_bootstrap':paired,
        'per_section_csv_sha256':sha_file(csv_path),
        'train_loss_csv_sha256':sha_file(out/'train_loss.csv'),
        'images_zip_sha256':sha_file(images_path),
        'annotations_zip_sha256':sha_file(annotations_path),
        'scientific_interpretation':'Post-V23/V25 exploratory single fit of original AZConv2d operator, not original ANZA checkpoint; same real F3 expert masks, semantic not physical fault-instance labels. No end-to-end natural link/FPR claim.'
    }
    (out/'report_v26.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'status':report['status'],'thresholds':tau,
                      'result':aggregate,'paired':paired},ensure_ascii=False,indent=2))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('results/cracks_real_neural_v26'))
    p.add_argument('--max-expert',type=int,default=40)
    a=p.parse_args()
    experiment(a.images,a.annotations,a.out,a.max_expert)

if __name__=='__main__':
    main()
