#!/usr/bin/env python3
"""End-to-end *exploratory* real-CRACKS image-only fault linkage V23.

Learn a lightweight anisotropic-feature segmenter from nonexpert partial labels
on spatially separated sections, calibrate on different nonexpert sections,
then detect all fragments/ports/candidates/NO_LINK/paths from images alone.
Expert masks are opened ONLY for post-prediction evaluation. This is not the
original neural ANZA checkpoint and cannot prove geological instance identity.
"""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import joblib
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.spatial import cKDTree

from scripts.cracks_real_auto_core_v23 import anisotropic_features,run_pipeline,extract_traces

BLUE=(31,119,180)
ORANGE=(255,127,14)
GREEN=(44,160,44)
WHITE=(255,255,255)
SEED=20261113
TRAIN_IDS=tuple(range(340,393,2))
CAL_IDS=tuple(range(311,329,2))
CANDIDATE_THRESHOLDS=(0.35,0.45,0.55,0.65,0.75,0.85)
POLICIES=('raw','nearest','geometry','context')


def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda:stream.read(1048576),b''):h.update(b)
    return h.hexdigest()


def zip_index(images:zipfile.ZipFile,annotations:zipfile.ZipFile):
    r=re.compile(r'section_(\d+)\.png$',re.I)
    imgs={}
    masks=defaultdict(dict)
    for name in images.namelist():
        m=r.search(name)
        if m:
            sid=int(m.group(1))
            if sid in imgs:raise ValueError('duplicate image section '+str(sid))
            imgs[sid]=name
    for name in annotations.namelist():
        m=r.search(name)
        if not m:continue
        parts=name.strip('/').split('/')
        if len(parts)!=3 or parts[0].lower()!='fault segmentations':continue
        annotator=parts[1].lower();sid=int(m.group(1))
        if annotator in masks[sid]:raise ValueError('duplicate mask')
        masks[sid][annotator]=name
    return imgs,dict(masks)


def load_image(archive,path):
    with Image.open(io.BytesIO(archive.read(path))) as img:
        return np.asarray(img.convert('L'),dtype=np.float32)/255.


def load_mask(archive,path):
    with Image.open(io.BytesIO(archive.read(path))) as img:
        return np.asarray(img.convert('RGB'),dtype=np.uint8)


def consensus_training_mask(archive,section_masks):
    """Only explicit certain positive/negative votes, never uncertain/white."""
    names=sorted(x for x in section_masks if x!='expert')
    if len(names)<2:return None
    positive=negative=None
    for name in names:
        rgb=load_mask(archive,section_masks[name])
        if positive is None:
            shape=rgb.shape[:2];positive=np.zeros(shape,np.int16)
            negative=np.zeros(shape,np.int16)
        if rgb.shape[:2]!=shape:raise ValueError('inconsistent mask shapes')
        positive+=np.all(rgb==BLUE,axis=2)
        negative+=np.all(rgb==ORANGE,axis=2)
    pos=(positive>=3)&(positive>=2*negative)
    neg=(negative>=3)&(positive==0)
    if np.any(pos&neg):raise AssertionError('overlapping targets')
    return pos,neg


def sample_known(image,positive,negative,rng,max_pos=260,max_neg=360):
    feats=anisotropic_features(image)
    pos=np.flatnonzero(positive);neg=np.flatnonzero(negative)
    if len(pos)<15 or len(neg)<30:return None
    ixpos=rng.choice(pos,min(max_pos,len(pos)),replace=False)
    ixneg=rng.choice(neg,min(max_neg,len(neg)),replace=False)
    ids=np.concatenate([ixpos,ixneg]);labels=np.r_[np.ones(len(ixpos)),np.zeros(len(ixneg))]
    return feats.reshape(-1,feats.shape[-1])[ids],labels.astype(int)


def _train(zi,za,images,ann,*,out):
    rng=np.random.default_rng(SEED)
    blocks=[];labels=[];train_sections=[]
    for sid in TRAIN_IDS:
        if sid not in images or sid not in ann or 'expert' in ann[sid]:continue
        target=consensus_training_mask(za,ann[sid])
        if target is None:continue
        p,n=target
        pair=sample_known(load_image(zi,images[sid]),p,n,rng)
        if pair is None:continue
        xx,yy=pair;blocks.append(xx);labels.append(yy)
        train_sections.append(sid)
    if len(train_sections)<5:raise RuntimeError('Not enough crowd-only training sections')
    features=np.concatenate(blocks);truth=np.concatenate(labels)
    if len(np.unique(truth))!=2:raise RuntimeError('Training has one class')
    model=HistGradientBoostingClassifier(max_iter=85,max_leaf_nodes=15,
          learning_rate=.085,l2_regularization=5.,max_bins=127,
          random_state=SEED)
    model.fit(features,truth)
    model_path=out/'crowd_only_anisotropic_hgb.joblib'
    joblib.dump(model,model_path)
    cal=[]
    for sid in CAL_IDS:
        if sid not in images or sid not in ann or 'expert' in ann[sid]:continue
        target=consensus_training_mask(za,ann[sid])
        if target is None:continue
        p,n=target
        pair=sample_known(load_image(zi,images[sid]),p,n,rng,max_pos=1000,max_neg=1200)
        if pair is None:continue
        xx,yy=pair;prob=model.predict_proba(xx)[:,1]
        cal.append((sid,yy,prob))
    if len(cal)<3:raise RuntimeError('Not enough crowd-only calibration sections')
    # Precision-gated to avoid extreme overprediction; all tuning only on crowd.
    candidates=[]
    for threshold in CANDIDATE_THRESHOLDS:
        tp=fp=fn=0
        for _,yy,prob in cal:
            pred=prob>=threshold
            tp+=int(np.sum(pred & (yy==1)))
            fp+=int(np.sum(pred & (yy==0)))
            fn+=int(np.sum(~pred & (yy==1)))
        prec=tp/max(tp+fp,1)
        rec=tp/max(tp+fn,1)
        f1=2*prec*rec/max(prec+rec,1e-12)
        candidates.append({'threshold':threshold,'precision':prec,'recall':rec,
                           'f1':f1,'tp':tp,'fp':fp,'fn':fn})
    eligible=[x for x in candidates if x['precision']>=.60]
    best=max(eligible if eligible else candidates,key=lambda x:(x['f1'],x['threshold']))
    train_report={'segmentation':'new anisotropic-feature HGB surrogate, NOT legacy ANZA neural model',
         'training_source':'crowd certain BLUE vs ORANGE, ignore GREEN/WHITE, no expert',
         'train_sections':train_sections,'calibration_sections':[x[0] for x in cal],
         'N_pixel_training':len(truth),'N_pos':int(truth.sum()),
         'model_sha256':sha_file(model_path),'calibration_curves':candidates,
         'threshold_rule':'max crowd CAL F1 with precision >=0.60 if feasible',
         'chosen_threshold':best['threshold']}
    return model,best['threshold'],train_report


def _nearby_label_component(labels,point,radius=2):
    h,w=labels.shape;y,x=point
    ys=range(max(0,y-radius),min(h,y+radius+1))
    xs=range(max(0,x-radius),min(w,x+radius+1))
    candidates=[(mathdist((y,x),(yy,xx)),int(labels[yy,xx]))
        for yy in ys for xx in xs if labels[yy,xx]!=0]
    return min(candidates)[1] if candidates else 0


def mathdist(a,b):return float(np.linalg.norm(np.asarray(a,float)-np.asarray(b,float)))


def expert_paths(blue):
    """Independent annotation-only evaluation; no generator-visible anchors."""
    _,traces,_=extract_traces(blue.astype(np.float32),threshold=.5,
                             min_pixels=22,border_margin=6)
    return traces


def topology_measure(mask,expert):
    """Evaluate connected predicted components near two preselected GT points."""
    paths=expert_paths(expert)
    components,_=ndi.label(mask,structure=np.ones((3,3),np.uint8))
    tot=observed=connected=0
    for tr in paths:
        n=len(tr.points)
        if n<36:continue
        # Gold defines *evaluation endpoints* only, not inference candidates.
        a=tuple(tr.points[max(5,n//5)]);b=tuple(tr.points[min(n-6,4*n//5)])
        tot+=1
        ca=_nearby_label_component(components,a)
        cb=_nearby_label_component(components,b)
        if ca and cb:
            observed+=1
            connected+=int(ca==cb)
    return {'eligible_paths':tot,'both_gold_anchors_detected':observed,
            'connected_gold_anchor_pairs':connected}


def evaluate_mask(pred_mask,expert_rgb):
    blue=np.all(expert_rgb==BLUE,axis=2)
    orange=np.all(expert_rgb==ORANGE,axis=2)
    # Some expert masks contain only one explicitly labeled class. Preserve
    # observed class denominators without inventing absent class labels.
    matched=ndi.binary_dilation(pred_mask,iterations=2)
    tp=int(np.sum(matched & blue))
    false=int(np.sum(pred_mask & orange))
    predicted=int(np.sum(pred_mask & (blue|orange)))
    return {'expert_blue_recalled_tol2':tp,'N_expert_blue':int(blue.sum()),
        'explicit_orange_false_pixels':false,'N_explicit_orange':int(orange.sum()),
        'predicted_explicit_pixels':predicted,
        **topology_measure(pred_mask,blue)}


def _overlay(image,base,full,filename):
    img=Image.fromarray((np.clip(image,0,1)*255).astype(np.uint8)).convert('RGB')
    arr=np.asarray(img).copy()
    novel=full & ~base
    arr[base]=np.clip(.6*arr[base]+np.array([30,185,250])*.4,0,255).astype(np.uint8)
    arr[novel]=np.array([255,130,55],np.uint8)
    Image.fromarray(arr).save(filename,optimize=True)


def _bootstrap(rows,policy_a,policy_b,n_iter=3000):
    """Section-resampling; compare connected GT-pair frequency, all eligible."""
    if not rows:return None
    ids=sorted({r['section'] for r in rows})
    if len(ids)<2:return None
    grouped={s:[r for r in rows if r['section']==s] for s in ids}
    numer=[];denom=[]
    for sid in ids:
        a=grouped[sid]
        assert len(a)==1
        numer.append(a[0][policy_a+'_connected_gold_anchor_pairs']-
                     a[0][policy_b+'_connected_gold_anchor_pairs'])
        denom.append(a[0][policy_a+'_eligible_paths'])
    num=np.asarray(numer,float);den=np.asarray(denom,float)
    if den.sum()==0:return None
    rng=np.random.default_rng(SEED)
    sample=rng.integers(0,len(ids),size=(n_iter,len(ids)))
    n=num[sample].sum(axis=1)
    d=den[sample].sum(axis=1)
    valid=d>0
    if not valid.any():return None
    boot=n[valid]/d[valid]
    return {'difference':float(num.sum()/den.sum()),
            'interval_95':list(map(float,np.quantile(boot,[.025,.975]))),
            'groups':len(ids)}


def run(images_path:Path,annotations_path:Path,out:Path,*,max_expert=40):
    out.mkdir(parents=True,exist_ok=True)
    if images_path.stat().st_size<10_000_000 or annotations_path.stat().st_size<1_000_000:
        raise RuntimeError('CRACKS archives unexpectedly small')
    with zipfile.ZipFile(images_path) as zi,zipfile.ZipFile(annotations_path) as za:
        images,ann=zip_index(zi,za)
        expert_ids=sorted(sid for sid in set(images)&set(ann) if 'expert' in ann[sid])[:max_expert]
        if len(expert_ids)<10:raise RuntimeError('Need at least 10 expert sections for real evaluation')
        model,tau,training=_train(zi,za,images,ann,out=out)
        rows=[]
        for idx,sid in enumerate(expert_ids):
            image=load_image(zi,images[sid])
            feats=anisotropic_features(image)
            prob=model.predict_proba(feats.reshape(-1,feats.shape[-1]))[:,1].reshape(image.shape).astype(np.float32)
            # Inference explicitly precomputed with no access to the expert mask.
            results={}
            for name in POLICIES:
                strategy='context' if name=='raw' else name
                result=run_pipeline(prob,threshold=tau,strategy=strategy)
                results[name]=result
            # Delayed expert read AFTER all four image-only choices.
            gold=load_mask(za,ann[sid]['expert'])
            row={'section':sid,'split':'all_exploratory_expert',
                 'num_predicted_ports':len(results['raw']['ports']),
                 'num_generated_candidates':len(results['raw']['candidates'])}
            for name in POLICIES:
                res=results[name]
                visible=res['raw_mask'] if name=='raw' else res['completed_mask']
                m=evaluate_mask(visible,gold)
                row[name+'_accepted_links']=len(res['selected'])
                row[name+'_no_link_ports']=res['none_count']
                for k,v in m.items():row[name+'_'+k]=v
            rows.append(row)
            if idx<4:
                _overlay(image,results['raw']['raw_mask'],
                        results['context']['completed_mask'],
                        out/f'example_section_{sid:03}.png')
                np.savez_compressed(out/f'image_only_section_{sid:03}.npz',
                    probability=prob.astype(np.float16),
                    input_mask=results['raw']['raw_mask'].astype(np.uint8),
                    completed_mask=results['context']['completed_mask'].astype(np.uint8))
            print('EXPERT EVAL',sid,'paths',row['raw_eligible_paths'],
                  'raw',row['raw_connected_gold_anchor_pairs'],
                  'context',row['context_connected_gold_anchor_pairs'],
                  'ports',row['num_predicted_ports'],flush=True)
    outcsv=out/'per_section.csv'
    with outcsv.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    aggregate={}
    for name in POLICIES:
        tp=sum(r[name+'_expert_blue_recalled_tol2'] for r in rows)
        npos=sum(r[name+'_N_expert_blue'] for r in rows)
        fp=sum(r[name+'_explicit_orange_false_pixels'] for r in rows)
        nneg=sum(r[name+'_N_explicit_orange'] for r in rows)
        paths=sum(r[name+'_eligible_paths'] for r in rows)
        both=sum(r[name+'_both_gold_anchors_detected'] for r in rows)
        connected=sum(r[name+'_connected_gold_anchor_pairs'] for r in rows)
        aggregate[name]={'recall_blue_tol2':tp/npos if npos else None,
           'false_pixel_fraction_orange':fp/nneg if nneg else None,
           'N_expert_blue':npos,'N_explicit_orange':nneg,
           'N_sections_with_blue':sum(r[name+'_N_expert_blue']>0 for r in rows),
           'N_sections_with_explicit_orange':sum(r[name+'_N_explicit_orange']>0 for r in rows),
           'eligible_annotated_paths':paths,'both_anchors_present':both,
           'connected_annotated_pairs':connected,
           'connected_fraction_all_paths':connected/max(paths,1),
           'accepted_links':sum(r[name+'_accepted_links'] for r in rows),
           'mean_predicted_ports':float(np.mean([r['num_predicted_ports'] for r in rows]))}
    report={
      'protocol':'CRACKS_V23_FIRST_FULLY_AUTOMATIC_IMAGE_TO_LINK_TO_PATH_PROTOTYPE',
      'status':'EXECUTED_EXPLORATORY',
      'source':'CRACKS v2 DOI 10.5281/zenodo.13926822',
      'images_zip_sha256':sha_file(images_path),
      'annotations_zip_sha256':sha_file(annotations_path),
      'segmentation_model':training,
      'N_evaluated_expert_sections':len(rows),
      'evaluation_sections':[r['section'] for r in rows],
      'evaluation_sees_truth_only_after_inference':True,
      'expert_masks_used_for_training_or_calibration':False,
      'missing_class_denominators_are_not_fabricated':True,
      'predicted_endpoints_and_candidates':True,
      'links_and_paths_generated_without_gt':True,
      'has_explicit_NO_LINK':True,
      'image_only_real_cracks':True,
      'not_original_ANZA_checkpoint':True,
      'not_independent_geological_instance_link_truth':True,
      'one_F3_volume_and_all_experts_previously_explored':True,
      'methods':aggregate,
      'context_vs_raw_cluster_bootstrap':_bootstrap(rows,'context','raw'),
      'context_vs_nearest_cluster_bootstrap':_bootstrap(rows,'context','nearest'),
      'per_section_csv_sha256':sha_file(outcsv),
      'claim':'This verifies implementation and exploratory measures, not natural geological link accuracy or superiority over pretrained ANZA/LIRA.'}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':report['status'],'sections':len(rows),'metrics':aggregate},indent=2))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--images',required=True,type=Path)
    p.add_argument('--annotations',required=True,type=Path)
    p.add_argument('--out',type=Path,default=Path('results/cracks_real_autonomous_v23'))
    p.add_argument('--max-expert',type=int,default=40)
    a=p.parse_args()
    run(a.images,a.annotations,a.out,max_expert=a.max_expert)

if __name__=='__main__':main()
