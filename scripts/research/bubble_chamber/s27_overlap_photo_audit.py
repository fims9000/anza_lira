#!/usr/bin/env python3
"""S27 real-H50Z NO-LABEL seam audit: fixed original ANZA/U-Net checkpoints.

Changes only 64x64 crop blending (stride64 -> stride32). No threshold is
selected on H50Z-native pixels. The frozen thresholds were calibrated on
photo005 SYNTHETIC marks (see S26). Real candidate IOU != track accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s27_overlap_inference import overlap_predict,tile_boundary_jump,disagreement_stats
from scripts.research.bubble_chamber.s26_visual_inspection import model_for_checkpoint,tiled_predict,overlay
from scripts.research.bubble_chamber.s26_two_photo_baselines import classical_score

SCHEMA='s27_h50z_no_label_overlap_robustness_v1'


def sha256(path:Path)->str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_photo_rows(rows: list[dict],crop_dir:Path)->list[dict]:
    if not rows:raise ValueError('No source-photo audited ROIs')
    seen=set();valid=[]
    for r in rows:
        id=r['crop_id'];p=Path(crop_dir)/(id+'.png')
        if id in seen or id.rsplit('_',1)[-1]=='':raise ValueError('duplicate/invalid crop ID')
        seen.add(id)
        if not p.is_file() or sha256(p)!=r['source_crop_sha256']:
            raise ValueError(f'source image changed or missing: {id}')
        if r['source_photo'] not in ('005','006') or r['source_photo'] not in id:
            raise ValueError('source photo ID mismatch')
        valid.append(r)
    return valid


def model_thresholds(report: dict,guided_report:dict)->dict[str,float]:
    if report.get('input',{}).get('train_and_validation_photo')!='005' or report['input']['heldout_test_photo']!='006':
        raise ValueError('non-photo005 synthetic calibration')
    t={
       'unet':float(report['methods']['unet_local']['validation']['threshold']),
       'azunet':float(report['methods']['azunet_local']['validation']['threshold']),
       'guided_az':float(guided_report['val_005_calibration']['threshold']),
       'sato':float(report['methods']['sato']['validation']['threshold'])}
    if any(not 0<x<=1.000001 for x in t.values()):raise ValueError('invalid frozen threshold')
    return t


def prepare_models(checkpoints:dict[str,Path])->dict:
    models={}
    for name,p in checkpoints.items():
        models[name]=model_for_checkpoint(p)
    return models


def run_one(row:dict,path:Path,models:dict,thresholds:dict,photo_jitter:bool=True):
    im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if im is None or im.shape!=(512,512):raise ValueError('must be original-resolution 512x512 grayscale')
    records={};scores={}
    for name,(model,preproc) in models.items():
        original=tiled_predict(im,model,preproc)
        smooth=overlap_predict(im,model,preproc,stride=32,batch_size=32)
        t=thresholds[name]
        delta=disagreement_stats(original,smooth,t)
        before=tile_boundary_jump(original)
        after=tile_boundary_jump(smooth)
        records[name]={'no_overlap':before,'with_overlap':after,'candidate_difference':delta,
                'mean_boundary_jump_reduction_fraction':float(1-after['mean_boundary_jump']/max(before['mean_boundary_jump'],1e-6)),
                'genuine_track_pixel_truth':False}
        if photo_jitter and name in ('unet','azunet'):
            pert=[]
            base=smooth>=t
            for d in (-25,25):
                # Fixed offset: target stays same real source scene. Values are
                # allowed to clip, a realistic camera/intensity stress test.
                transformed=np.clip(im.astype('int16')+d,0,255).astype('uint8')
                shifted=overlap_predict(transformed,model,preproc,stride=32,batch_size=32)
                candidate=shifted>=t
                union=int((base|candidate).sum());inter=int((base&candidate).sum())
                pert.append({'brightness_offset_added':d,'change_fraction':float(np.mean(base^candidate)),
                             'candidate_mask_iou_NOT_GROUND_TRUTH':float(inter/union) if union else None,
                             'selected_fraction_UNVERIFIED':float(candidate.mean())})
            records[name]['brightness_offset_robustness']=pert
        scores[name]=smooth
    sato=classical_score(im,'sato')
    masks={k:scores[k]>=thresholds[k] for k in scores}
    masks['sato']=sato>=thresholds['sato']
    stable=masks['unet']&masks['guided_az']
    disag=masks['unet']^masks['guided_az']
    combined={'unverified_consensus_fraction':float(stable.mean()),
              'unverified_neural_disagreement_fraction':float(disag.mean()),
              'unverified_sato_fraction':float(masks['sato'].mean()),
              'unverified_consensus_sato_overlap_fraction':float(np.mean(stable&masks['sato'])),
              'human_certified_trace_pixels':0,'human_certified_relation_labels':0,
              'inference_model_only_review_status':True}
    return {'source':{'crop_id':row['crop_id'],'photo_id':row['source_photo'],
                      'review_stratum':row.get('stratum','unknown'),
                      'source_crop_sha256':sha256(path)},
            'original_score_to_overlap':records,'mixed_model_candidates_NO_TRUTH':combined},(im,scores,masks)


def summary(rows:list[dict],thresholds:dict,checkpoints:dict)->dict:
    if not rows:raise ValueError('no results')
    by_photo={}
    for photo in ('005','006'):
        q=[r for r in rows if r['source']['photo_id']==photo]
        if not q:raise ValueError('both photos required')
        by_photo[photo]={'crops':len(q),'strata':{st:sum(r['source']['review_stratum']==st for r in q)
            for st in sorted(set(r['source']['review_stratum'] for r in q))}}
        for model in ('unet','azunet','guided_az'):
            before=np.array([r['original_score_to_overlap'][model]['no_overlap']['mean_boundary_jump'] for r in q])
            after=np.array([r['original_score_to_overlap'][model]['with_overlap']['mean_boundary_jump'] for r in q])
            by_photo[photo][model]={'mean_no_overlap_boundary_jump':float(before.mean()),
                                   'mean_overlap_boundary_jump':float(after.mean()),
                                   'pooled_boundary_jump_reduction_pct':float(100*(1-after.sum()/max(before.sum(),1e-8))),
                                   'cropped_regions_with_reduced_boundary_jump':int((after<before).sum()),
                                   'median_candidate_fraction_changed':float(np.median([r['original_score_to_overlap'][model]['candidate_difference']['candidate_pixel_fraction_changed'] for r in q]))}
    return {'schema':SCHEMA,'source_photos':by_photo,'source_crops':len(rows),
            'models':['unet','azunet','guided_az'],'frozen_thresholds_calibrated_only_on_SYNTHETIC_photo005_marks':thresholds,
            'source_model_checkpoint_sha256':{k:sha256(v) for k,v in checkpoints.items()},
            'overlap_parameters':{'tile':64,'old_stride':64,'new_stride':32,'blend':'Hann with 0.05 positive floor'},
            'conclusions_are_only_seam_robustness_NOT_NATIVE_PHYSICAL_ACCURACY':True,
            'human_pixel_track_masks':0,'real_particle_joins_approved':0,
            'independent_physical_run':False,
            'source_used_for_model_selection':'photo005 synthetic marks only; photo006 was already used for S26 exploratory comparisons'}


def preview(im,scores,masks,caption):
    out=[]
    original=cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    out.append(original)
    for name in ('unet','azunet','guided_az'):
        out.append(overlay(im,masks[name],[60,220,105]))
    out.append(overlay(im,masks['unet']^masks['guided_az'],[230,115,60]))
    out.append(overlay(im,masks['unet']&masks['guided_az'],[50,220,100]))
    n=np.hstack([cv2.resize(z,(256,256),interpolation=cv2.INTER_AREA) for z in out])
    cv2.putText(n,caption[:62],(6,20),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
    return n


def run(args):
    torch.set_num_threads(2)
    roibundle=json.loads(args.roi_manifest.read_text(encoding='utf-8'))
    if roibundle.get('schema')!='s26b_full_frame_provenance_bundle_v1':
        raise ValueError('invalid real photo source bundle')
    rows=validate_photo_rows(roibundle['review_crops'],args.crop_dir)
    report=json.loads(args.baseline_report.read_text(encoding='utf-8'))
    guided_report=json.loads(args.guided_report.read_text(encoding='utf-8'))
    thresholds=model_thresholds(report,guided_report)
    checkpoints={'unet':args.unet,'azunet':args.azunet,'guided_az':args.guided_az}
    models=prepare_models(checkpoints)
    args.out.mkdir(parents=True,exist_ok=True)
    cache=args.out/'audited_crops';cache.mkdir(exist_ok=True)
    audit=[];images=[]
    for i,row in enumerate(rows):
        if args.photo and row['source_photo']!=args.photo:continue
        p=args.crop_dir/(row['crop_id']+'.png')
        receipt=cache/(row['crop_id']+'.json')
        if receipt.exists() and args.use_cache:
            d=json.loads(receipt.read_text())
            if d['source']['source_crop_sha256']!=sha256(p):
                raise ValueError('cached provenance changed')
            audit.append(d);continue
        d,(im,scores,masks)=run_one(row,p,models,thresholds,photo_jitter=not args.skip_brightness)
        audit.append(d)
        receipt.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
        # A small human-viewable subset is enough; never hide large original
        # pixels in a generated/repainted inference visualization.
        if row['stratum'] in ('start_anchor_nearby','fiducial_control'):
            image=preview(im,scores,masks,row['crop_id'])
            cv2.imwrite(str(args.out/(row['crop_id']+'_overlap_review.jpg')),cv2.cvtColor(image,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,88])
        print('S27',row['crop_id'],{k:round(d['original_score_to_overlap'][k]['mean_boundary_jump_reduction_fraction'],3) for k in models},flush=True)
    if args.photo:
        print('Subset',args.photo,len(audit));return
    byid={r['source']['crop_id']:r for r in audit}
    if len(byid)!=len(rows):raise ValueError('missing audited real ROI')
    audit=[byid[r['crop_id']] for r in rows]
    stat=summary(audit,thresholds,checkpoints)
    (args.out/'S27_real_photo_robustness_summary.json').write_text(json.dumps(stat,ensure_ascii=False,indent=2)+'\n')
    (args.out/'S27_all_crop_metrics.json').write_text(json.dumps({'schema':SCHEMA,'per_crop':audit},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(stat['source_photos'],ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser()
    for name in ('roi_manifest','crop_dir','baseline_report','guided_report','unet','azunet','guided_az','out'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--photo',choices=['005','006'])
    p.add_argument('--skip-brightness',action='store_true')
    p.add_argument('--use-cache',action='store_true')
    a=p.parse_args();run(a)

if __name__=='__main__':main()
