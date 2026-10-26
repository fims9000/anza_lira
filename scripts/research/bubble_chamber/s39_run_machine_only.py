#!/usr/bin/env python3
"""Run source-locked H50Z machine-only pseudo-annotation. PRIVATE output only.

No physicist review is required; no pseudo label is promoted to native GT.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json,sys,zipfile
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s39_machine_auto import annotate_tile,SCHEMA
from bubble_chamber.s33_photometric_domain import image_input
from scripts.research.bubble_chamber.s35_benchmark_fragment_links import load_model

EXPECTED_SOURCE_SHA='903cfae7e8742bab203ee12aecfa454b75a71454dbabaee630982557a84b13d7'


def _sha_bytes(b:bytes)->str:return sha256(b).hexdigest()


def predict_cross_photo(gray:np.ndarray,model,*,step=56,batch=32)->np.ndarray:
    """Tiled 64px original model inputs, overlap-weighted, no architecture edits."""
    if gray.ndim!=2 or gray.shape[0]!=gray.shape[1] or gray.shape[0]<128 or gray.dtype!=np.uint8 or not 16<=step<=64:
        raise ValueError('Unexpected source tiles / overlap step')
    size=gray.shape[0]
    xy=sorted(set(list(range(0,size-64+1,step))+[size-64]))
    patches=[image_input(gray[y:y+64,x:x+64],'quantile') for y in xy for x in xy]
    weighted=np.zeros(gray.shape,dtype='float64');coverage=np.zeros_like(weighted)
    window=np.clip(np.hanning(64),.08,1)
    weights=np.outer(window,window)
    locations=[(y,x) for y in xy for x in xy]
    with torch.no_grad():
        model.eval()
        for i in range(0,len(locations),batch):
            x=torch.from_numpy(np.stack(patches[i:i+batch]).astype('float32')[:,None])
            y=torch.sigmoid(model(x)).cpu().numpy()[:,0]
            for p,(oy,ox) in zip(y,locations[i:i+batch]):
                weighted[oy:oy+64,ox:ox+64]+=p*weights
                coverage[oy:oy+64,ox:ox+64]+=weights
    if np.any(coverage<=0):raise AssertionError('Uncovered tiled model pixel')
    return np.clip(weighted/coverage,0,1).astype('float32')


def overlay(gray:np.ndarray,original:list,proposals:list)->np.ndarray:
    # Blue/new, yellow/old, green/multisource support: all are hypotheses.
    image=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    bgr={'S31_EXISTING_MACHINE_DRAFT':(70,185,240),'S39_MULTISCALE_RIDGE_PROPOSAL':(245,200,60)}
    for c in sorted(proposals,key=lambda r:r['review_priority_NOT_PHYSICS_PROBABILITY']):
        coords=np.rint(np.asarray(c['points_xy'])).astype('int32')
        color=bgr[c['source']]
        if (c['raw_center_minus_flank_8bit']>=8 and
            c['ridge_scale_consensus_fraction']>=.34 and
            (c['S31_other_candidate_overlap_fraction']>=.35 or
             (c['AZ_cross_photo_center_vs_flank_UNCALIBRATED'] or 0.)>=.018)):
            color=(30,205,75)
        cv2.polylines(image,[coords],False,color,2,lineType=cv2.LINE_AA)
    return image


def _make_montage(tiles,chosen:tuple,output:Path):
    from PIL import Image,ImageDraw
    rowh=350;colw=345
    sheet=Image.new('RGB',(colw*3,rowh*len(chosen)),(236,236,236))
    dr=ImageDraw.Draw(sheet)
    for row,cid in enumerate(chosen):
        gray,paint,mask=tiles[cid]
        img=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        after=cv2.cvtColor(paint,cv2.COLOR_BGR2RGB)
        masks=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        masks[mask==1]=(50,225,100)
        masks[mask==0]=(215,80,160)
        for col,arr in enumerate((img,after,masks)):
            image=Image.fromarray(cv2.resize(arr,(330,330)))
            sheet.paste(image,(col*colw,row*rowh+20))
        dr.text((3,row*rowh+4),cid[-16:],fill=(0,0,0))
        dr.text((colw+3,row*rowh+4),'machine proposals, not truth',fill=(0,0,0))
        dr.text((2*colw+3,row*rowh+4),'pseudo positives / offsets / unknown',fill=(0,0,0))
    sheet.save(output)


def run(source:Path,weights:Path,outdir:Path,*,keep_source_zip=False):
    if sha256(source.read_bytes()).hexdigest()!=EXPECTED_SOURCE_SHA:
        raise ValueError('Private S35 ZIP must match immutable verified source')
    outdir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    with zipfile.ZipFile(source) as z:
        drafts=json.loads(z.read('annotations/S31_MACHINE_DRAFTS.json'))
        if len(drafts['photos'])!=39 or drafts['model_drafts_are_ground_truth'] is not False:
            raise ValueError('Wrong S31 provenance; native ground truth forbidden')
        models={};checks={}
        results=[];tiles={}
        for index,p in enumerate(drafts['photos']):
            photo=p['photo_id'];cid=p['crop_id']
            raw=z.read('source_39_H50Z_crops/'+cid+'.png')
            gray=cv2.imdecode(np.frombuffer(raw,dtype='uint8'),cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape!=(512,512):raise ValueError('Wrong crop pixels')
            if _sha_bytes(gray.tobytes())!=p['source_gray_pixels_sha256']:
                raise ValueError('Decoded source H50Z pixel SHA mismatch')
            # Model training photograph is ALWAYS the other photograph.
            trained_on='006' if photo=='005' else '005'
            if trained_on not in models:
                models[trained_on],checks[trained_on]=load_model(weights,trained_on)
            az=predict_cross_photo(gray,models[trained_on])
            proposed,extra=annotate_tile(gray,p['drafts'],az_map=az)
            st=extra['stats']
            st['photo_id']=photo;st['stratum']=p['stratum'];st['crop_id']=cid
            print(index+1,'/',39,cid,'added',st['new_ridge_proposals'],'machine pseudo+',st['machine_consensus_positive_lines'],flush=True)
            results.append({'crop_id':cid,'photo_id':photo,'stratum':p['stratum'],
                            'source_gray_sha256':p['source_gray_pixels_sha256'],
                            'native_origin_xy_PRIVATE':p['native_photo_origin_xy_px'],
                            'source_AZ_model_trained_on_other_photo':trained_on,
                            'proposals':proposed,'summary':st})
            name=outdir/'private_overlays'/f'{cid}_AUTO.png';name.parent.mkdir(exist_ok=True)
            paint=overlay(gray,p['drafts'],proposed);cv2.imwrite(str(name),paint)
            maskname=outdir/'machine_masks'/f'{cid}_PSEUDO_NOT_GT.png';maskname.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(maskname),extra['pseudo_mask'])
            tiles[cid]=(gray,paint,extra['pseudo_mask'])
    stats=[p['summary'] for p in results]
    count=Counter()
    for s in stats:
        for key in ('new_ridge_proposals','original_drafts','machine_consensus_positive_lines',
                    'positive_pixels_not_gt','local_offset_assumed_negative_pixels_not_gt','unknown_pixels'):
            count[key]+=s[key]
    empty_old=[s for s in stats if not s['original_drafts']]
    public={'schema':'s39_machine_only_public_aggregate_v1','source': '39 authentic 512x512 H50Z crops, 2 BMP same run',
            'genuine_H50Z_photos':2,'independent_physical_runs':1,
            'original_empty_crops':len(empty_old),
            'original_empty_crops_with_new_ridge_proposals':sum(bool(s['new_ridge_proposals']) for s in empty_old),
            'original_line_proposals':count['original_drafts'],
            'additional_new_ridge_proposals':count['new_ridge_proposals'],
            'all_machine_proposals':count['original_drafts']+count['new_ridge_proposals'],
            'machine_consensus_positive_lines_not_gt':count['machine_consensus_positive_lines'],
            'pseudo_positive_pixels_not_gt':count['positive_pixels_not_gt'],
            'assumed_offset_negative_pixels_not_gt':count['local_offset_assumed_negative_pixels_not_gt'],
            'unknown_pixels':count['unknown_pixels'],
            'model_training_photo_always_different_from_inference_photo':True,
            'source_and_models_SHA_verified':True,
            'native_physics_GT_labels':0,'auto_particle_identity_claim':False,
            'automatic_physical_joins_authorized':False,
            'pixel_labels_in_private_manifest_not_public':True,
            'important_note':'Ridge candidates may include scratches, fiducials or scanner artefacts; consensus is not independent physics truth.'}
    private={'schema':SCHEMA,'machine_only':True,'source_file_sha256':EXPECTED_SOURCE_SHA,
             'book_constraints':['visualMetro three-view + fiducials required for native 3D',
                    'Belyaev hadroide only for certified stopping tracks, not generic 2D lines'],
             'architecture_original_files_untouched':True,
             'training_model_photo_id_per_crop':'opposite H50Z photo, same run',
             'checkpoint_provenance':checks,'native_expert_gt_created':0,
             'auto_physical_join_allowed':False,'tiles':results}
    (outdir/'S39_MACHINE_HYPOTHESES_PRIVATE.json').write_text(json.dumps(private,ensure_ascii=False,indent=2)+'\n')
    (outdir/'S39_MACHINE_SUMMARY_PUBLIC.json').write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n')
    chosen=(drafts['photos'][1]['crop_id'],drafts['photos'][3]['crop_id'],
            drafts['photos'][15]['crop_id'],drafts['photos'][24]['crop_id'],
            drafts['photos'][30]['crop_id'],drafts['photos'][36]['crop_id'])
    _make_montage(tiles,chosen,outdir/'S39_PRIVATE_MACHINE_DEMO_MONTAGE.png')
    return public


def cli():
    parser=argparse.ArgumentParser()
    parser.add_argument('--private-source-zip',required=True,type=Path)
    parser.add_argument('--model-checkpoints',required=True,type=Path)
    parser.add_argument('--out-dir',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(run(args.private_source_zip,args.model_checkpoints,args.out_dir),indent=2))
if __name__=='__main__':cli()
