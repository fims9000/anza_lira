#!/usr/bin/env python3
"""Compile actual human strokes from S26 HTML into original AZUNet 0/1/255 manifest.

UNKNOWN pixels (255) are never used as background. Reviews are paired with
cryptographic source-crop checksums. Absence of signed expert decisions is a
hard training block, not permission to create pseudo-ground truth.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np,cv2


def stroke_pixels(row,shape=(512,512)):
    if row.get('label') not in ('track','artifact'):raise ValueError('invalid review class')
    rad=row.get('radius_px')
    if not isinstance(rad,(int,float)) or not 1<=rad<=12:raise ValueError('bad radius')
    pts=row.get('points_xy')
    if not isinstance(pts,list) or not pts:raise ValueError('empty hand stroke')
    if any(not isinstance(p,list) or len(p)!=2 or any(not isinstance(q,(int,float)) or not np.isfinite(q) or q<0 or q>511 for q in p) for p in pts):
        raise ValueError('nonfinite or outside-source-bounds hand drawn coordinates')
    arr=np.rint(np.asarray(pts)).astype('int32')
    out=np.zeros(shape,'uint8')
    if len(pts)==1:cv2.circle(out,tuple(arr[0]),int(round(rad)),1,-1)
    else:cv2.polylines(out,[arr.reshape(-1,1,2)],False,1,max(1,int(round(2*rad))),cv2.LINE_8)
    return out.astype(bool)


def compile_review(review_file:Path,source_crops:Path,out_dir:Path):
    review=json.loads(review_file.read_text(encoding='utf-8'))
    if review.get('schema')!='s26_real_track_manual_review_v1' or not isinstance(review.get('images'),list):
        raise ValueError('source-matched S26 human stroke JSON required')
    if review.get('generated_model_labels')!=0 or review.get('unknown_pixels_not_background') is not True:
        raise ValueError('do not promote generated model predictions into human labels')
    if len(review['images'])!=9:raise ValueError('expected the 9 reviewed source ROIs, even if some are unknown')
    # Preflight all nine source files, labels and conflicts BEFORE writing any
    # derived masks. A forged digest on the final crop must not leave an
    # apparently valid partial manifest from the first eight crops.
    seen_pre=set()
    for row in review['images']:
        id=row.get('crop_id')
        if not isinstance(id,str) or not id.startswith('H50Z_030_01_') or '/' in id or '\\' in id or '..' in id or id in seen_pre:
            raise ValueError('unknown, duplicated or unsafe crop id')
        seen_pre.add(id)
        src=(source_crops/f'{id}.png').resolve()
        if not src.is_relative_to(source_crops.resolve()) or not src.is_file():raise ValueError('original crop absent')
        if row.get('source_crop_sha256')!=hashlib.sha256(src.read_bytes()).hexdigest() or row.get('width')!=512 or row.get('height')!=512:
            raise ValueError('image digest or source coordinate system mismatch')
        if not isinstance(row.get('strokes'),list):raise ValueError('missing manual strokes array')
        tentative=np.full((512,512),255,dtype='uint8')
        for stroke in row['strokes']:
            pts=stroke_pixels(stroke)
            category=1 if stroke['label']=='track' else 0
            if np.any(pts&(tentative!=255)&(tentative!=category)):
                raise ValueError('conflicting track and artifact labels on same pixel')
            tentative[pts]=category
    if len(seen_pre)!=9:raise ValueError('9 distinct, source-verified crop IDs required')
    out_dir.mkdir(parents=True,exist_ok=True)
    entries=[];coverage=[];seen=set()
    for row in review['images']:
        id=row.get('crop_id')
        if not isinstance(id,str) or not id.startswith('H50Z_030_01_') or '/' in id or '\\' in id or '..' in id or id in seen:
            raise ValueError('unknown, duplicated or unsafe crop id')
        seen.add(id)
        src=(source_crops/f'{id}.png').resolve()
        if not src.is_relative_to(source_crops.resolve()) or not src.is_file():raise ValueError('original crop absent')
        sha=hashlib.sha256(src.read_bytes()).hexdigest()
        if row.get('source_crop_sha256')!=sha or row.get('width')!=512 or row.get('height')!=512:
            raise ValueError('image digest or source coordinate system mismatch')
        img=cv2.imread(str(src),cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape!=(512,512):raise ValueError('invalid source crop')
        mask=np.full((512,512),255,dtype='uint8')
        count={'track':0,'artifact':0}
        for stroke in row.get('strokes',[]):
            pts=stroke_pixels(stroke)
            category=1 if stroke['label']=='track' else 0
            if np.any(pts & (mask!=255) & (mask!=category)):
                raise ValueError('conflicting track and artifact labels on same pixel')
            mask[pts]=category
            count[stroke['label']]+=1
        photo='005' if id.startswith('H50Z_030_01_005_') else '006' if id.startswith('H50Z_030_01_006_') else None
        if photo is None:raise ValueError('unexpected photo ID')
        split='train' if photo=='005' else 'val'
        if not count['track'] and not count['artifact']:split='unlabeled'
        sub=out_dir/'crops';sub.mkdir(exist_ok=True)
        dest=sub/src.name
        if not dest.exists():shutil.copy2(src,dest)
        if hashlib.sha256(dest.read_bytes()).hexdigest()!=sha:raise ValueError('copy integrity failed')
        record={'film_id':'030','run_id':'H50Z_030','frame_id':photo,'image':'crops/'+src.name,
             'image_sha256':sha,'split':split,'mask':None}
        if split!='unlabeled':
            mask_dir=out_dir/'masks';mask_dir.mkdir(exist_ok=True)
            mask_path=mask_dir/f'{id}_manual_01255.png'
            cv2.imwrite(str(mask_path),mask)
            record['mask']='masks/'+mask_path.name
            record['mask_sha256']=hashlib.sha256(mask_path.read_bytes()).hexdigest()
        entries.append(record)
        coverage.append({'id':id,'photo':photo,'split':split,'human_strokes_declared':count,
                         'positive_pixels':int((mask==1).sum()),'negative_pixels':int((mask==0).sum()),
                         'unknown_pixels':int((mask==255).sum())})
    g={s:[r for r in entries if r['split']==s] for s in ['train','val','unlabeled']}
    ntrain=[r for r in coverage if r['split']=='train'];nval=[r for r in coverage if r['split']=='val']
    sufficient=bool(ntrain and nval and all(any(r[key]>0 for r in group) for group in (ntrain,nval) for key in ('positive_pixels','negative_pixels')))
    manifest={'schema':'s14_sparse_image_reviews_v1','entries':entries}
    out=(out_dir/'human_only_manifest.json');out.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    receipt={'schema':'s26_verified_source_sparse_review_receipt_v1',
             'human_review_declared_not_independently_authenticated':True,
             'all_other_real_pixels_unknown':True,'positive_reviewed_pixel_total':sum(x['positive_pixels'] for x in coverage),
             'negative_reviewed_pixel_total':sum(x['negative_pixels'] for x in coverage),
             'trainable_under_original_sparse_pipeline':sufficient,
             'status':'READY_FOR_SPARSE_REVIEW_TRAINING' if sufficient else 'BLOCKED_NEED_0_AND_1_MARKS_IN_BOTH_PHOTOS',
             'independent_physical_run_test':False,'records':coverage}
    (out_dir/'review_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review-json',type=Path,required=True)
    p.add_argument('--source-crops',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();report=compile_review(a.review_json,a.source_crops,a.out)
    print(json.dumps({'status':report['status'],'positives':report['positive_reviewed_pixel_total'],
                      'negative':report['negative_reviewed_pixel_total']},ensure_ascii=False))
