"""Create weak train-only targets from S31 drafts and optional local user corrections.

Machine positives/negatives do NOT become certified real-photo truth. Every
unmarked original pixel remains UNKNOWN (255). All model weights are <1.
"""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import cv2,numpy as np
from .s31_machine_first import validate_draft_pack,actual_pixel_sha,file_sha

CORRECTION_SCHEMA='s31_machine_corrections_v1'


def validate_corrections(drafts:dict, corrections:dict|None):
    if corrections is None:return {}
    validate_draft_pack(drafts)
    if corrections.get('schema')!=CORRECTION_SCHEMA or corrections.get('physical_joins_authorized') is not False or corrections.get('all_unreviewed_stay_unknown') is not True:
        raise ValueError('Not a safe provisional-corrections document')
    if corrections.get('machine_source_s28_sha256')!=drafts.get('source_S28_proposals_sha256') or corrections.get('machine_source_s29_sha256')!=drafts.get('source_S29_reviewer_sha256'):
        raise ValueError('Wrong source snapshot')
    source={r['crop_id']:(r['source_original_png_sha256_from_S29'],r['source_gray_pixels_sha256']) for r in drafts['photos']}
    rows=corrections.get('crops')
    if not isinstance(rows,list) or len(rows)!=len(source) or len({r.get('crop_id') for r in rows})!=len(source):raise ValueError('Missing or duplicate photos')
    for r in rows:
        if r.get('crop_id') not in source or (r.get('source_crop_sha256'),r.get('source_gray_pixels_sha256'))!=source[r['crop_id']]:
            raise ValueError('Source photo changed')
    known={d['draft_id']:d for r in drafts['photos'] for d in r['drafts']}
    changes=corrections.get('decisions')
    if not isinstance(changes,dict):raise ValueError('Missing decision map')
    approved={}
    allowed={'USER_ACCEPTED_PROVISIONAL','USER_REJECTED_PROVISIONAL','USER_UNCERTAIN','USER_EDITED_PENDING_REVIEW'}
    for id,r in changes.items():
        if id not in known or not isinstance(r,dict) or r.get('status') not in allowed:raise ValueError('Unrecognized candidate/review status')
        edit=r.get('edited_points_xy')
        if edit is not None:
            p=np.asarray(edit,float)
            if p.shape!=(len(known[id]['points_xy']),2) or not np.isfinite(p).all() or p.min()<0 or p.max()>511:
                raise ValueError('Out-of-bounds or mismatched edited path')
        approved[id]=r
    relation_known={rel['relation_id'] for photo in drafts['photos'] for rel in photo.get('relation_drafts',[])}
    rels=corrections.get('relation_decisions',{})
    if not isinstance(rels,dict):raise ValueError('Malformed provisional relation review')
    for rid,r in rels.items():
        if rid not in relation_known or not isinstance(r,dict) or r.get('status') not in {'LOCAL_SAME_PROVISIONAL','LOCAL_DIFFERENT_PROVISIONAL','LOCAL_UNSURE'}:
            raise ValueError('Unverified invented relation decision ID/status')
    return approved


def _polyline_mask(shape,pts,thickness):
    points=np.asarray(pts,dtype='float64')
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<12 or not np.isfinite(points).all() or points.min()<0 or points.max()>511:
        raise ValueError('Invalid machine centerline')
    mask=np.zeros(shape,np.uint8)
    cv2.polylines(mask,[np.rint(points).astype(np.int32).reshape(-1,1,2)],False,1,int(thickness),lineType=cv2.LINE_8)
    return mask.astype(bool)


def choose_weak_candidates(photo_rec:dict, corrections:dict):
    """Use one conservative agreed anchor, or up to two pure-photo LSD lines.

    Rejected/edited-but-unapproved paths are never positive. Fiducial/quiet
    control crops have ZERO automatic positive training labels by design.
    """
    group=[];rec=photo_rec;stratum=rec['stratum']
    corrected_ids=set()
    for d in rec['drafts']:
        decision=corrections.get(d['draft_id'])
        if decision:
            corrected_ids.add(d['draft_id'])
            if decision['status']=='USER_ACCEPTED_PROVISIONAL':
                group.append({'draft':d,'points':decision.get('edited_points_xy') or d['points_xy'],
                   'weight':.6,'provenance':'LOCAL_USER_PROVISIONAL_NOT_EXPERT'})
    if corrected_ids:return group  # Any local correction overrides automatic selections for this crop.
    if stratum in {'fiducial_control','quiet'}:return group
    if 'start_xy_OPERATOR_REAL' in rec:
        original=[d for d in rec['drafts'] if d['source_method']=='classical' and
                  d['cross_method_agreement_UNCALIBRATED'] and d['draft_id'] not in corrected_ids]
        if original:
            best=max(original,key=lambda d:(d['photometry_support_fraction_UNCALIBRATED'],d['model_score_UNCALIBRATED']))
            group.append({'draft':best,'points':best['points_xy'],'weight':.35,
                          'provenance':'MACHINE_CROSS_METHOD_AGREEMENT_UNCALIBRATED'})
    else:
        # Only plausible long dark LINE PIECES, never an inferred merged track.
        selected=sorted([d for d in rec['drafts'] if d['source_method']=='LSD_line_piece'
                         and d.get('observed_contrast_8bit',0)>=6 and d['geometry']['pixel_arclength']>=58
                         and d['draft_id'] not in corrected_ids],key=lambda d:-d['model_score_UNCALIBRATED'])[:2]
        group.extend({'draft':d,'points':d['points_xy'],'weight':.16,'provenance':'MACHINE_LSD_LINE_PIECE'} for d in selected)
    return group


def build_weak_masks(drafts:dict,tile_dir:Path,output_dir:Path,*,corrections:dict|None=None,seed=20261025,
                     negative_points_per_crop=650):
    validate_draft_pack(drafts);changes=validate_corrections(drafts,corrections)
    if negative_points_per_crop<0 or negative_points_per_crop>2000:raise ValueError('Unbounded pseudo negative count')
    tile_dir=Path(tile_dir);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    details=[];summary={'schema':'s31_weak_training_masks_source_locked_v1','source_machine_schema':drafts['schema'],
       'negative_pixels_are_machine_assumptions':True,'positive_pixels_are_machine_or_local_user_unverified':True,
       'all_other_pixels_unknown_255':True,'human_certified_track_pixels':0,'physics_auto_join_allowed':False,
       'train_photo':'005','heldout_photo':'006','independent_physical_run':False,'entries':details}
    for rec in drafts['photos']:
        name=rec['crop_id'];path=tile_dir/(name+'.png')
        if not path.is_file() or file_sha(path)!=rec['derived_png_sha256_from_BMP']:raise ValueError('Photo crop byte checksum failed: '+name)
        gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if actual_pixel_sha(gray)!=rec['source_gray_pixels_sha256']:raise ValueError('Authentic photo pixel checksum failed: '+name)
        target=np.full((512,512),255,np.uint8);weight=np.zeros((512,512),np.float32)
        selected=choose_weak_candidates(rec,changes)
        any_draft=np.zeros_like(target)
        for d in rec['drafts']:
            any_draft |= _polyline_mask(target.shape,d['points_xy'],9).astype('uint8')
        for d in selected:
            m=_polyline_mask(target.shape,d['points'],3)
            target[m]=1;weight[m]=np.maximum(weight[m],d['weight'])
        # Unreviewed dark lines MUST NOT become negative. Machine-generated
        # 'probably empty background' probes stay soft, low weight and separate.
        blur=cv2.GaussianBlur(gray.astype('float32'),(0,0),4)
        dark=blur-gray.astype('float32')
        safe=(np.abs(dark)<3.0)&(any_draft==0)&(target==255)
        if (target==1).any():
            # The local sampling band prevents the negative class from being
            # almost absent in 64px training patches around the draft track.
            distance=cv2.distanceTransform((target!=1).astype('uint8'),cv2.DIST_L2,5)
            safe &= (distance>=10)&(distance<=38)
        if 'start_xy_OPERATOR_REAL' in rec:
            sx,sy=rec['start_xy_OPERATOR_REAL'];xx,yy=np.meshgrid(np.arange(512),np.arange(512))
            safe &= (xx-sx)**2+(yy-sy)**2>32**2
        coords=np.argwhere(safe)
        if len(coords):
            rng=np.random.default_rng(seed+int(hashlib.sha256(name.encode()).hexdigest()[:8],16))
            pick=coords[rng.choice(len(coords),size=min(negative_points_per_crop,len(coords)),replace=False)]
            target[pick[:,0],pick[:,1]]=0;weight[pick[:,0],pick[:,1]]=.03
        if (weight[target==255]!=0).any():raise AssertionError('Unknown pixel incorrectly weighted')
        dest=output_dir/'masks';dest.mkdir(exist_ok=True)
        mfile=dest/(name+'_pseudo_01255.png');npfile=dest/(name+'_weak_weights.npy')
        if not cv2.imwrite(str(mfile),target):raise IOError('Cannot save source draft mask')
        np.save(npfile,weight.astype('float16'))
        details.append({'crop_id':name,'photo_id':rec['photo_id'],'stratum':rec['stratum'],
          'image_sha256':rec['derived_png_sha256_from_BMP'],'image_pixel_sha256':rec['source_gray_pixels_sha256'],
          'mask':'masks/'+mfile.name,'mask_sha256':file_sha(mfile),
          'weight':'masks/'+npfile.name,'weight_sha256':file_sha(npfile),
          'provisional_positive_count':int((target==1).sum()),'assumed_background_count':int((target==0).sum()),
          'unknown_pixels':int((target==255).sum()),'used_candidate_ids':[d['draft']['draft_id'] for d in selected],
          'training_sources':[d['provenance'] for d in selected],'expert_confirmed':False})
    summary['counts']={'crops':len(details),'source_start_crops':sum('start' in r['crop_id'] for r in details),
      'weak_positive_px':sum(r['provisional_positive_count'] for r in details),
      'weak_negative_px':sum(r['assumed_background_count'] for r in details),
      'unlabeled_px':sum(r['unknown_pixels'] for r in details),
      'crops_with_weak_positive':sum(r['provisional_positive_count']>0 for r in details)}
    (output_dir/'weak_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary
