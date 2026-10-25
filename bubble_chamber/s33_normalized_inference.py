"""Original AZUNet normalized-score evidence for S31 machine photo drafts.

All outputs are provisional reviewer suggestions, NEVER physical track truth.
No train-on-test, pseudo-human promotion or silent joins are permitted.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import cv2
import numpy as np
import torch
from .s33_photometric_domain import image_input

SCHEMA='s33_uncalibrated_azunet_candidate_evidence_v1'


def patchwise_probabilities(model, gray:np.ndarray, *, mode:str, patch:int=64, stride:int=64, batch_size:int=4):
    """No-overlap baseline by default: matches training and avoids S27 AUC-confounded blending.

    Optional overlap averages predictions and is **not** presumed to improve
    physical trace recognition. 512 is exactly covered with final patch at 448.
    """
    if gray.shape!=(512,512) or gray.dtype!=np.uint8 or patch!=64 or stride not in (32,64):
        raise ValueError('Only source-locked 512px H50Z crops, 64px model windows')
    positions=list(range(0,512-patch+1,stride))
    if positions[-1]!=512-patch:positions.append(512-patch)
    tile_coords=[(y,x) for y in positions for x in positions]
    sums=np.zeros((512,512),np.float32);count=np.zeros_like(sums)
    model.eval()
    with torch.no_grad():
        for k in range(0,len(tile_coords),batch_size):
            subset=tile_coords[k:k+batch_size]
            batch=np.stack([image_input(gray[y:y+patch,x:x+patch],mode) for y,x in subset])[:,None]
            probs=torch.sigmoid(model(torch.from_numpy(batch))).squeeze(1).cpu().numpy()
            for (y,x),p in zip(subset,probs):
                sums[y:y+patch,x:x+patch]+=p
                count[y:y+patch,x:x+patch]+=1
    if np.any(count==0):raise AssertionError('Patch coverage gap')
    final=sums/count
    if not np.isfinite(final).all() or final.min()<0 or final.max()>1:
        raise ValueError('Bad network response')
    return final


def centerline_evidence(prob_map:np.ndarray,points:list)->dict:
    """Centerline-vs-flanking ANZA response; not a calibrated track probability."""
    if prob_map.shape!=(512,512) or not np.isfinite(prob_map).all():raise ValueError('Bad response map')
    p=np.asarray(points,dtype=np.float32)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<12 or not np.isfinite(p).all() or (p<0).any() or (p>512).any():
        raise ValueError('Malformed machine draft centerline')
    tangent=np.gradient(p,axis=0);norm=np.linalg.norm(tangent,axis=1)
    ok=(norm>1e-5)&(p[:,0]>7)&(p[:,0]<504)&(p[:,1]>7)&(p[:,1]<504)
    if np.count_nonzero(ok)<6:
        return {'status':'INSUFFICIENT_GEOMETRY','calibrated':False}
    p=p[ok];tangent=tangent[ok];norm=norm[ok,None]
    n=np.stack((-tangent[:,1],tangent[:,0]),axis=1)/norm
    def sample(xy):
        return cv2.remap(prob_map,xy[:,0].astype('float32').reshape(1,-1),
                         xy[:,1].astype('float32').reshape(1,-1),
                         interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101).reshape(-1)
    c=sample(p)
    left=sample(p+5*n);right=sample(p-5*n)
    contrast=c-.5*(left+right)
    return {'status':'UNVERIFIED_MODEL_EVIDENCE','calibrated':False,
            'center_mean_probability_NOT_TRACK_PROB':round(float(c.mean()),6),
            'flank_mean_probability_NOT_BACKGROUND_PROB':round(float(.5*(left+right).mean()),6),
            'median_center_minus_flank':round(float(np.median(contrast)),6),
            'mean_center_minus_flank':round(float(contrast.mean()),6),
            'sampled_machine_polyline_points':int(len(c))}


def score_s31_drafts(drafts:dict,tiles:Path,models:dict[str,torch.nn.Module], *,stride=64, crop_subset=None):
    if drafts.get('schema')!='s31_machine_first_h50z_draft_v1' or drafts.get('physical_joins_authorized') is not False:
        raise ValueError('S31 machine-only source snapshot required')
    if set(models)!={'quantile','local'}:
        raise ValueError('Fixed source-aware two-view AZUNet evidence only')
    if len(drafts.get('photos',[]))!=39:raise ValueError('Missing image source crops')
    tiles=Path(tiles);rows=[];summary={'crops':0,'proposals':0,'untrusted':True}
    selected=drafts['photos'] if crop_subset is None else [item for item in drafts['photos'] if item['crop_id'] in crop_subset]
    if len(selected)!=(len(drafts['photos']) if crop_subset is None else len(set(crop_subset))):
        raise ValueError('Missing or duplicate requested photo identities')
    for item in selected:
        cid=item['crop_id'];image=tiles/(cid+'.png')
        if not image.is_file() or sha256(image.read_bytes()).hexdigest()!=item['derived_png_sha256_from_BMP']:
            raise ValueError('Source crop SHA mismatch '+cid)
        gray=cv2.imread(str(image),cv2.IMREAD_GRAYSCALE)
        if gray is None or sha256(np.ascontiguousarray(gray).tobytes()).hexdigest()!=item['source_gray_pixels_sha256']:
            raise ValueError('Source original BMP pixels changed')
        maps={mode:patchwise_probabilities(model,gray,mode=mode,stride=stride) for mode,model in models.items()}
        for draft in item['drafts']:
            ev={mode:centerline_evidence(maps[mode],draft['points_xy']) for mode in models}
            if all(q['status']=='UNVERIFIED_MODEL_EVIDENCE' for q in ev.values()):
                delta=abs(ev['quantile']['median_center_minus_flank']-ev['local']['median_center_minus_flank'])
            else:delta=None
            rows.append({'crop_id':cid,'source_photo':item['photo_id'],'draft_id':draft['draft_id'],
                         'source_stratum':item['stratum'],'source_method':draft['source_method'],
                         'AZUNet_quantile':ev['quantile'],'AZUNet_local':ev['local'],
                         'cross_normalization_disagreement_UNCALIBRATED':delta,
                         'machine_draft_remains':'MACHINE_DRAFT_UNREVIEWED',
                         'expert_approved':False,'physics_auto_merge_allowed':False})
        summary['crops']+=1
    summary['proposals']=len(rows)
    result={'schema':SCHEMA,'original_AZConv_unchanged':True,
            'source_machine_draft_count':sum(len(x['drafts']) for x in drafts['photos']),
            'source_crops':len(drafts['photos']),'train_photo':'005',
            'unseen_real_photo':'006','same_physical_run':True,
            'human_verified_track_masks':0,'human_verified_links':0,
            'NO_MODEL_SCORES_ARE_CERTIFIED_PHYSICAL_TRACKS':True,
            'physics_auto_join_authorized':False,
            'stride':stride,'research_only':True,'rows':rows,'summary':summary}
    expected=sum(len(item['drafts']) for item in selected)
    if len(rows)!=expected:
        raise AssertionError('Lost machine draft candidate')
    result['selected_crops']=len(selected)
    result['selected_drafts']=expected
    result['partial_source_only']=crop_subset is not None
    return result
