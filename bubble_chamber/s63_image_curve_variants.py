"""S63 zero-GT pixel-only fragment extraction variants, frozen before CAL/TEST."""
from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize
from .s39_machine_auto import (ridge_evidence, contrast_baseline, raw_detector_mask,
    fixed_budget_mask, extract_ridges, _trace_component, sample_bilinear)

METHODS=('s39_original','junction_less','darkness_skeleton','ridge_dark_fusion')


def _pixel_to_curves(mask:np.ndarray,score:np.ndarray,*,dilated_junction:bool)->list[np.ndarray]:
    """Never sees source curve identities. Only an image-derived candidate mask."""
    if mask.ndim!=2 or score.shape!=mask.shape or mask.dtype!=bool or not np.isfinite(score).all():
        raise ValueError('Invalid pixel-only candidate mask or scores')
    mask=mask.copy()
    mask[:7]=False;mask[-7:]=False;mask[:,:7]=False;mask[:,-7:]=False
    mask=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))>0
    skel=skeletonize(mask).astype('uint8')
    n=convolve(skel,np.ones((3,3),dtype='uint8'),mode='constant')-skel
    junction=(skel>0)&(n>=(4 if dilated_junction else 5))
    blocked=cv2.dilate(junction.astype('uint8'),np.ones((3,3),dtype='uint8'))>0 if dilated_junction else junction
    skel[blocked]=0
    nc,components,stats,_=cv2.connectedComponentsWithStats(skel,connectivity=8)
    outputs=[]
    for i in range(1,nc):
        if stats[i,cv2.CC_STAT_AREA]<18:continue
        x,y,w,h,_=stats[i]
        curve=_trace_component(components[y:y+h,x:x+w]==i)
        if curve is None:continue
        curve[:,0]+=x;curve[:,1]+=y
        if np.any(curve<7) or np.any(curve[:,0]>=mask.shape[1]-7) or np.any(curve[:,1]>=mask.shape[0]-7):
            continue
        outputs.append((float(np.median(sample_bilinear(score,curve))),curve))
    outputs.sort(key=lambda z:z[0],reverse=True)
    return [poly for _,poly in outputs[:32]]


def predict_variants(gray:np.ndarray)->dict[str,list[np.ndarray]]:
    if gray.dtype!=np.uint8 or gray.shape!=(192,192):
        raise ValueError('Only 192x192 uint8 image; no geometry or identity information')
    e=ridge_evidence(gray)
    ridge=e['score']
    contrast=contrast_baseline(gray)
    # Percentile-normalized positive responses, fused with a pre-frozen mixture.
    rscale=max(float(np.quantile(ridge,.99)),1e-7)
    cscale=max(float(np.quantile(contrast,.99)),1e-7)
    fused=np.clip(ridge/rscale,0,3)*.65+np.clip(contrast/cscale,0,3)*.35
    return {
        's39_original':extract_ridges(e,max_candidates=32),
        'junction_less':_pixel_to_curves(raw_detector_mask(e),ridge,dilated_junction=False),
        'darkness_skeleton':_pixel_to_curves(fixed_budget_mask(contrast,fraction=.035),contrast,dilated_junction=True),
        'ridge_dark_fusion':_pixel_to_curves(fixed_budget_mask(fused,fraction=.035),fused,dilated_junction=False),
    }


def choose_cal_only(aggregate:dict[str,dict[str,int]])->dict:
    if set(aggregate)!=set(METHODS):raise ValueError('Require all frozen arm results')
    def checked(v):
        for name in ('true_links','decoy_supported','full_paths','scenes'):
            if name not in v or int(v[name])!=v[name] or v[name]<0:raise ValueError(name)
    for v in aggregate.values():checked(v)
    base=aggregate['s39_original']
    from math import ceil
    minimum=max(5,int(ceil(base['true_links']*.15)))
    eligible=[m for m in METHODS if m!='s39_original' and
        aggregate[m]['true_links']-base['true_links']>=minimum and
        aggregate[m]['decoy_supported']-base['decoy_supported']<=2]
    selected=min(eligible,key=lambda m:(aggregate[m]['decoy_supported'],
        -aggregate[m]['true_links'],m)) if eligible else 's39_original'
    return {'chosen_on_CAL_only':selected,'min_required_true_join_gain':minimum,
            'eligible_methods':eligible,'synthetic_only':True}
