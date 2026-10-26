"""S66: ridge-guided subpixel-free (1px-step) endpoint extension from pixels only.

A deliberately bounded, reversible candidate operation. It never receives GT
and never asserts that lines are physical particle tracks.
"""
from __future__ import annotations
import cv2
import numpy as np
from .s39_machine_auto import ridge_evidence,fixed_budget_mask,contrast_baseline

ARMS=('s39_original','ridge_extend_strict','ridge_extend_relaxed')
POLICY={
    'ridge_extend_strict':{'fraction':.035,'min_darkness':.012,'max_pixels':8},
    'ridge_extend_relaxed':{'fraction':.075,'min_darkness':.008,'max_pixels':10},
}
STEER_DEGREES=(-12.,0.,12.)


def _extend_tip(poly,which,mask,dark,max_steps,min_darkness):
    if which not in (0,-1):raise ValueError('Expected one physical endpoint')
    p=np.asarray(poly,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<8 or not np.isfinite(p).all():
        raise ValueError('Invalid pixel-derived polyline')
    tip=p[0] if which==0 else p[-1]
    interior=p[min(3,len(p)-1)] if which==0 else p[max(0,len(p)-4)]
    direction=tip-interior
    size=float(np.linalg.norm(direction))
    if size<1e-6:return []
    direction/=size
    accepted=[]
    h,w=mask.shape
    for _ in range(max_steps):
        choices=[]
        for angle in STEER_DEGREES:
            theta=np.deg2rad(angle)
            c,s=np.cos(theta),np.sin(theta)
            rotated=np.array([direction[0]*c-direction[1]*s,
                              direction[0]*s+direction[1]*c])
            trial=tip+rotated
            x,y=np.rint(trial).astype(int)
            if x<7 or y<7 or x>=w-7 or y>=h-7 or not mask[y,x] or dark[y,x]<min_darkness:
                continue
            choices.append((float(dark[y,x]),-abs(angle),trial,rotated))
        if not choices:break
        _,_,tip,direction=max(choices,key=lambda z:(z[0],z[1]))
        accepted.append(tip.copy())
    return accepted


def extend_image_curves(gray:np.ndarray,pieces:list[np.ndarray],arm:str)->list[np.ndarray]:
    """Original S39 curves, optional bounded pixel-only endpoint tracing."""
    if gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('Need real-backed uint8 192x192 pixel tile')
    if arm not in ARMS:raise ValueError('Only frozen S66 arms')
    if len(pieces)>32:raise ValueError('Pixel detector budget exceeded')
    if arm=='s39_original':return [np.asarray(p,float).copy() for p in pieces]
    p=POLICY[arm]
    ridge=ridge_evidence(gray)
    mask=fixed_budget_mask(ridge['score'],fraction=p['fraction'])
    dark=contrast_baseline(gray)
    output=[]
    for poly in pieces:
        p0=np.asarray(poly,float)
        left=_extend_tip(p0,0,mask,dark,p['max_pixels'],p['min_darkness'])
        right=_extend_tip(p0,-1,mask,dark,p['max_pixels'],p['min_darkness'])
        extended=np.vstack([*left[::-1],*p0,*right]) if left or right else p0.copy()
        # Keep central samples and valid graph-geometry constraints unchanged.
        assert len(extended)>=len(p0)
        assert np.isfinite(extended).all()
        output.append(extended)
    return output


def choose_calibration_only(aggregate:dict[str,dict[str,int]])->dict:
    if set(aggregate)!=set(ARMS):raise ValueError('Frozen S66 arms missing')
    fields=('synthetic_correct','synthetic_noncorrect','matched_source_fragments',
            'synthetic_decoy_matched','scenes')
    if any(any(key not in values or type(values[key]) is not int or values[key]<0 for key in fields)
           for values in aggregate.values()):
        raise ValueError('Invalid CAL totals')
    base=aggregate['s39_original']
    options=[]
    for arm in ARMS[1:]:
        a=aggregate[arm]
        if (a['synthetic_correct']-base['synthetic_correct']>=2 or
                a['matched_source_fragments']-base['matched_source_fragments']>=5) and \
                a['synthetic_noncorrect']<=base['synthetic_noncorrect'] and \
                a['synthetic_decoy_matched']<=base['synthetic_decoy_matched']:
            options.append(arm)
    choice=min(options,key=lambda name:(-aggregate[name]['synthetic_correct'],
                -aggregate[name]['matched_source_fragments'],aggregate[name]['synthetic_noncorrect'],
                ARMS.index(name))) if options else 's39_original'
    return {'CAL_only_selected':choice,'eligible':options,
            'original_default':'s39_original','synthetic_only':True,
            'physical_auto_join_authorized':False}
