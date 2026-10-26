"""Synthetic multi-gap 2D *source-measurement-shape* scenes on native backgrounds.

The known labels are synthetic generator identities ONLY. Original H50Z is
unlabeled and the visualMetro measurements belong to film 070, not H50Z.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .s35_simulated_gaps import _shape_curve, _draw_stroke

TRUE_EDGES=frozenset(((0,1),(1,2),(3,4),(4,5)))
SCENARIOS=('parallel','near_parallel','crossing','scratched','empty_gap')

@dataclass
class ThreePieceScene:
    gray: np.ndarray
    nodes: list[np.ndarray]
    truth_SYNTHETIC_ONLY: frozenset[tuple[int,int]]
    scenario: str
    private_source_frames: tuple[str,str]
    jitter_px: float


def make_three_piece_scene(photo_tile:Path, templates:list[dict], *, seed:int,
                           scenario:str='parallel') -> ThreePieceScene:
    if scenario not in SCENARIOS or len(templates)<24:
        raise ValueError('Bad locked scenario or insufficient measurement templates')
    source=cv2.imread(str(photo_tile), cv2.IMREAD_GRAYSCALE)
    if source is None or source.shape!=(512,512):
        raise ValueError('Authenticated source tile must be 512x512 gray')
    rng=np.random.default_rng(seed)
    x,y=rng.integers(0,321,size=2)
    crop=source[y:y+192,x:x+192].astype(np.float32)
    strokes=np.zeros((192,192),dtype=bool)
    t0,t1=(templates[int(rng.integers(len(templates)))] for _ in range(2))
    length=float(rng.uniform(112,140))
    ref_y=float(rng.uniform(83,110))
    a=_shape_curve(t0,length=length,center_y=ref_y,tilt=float(rng.uniform(-9,9)))
    b=_shape_curve(t1,length=length,center_y=ref_y,tilt=float(rng.uniform(-9,9)))
    u=np.linspace(0,1,len(b))
    if scenario=='near_parallel':offset=float(rng.choice((-1,1))*rng.uniform(4.5,10))
    else:offset=float(rng.choice((-1,1))*rng.uniform(17,29))
    b[:,1]+=offset*(1.3-2.6*u) if scenario=='crossing' else offset
    tracks=(a,b)
    if any(np.min(t)<10 or np.max(t[:,0])>180 or np.max(t[:,1])>180
           for t in tracks):
        raise ValueError('Injected source paths outside supported image')
    bands=((3,34),(44,75),(85,116))
    fragments=[]
    for curve in tracks:
        ink=float(rng.uniform(23,76));width=int(rng.integers(1,4))
        for l,r in bands:
            frag=curve[l:r].copy()
            _draw_stroke(crop,strokes,frag,ink=ink,width=width)
            fragments.append(frag)
        if scenario!='empty_gap':
            # Weak residual events are not added to labels, just hard visuals.
            for lo,hi in ((34,44),(75,85)):
                for pt in curve[lo:hi:3]:
                    if rng.random()<.65:
                        cv2.circle(crop,tuple(np.rint(pt).astype(int)),1,
                                   -float(ink*rng.uniform(.05,.20)),-1)
    # Hard false fragment: a displaced copy of a valid source shape, equal
    # to all genuine fragment lengths; its identity is always synthetic decoy.
    prototype=tracks[int(rng.integers(0,2))][85:116].copy()
    dy=float(rng.choice((-1,1))*rng.uniform(2.5,10.))
    prototype[:,1]+=dy+rng.uniform(-3,3)*np.linspace(0,1,len(prototype))
    if scenario=='scratched':prototype[:,1]+=1.5*np.sin(np.linspace(0,4,len(prototype)))
    _draw_stroke(crop,strokes,prototype,ink=float(rng.uniform(27,89)),width=int(rng.integers(1,4)))
    fragments.append(prototype)
    # Simulate independent errors in extraction. No oracle endpoints are passed
    # to the scoring/selection algorithms.
    jitter=float(rng.choice((.6,1.3,2.2,3.0)))
    observed=[]
    for fragment in fragments:
        tangent=np.gradient(fragment,axis=0)
        norm=np.maximum(np.linalg.norm(tangent,axis=1),1e-8)
        normal=np.column_stack((-tangent[:,1],tangent[:,0]))/norm[:,None]
        noise=gaussian_filter1d(rng.normal(0,jitter,size=31),sigma=2,mode='reflect')
        w=np.linspace(0,1,31)
        drift=(float(rng.normal(0,1.5*jitter))*(1-w)**4+
               float(rng.normal(0,1.5*jitter))*w**4)
        noisy=np.clip(fragment+normal*(noise+drift)[:,None],0,191)
        if rng.random()<.5:noisy=noisy[::-1].copy()
        observed.append(noisy)
    if len(observed)!=7 or any(x.shape!=(31,2) for x in observed):
        raise AssertionError('Synthetic candidate topology failure')
    return ThreePieceScene(cv2.GaussianBlur(np.clip(crop,0,255).astype('uint8'),(0,0),.35),
                           observed,TRUE_EDGES,scenario,(t0['frame'],t1['frame']),jitter)
