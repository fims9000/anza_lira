"""S65: image-only endpoint-termination abstention, never physical proof.

A post-filter on S64 pixel-extracted global path proposals, using ONLY pixels
and prediction geometry. Synthetic labels are confined to the external CAL
selection and held-out TEST evaluator, not present here.
"""
from __future__ import annotations
from math import ceil

import cv2
import numpy as np

from .s56_global_path_assembly import Edge

THRESHOLD_GRID=(.25,.40,.55,.70,.85,1.,1.25)
OUTWARD_SAMPLES=(2.,3.,4.,5.,6.)
MIN_INSIDE_DARKNESS=.010


def endpoint_termination_ratio(gray:np.ndarray,poly:np.ndarray,port:int)->float:
    """Continuing dark centerline outside an endpoint means possible truncation.

    Ratio is a feature, NOT a probability; invariant to point-list reversal
    when the selected physical endpoint is reversed as well.
    """
    if gray.shape!=(192,192) or gray.dtype!=np.uint8 or port not in (0,-1):
        raise ValueError('Invalid pixel/port input')
    p=np.asarray(poly,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<8 or not np.isfinite(p).all():
        raise ValueError('Bad inferred curve')
    idx=min(3,len(p)-1)
    tip=p[0] if port==0 else p[-1]
    inner=p[idx] if port==0 else p[-idx-1]
    outward=tip-inner
    length=np.linalg.norm(outward)
    if length<1e-6:return float('inf')
    outward=outward/length
    offsets=np.asarray(OUTWARD_SAMPLES)
    outside=tip+offsets[:,None]*outward
    inside=tip-offsets[:,None]*outward
    xy=np.vstack((outside,inside))
    if np.any(xy<0) or np.any(xy[:,0]>=gray.shape[1]) or np.any(xy[:,1]>=gray.shape[0]):
        return float('inf')
    im=gray.astype('float32')/255.
    dark=np.maximum(cv2.GaussianBlur(im,(0,0),7.)-im,0.)
    vals=cv2.remap(dark,xy[:,0].astype('float32')[None,:],
                   xy[:,1].astype('float32')[None,:],cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REFLECT_101)[0]
    if vals.shape!=(10,):raise AssertionError('Bad endpoint sampling')
    outside_strength=float(np.percentile(vals[:5],75))
    inside_strength=float(np.percentile(vals[5:],75))
    return float(outside_strength/max(inside_strength,MIN_INSIDE_DARKNESS))


def edge_termination_risk(gray:np.ndarray,polylines:list[np.ndarray],edge:Edge)->float:
    if edge.i>=len(polylines) or edge.j>=len(polylines):
        raise ValueError('Invalid edge source index')
    return max(endpoint_termination_ratio(gray,polylines[edge.i],edge.end_i),
               endpoint_termination_ratio(gray,polylines[edge.j],edge.end_j))


def retain_by_endpoint_guard(gray:np.ndarray,polylines:list[np.ndarray],
                             edges:list[Edge],threshold:float|None)->list[Edge]:
    """Never invent a connection; output is a stable subset of existing edges."""
    if threshold is None:return list(edges)
    if threshold not in THRESHOLD_GRID:
        raise ValueError('CAL threshold not in frozen grid')
    return [edge for edge in edges if edge_termination_risk(gray,polylines,edge)<=threshold]


def choose_cal_only(cal_records:dict[float|None,dict[str,int]])->dict:
    """Synthetic CAL labels are allowed HERE, never in pixel inference."""
    if set(cal_records)!={None,*THRESHOLD_GRID}:
        raise ValueError('Missing CAL-only grid arms')
    for name,v in cal_records.items():
        if not all(k in v for k in ('correct','noncorrect','scenes')):
            raise ValueError(f'Missing CAL summary: {name}')
        if any(type(v[k]) is not int or v[k]<0 for k in ('correct','noncorrect','scenes')):
            raise ValueError('Invalid CAL aggregate')
    base=cal_records[None]
    min_gain=max(3,int(ceil(.25*base['noncorrect'])))
    keep=int(ceil(.60*base['correct']))
    eligible=[th for th in THRESHOLD_GRID if
              base['noncorrect']-cal_records[th]['noncorrect']>=min_gain and
              cal_records[th]['correct']>=keep]
    chosen=min(eligible,key=lambda th:(cal_records[th]['noncorrect'],
                                     -cal_records[th]['correct'],th)) if eligible else None
    return {'selected_CAL_only':chosen,'eligible_thresholds':eligible,
            'required_noncorrect_reduction':min_gain,
            'min_correct_retained_CAL':keep,
            'fallback_is_unfiltered_original_S64':chosen is None,
            'synthetic_only':True}
