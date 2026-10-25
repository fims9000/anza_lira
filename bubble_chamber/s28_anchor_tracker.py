"""S28 seed-anchored, curvature-regularized path proposals for bubble-film ROIs.

Uses a 2D START landmark as an approximate pixel-space seed. It is NOT a
certified particle line or identity. All returned continuations require human
review and cannot be used to infer 3D momentum, dE/dx or particle species.
"""
from __future__ import annotations
import numpy as np
import cv2


def _sample_lateral_evidence(evidence,origin,theta,length,width,step):
    u=np.arange(0,length+1,step,dtype='float32')
    v=np.arange(-width,width+1,step,dtype='float32')
    ux,uy=np.cos(theta),np.sin(theta)
    vx,vy=-uy,ux
    xx=origin[0]+u[:,None]*ux+v[None,:]*vx
    yy=origin[1]+u[:,None]*uy+v[None,:]*vy
    values=cv2.remap(evidence,xx.astype('float32'),yy.astype('float32'),
                     interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=-1.)
    # prevent a path outside the physical crop from yielding false high reward
    out=(xx<0)|(xx>=evidence.shape[1])|(yy<0)|(yy>=evidence.shape[0])
    values[out]=-1.
    return values,u,v,xx,yy


def _dp_path(reward:np.ndarray, curvature_penalty:float=.07,slope_penalty:float=.035,
             start_radius_bins:int=3)->tuple[np.ndarray,float]:
    """Small state-space Viterbi: lateral displacement and tangent direction.

    Each advance is one 2px u-step. Lateral index jumps -1,0,+1, so tangent
    cannot jump arbitrarily. Curvature penalizes changes in the step direction.
    """
    if reward.ndim!=2 or reward.shape[0]<3 or reward.shape[1]<7 or not np.isfinite(reward).all():
        raise ValueError('bad finite sampling lattice')
    time_steps,width=reward.shape;mid=width//2
    steps=np.arange(-1,2,dtype=int)
    dp=np.full((width,3),-1e9,dtype='float32')
    dp[max(0,mid-start_radius_bins):min(width,mid+start_radius_bins+1),1]=reward[0,max(0,mid-start_radius_bins):min(width,mid+start_radius_bins+1)]
    ptr=np.full((time_steps,width,3),-1,dtype='int8')
    ar=np.arange(width)
    for t in range(1,time_steps):
        nextdp=np.full_like(dp,-1e9)
        for di,d in enumerate(steps):
            indices=ar-d
            good=(indices>=0)&(indices<width)
            previous=dp[np.clip(indices,0,width-1)]
            cost=np.asarray([curvature_penalty*(d-d0)**2 for d0 in steps],dtype='float32')
            vals=previous-cost[None,:]
            best=np.argmax(vals,axis=1)
            gains=vals[ar,best]+reward[t]-slope_penalty*abs(d)
            gains[~good]=-1e9
            nextdp[:,di]=gains
            ptr[t,:,di]=best.astype('int8')
        dp=nextdp
    idx=int(np.argmax(dp));vindex=idx//3;dindex=idx%3
    if dp[vindex,dindex] < -1e7:
        return np.array([],dtype=int),-1e9
    result=np.empty(time_steps,dtype=int)
    result[-1]=vindex
    for t in range(time_steps-1,0,-1):
        oldd=int(steps[dindex]);prevd=int(ptr[t,vindex,dindex])
        vindex=vindex-oldd;dindex=prevd
        result[t-1]=vindex
    return result,float(dp.reshape(-1)[idx]/time_steps)


def candidate_paths(evidence: np.ndarray,seed_xy:tuple[float,float], *,
                    length:int=190,width:int=48,step:int=2,
                    angles:int=24,top_k:int=5,
                    curvature_penalty:float=.07)->list[dict]:
    """Return unverified candidate polylines (pixel units, original orientation)."""
    ev=np.asarray(evidence,'float32')
    if ev.ndim!=2 or not np.isfinite(ev).all() or np.max(ev)>1.0001 or np.min(ev)<0:
        raise ValueError('finite [0,1] photo evidence required')
    if min(ev.shape)<128 or length<40 or width<8 or step<1 or length>max(ev.shape)*2:
        raise ValueError('invalid proposal geometry')
    if not 0<=seed_xy[0]<ev.shape[1] or not 0<=seed_xy[1]<ev.shape[0]:
        raise ValueError('seed outside photo')
    if angles<8 or angles>96 or top_k<1:raise ValueError('invalid angle count')
    candidates=[]
    for idx in range(angles):
        theta=2*np.pi*idx/angles
        rewards,u,v,xx,yy=_sample_lateral_evidence(ev,seed_xy,theta,length,width,step)
        choose,score=_dp_path(rewards,curvature_penalty=curvature_penalty)
        if len(choose)!=len(u) or score<=-1e5:continue
        pts=np.stack([xx[np.arange(len(u)),choose],yy[np.arange(len(u)),choose]],axis=1)
        # Require mostly in-image observed image evidence.
        on=(pts[:,0]>=0)&(pts[:,0]<ev.shape[1])&(pts[:,1]>=0)&(pts[:,1]<ev.shape[0])
        if not on.all():continue
        vals=rewards[np.arange(len(u)),choose]
        candidates.append({'direction_angle_deg':float(idx*360/angles),
             'model_selection_score_UNCALIBRATED':float(score),
             'mean_observed_photo_evidence':float(vals.mean()),
             'fraction_steps_above_local_signal_0_15':float((vals>.15).mean()),
             'path_points_original_xy':pts.round(2).tolist(),
             'human_confirmed_track':False,'automatic_physical_join_allowed':False})
    candidates.sort(key=lambda x:-x['model_selection_score_UNCALIBRATED'])
    selected=[]
    for row in candidates:
        if all(min(abs(row['direction_angle_deg']-other['direction_angle_deg']),
                   360-abs(row['direction_angle_deg']-other['direction_angle_deg'])) >=360/angles*2-1e-9 for other in selected):
            selected.append(row)
            if len(selected)==top_k:break
    return selected


def evidence_from_photo(gray:np.ndarray,az_score:np.ndarray|None=None)->np.ndarray:
    """A *fixed* photometric/ridge+AZ score mixture for review, NOT calibration.

    This is deliberately exploratory, not claimed to be optimum or better than
    either source model; none of these source pixels is expert labeled.
    """
    if gray.dtype!=np.uint8 or gray.ndim!=2:raise ValueError('8-bit gray photo required')
    blur=cv2.GaussianBlur(gray.astype('float32'),(0,0),5)
    local=np.clip((blur-gray.astype('float32'))/45.,0,1)
    # High-line-response directional kernel, scaled from S26's two-channel
    # preprocessing constants; not a learned real H50Z calibration.
    from scripts.research.bubble_chamber.s26_two_photo_baselines import classical_score
    ridge=np.clip(classical_score(gray,'steerable_ridge')/.065,0,1)
    if az_score is None:
        return np.clip(.65*local+.35*ridge,0,1).astype('float32')
    if az_score.shape!=gray.shape or not np.isfinite(az_score).all():
        raise ValueError('bad native-photo ANZA map')
    return np.clip(.45*local+.25*ridge+.30*az_score,0,1).astype('float32')
