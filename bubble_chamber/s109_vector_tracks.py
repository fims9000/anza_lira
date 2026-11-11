"""S109 prototype: vector trajectory candidates from existing S108 ordered data.

A binary ridge mask is an *evidence source*, not the output object. The output
contains cubic smoothing B-splines, candidate identities, cross/branch audit,
and two strictly separate sampled domains: source-backed vs inferred geometry.

A vector hypothesis is not a physically identified particle track, and fitting
curves cannot resolve track association across crossings or blank gaps.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from math import atan2, hypot, pi
from typing import Iterable

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class VectorConfig:
    min_segment_points: int = 9
    min_segment_length: float = 22.0
    gap_split_px: float = 32.0
    max_spatial_jump_px: float = 32.0
    branch_angle_deg: float = 72.0
    branch_window: int = 4
    spline_smooth_per_point: float = 2.0
    robust_iterations: int = 3
    robust_huber_threshold_px: float = 3.2
    max_median_residual_px: float = 2.8
    max_p90_residual_px: float = 5.7
    photo_distance_px: float = 2.2
    source_nearest_sample_px: float = 4.5
    min_photo_support_ratio: float = .25
    max_proposed_gap_px: float = 55.0
    min_endpoint_alignment_cos: float = .91
    max_rival_close_px: float = 3.0
    min_join_margin: float = .13


def _path_length(points):
    if len(points)<2: return 0.0
    return float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())


def separate_at_discontinuities(points: np.ndarray, cfg: VectorConfig):
    """Reject discontinuous runs instead of forcing a global parabola.

    Severe sudden turns are potential branch jumps; gently accumulated turns
    of a spiral are allowed. Preserve segment order and orientation.
    """
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2:raise ValueError('expected n x 2 points')
    if len(points)<cfg.min_segment_points:return [], {'too_short_input': True,'branch_cuts':0,'gap_cuts':0}
    d=np.linalg.norm(np.diff(points,axis=0),axis=1)
    discontinuities=set(int(i+1) for i in np.where(d>cfg.gap_split_px)[0])
    gap_count=len(discontinuities)
    branch_count=0
    n=len(points)
    w=cfg.branch_window
    # Local arc tangents. Avoid treating a gradual spiral turn as a sharp X.
    for j in range(w,n-w):
        if j in discontinuities:continue
        va=points[j]-points[j-w]
        vb=points[j+w]-points[j]
        na=np.linalg.norm(va);nb=np.linalg.norm(vb)
        if na<3 or nb<3:continue
        angle=np.rad2deg(np.arccos(np.clip(float(np.dot(va,vb)/(na*nb)),-1,1)))
        if angle>cfg.branch_angle_deg:
            if not any(abs(j-c)<=w for c in discontinuities):
                discontinuities.add(j)
                branch_count+=1
    boundaries=[0]+sorted(discontinuities)+[n]
    pieces=[]
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        q=points[a:b]
        # Drop duplicates (e.g., stagnation at a junction) before curve fit.
        if len(q)>0:
            new=[q[0]]
            for p in q[1:]:
                if float(np.linalg.norm(p-new[-1]))>.35:new.append(p)
            q=np.asarray(new)
        if len(q)>=cfg.min_segment_points and _path_length(q)>=cfg.min_segment_length:
            pieces.append((a,b,q))
    return pieces, {'too_short_input':False,'branch_cuts':branch_count,'gap_cuts':gap_count,
                    'pieces_accepted':len(pieces)}


def robust_cubic_spline(points:np.ndarray,cfg:VectorConfig=VectorConfig()):
    """Fit a cubic parametric B-spline to *ordered* source observations.

    Robust IRLS reduces point-spikes, but cannot discover the right track ID.
    Output sample and actual FITPACK (knots, coeffs, degree), not a mask.
    """
    x=np.asarray(points,float)
    if x.ndim!=2 or x.shape[1]!=2 or len(x)<cfg.min_segment_points:
        raise ValueError('insufficient ordered two-dimensional source points')
    keep=np.r_[True,np.linalg.norm(np.diff(x,axis=0),axis=1)>.4]
    x=x[keep]
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(x,axis=0),axis=1))]
    if arc[-1]<cfg.min_segment_length:raise ValueError('path length too small')
    t=arc/arc[-1]
    w=np.ones(len(x),dtype=float)
    tck=None
    for _ in range(cfg.robust_iterations):
        # Smooth per sample in pixel units. The result is cubic, not a global
        # parabola. Penalize outlier bubble points in later IRLS rounds.
        smooth=float(cfg.spline_smooth_per_point*len(x))
        tck,_=splprep([x[:,0],x[:,1]],u=t,w=w,s=smooth,k=3,per=False)
        pred=np.stack(splev(t,tck),axis=-1)
        residual=np.linalg.norm(pred-x,axis=1)
        w=np.clip(cfg.robust_huber_threshold_px/np.maximum(residual,1e-5),.22,1.)
    pred=np.stack(splev(t,tck),axis=-1)
    residual=np.linalg.norm(pred-x,axis=1)
    med=float(np.median(residual));p90=float(np.quantile(residual,.9))
    # Do not claim that a smooth but grossly off-line curve is a valid fit.
    accepted=(med<=cfg.max_median_residual_px and p90<=cfg.max_p90_residual_px)
    samplecount=int(np.clip(np.ceil(arc[-1]*1.5),64,2500))
    ts=np.linspace(0,1,samplecount)
    sampled=np.stack(splev(ts,tck),axis=-1)
    derivative=np.stack(splev(ts,tck,der=1),axis=-1)
    return {'tck':tck,'points':sampled,'parameter':ts,'derivative':derivative,
            'observed_xy':x,'fit_median_residual_px':med,'fit_p90_residual_px':p90,
            'length_source_px':float(arc[-1]),'length_spline_px':_path_length(sampled),
            'fit_accepted':bool(accepted)}


def source_support(sampled,observed,source_mask,blocked,cfg=VectorConfig()):
    """Photo-backed samples require BOTH an existing source ridge pixel and
    proximity to the original assigned ordered observations of this track.
    Otherwise the spline is a visually interpolated hypothesis, NOT source.
    """
    points=np.asarray(sampled,float)
    obs=np.asarray(observed,float)
    valid=np.asarray(source_mask,bool)&~np.asarray(blocked,bool)
    if source_mask.shape!=blocked.shape:raise ValueError('mask dimensions mismatch')
    h,w=valid.shape
    xs=np.rint(points[:,0]).astype(int);ys=np.rint(points[:,1]).astype(int)
    inside=(xs>=0)&(xs<w)&(ys>=0)&(ys<h)
    xs=np.clip(xs,0,w-1);ys=np.clip(ys,0,h-1)
    dimage=distance_transform_edt(~valid)
    from_photo=dimage[ys,xs]<=cfg.photo_distance_px
    nearest=cKDTree(obs).query(points)[0]
    observed_sample=inside&from_photo&(nearest<=cfg.source_nearest_sample_px)&~blocked[ys,xs]
    return observed_sample, {'supported_sample_fraction':round(float(observed_sample.mean()),4),
                            'supported_sample_count':int(observed_sample.sum()),
                            'inferred_sample_count':int((~observed_sample).sum()),
                            'not_expert_ground_truth':True}


def bezier_from_tck(tck):
    """Exact cubic Bezier segmentation of the spline's nonzero knot spans."""
    knots=np.asarray(tck[0]);p=int(tck[2]);
    if p!=3:raise ValueError('expected cubic spline')
    spans=[]
    for a,b in zip(np.unique(knots)[:-1],np.unique(knots)[1:]):
        if b-a<1e-10 or a<0-1e-6 or b>1+1e-6:continue
        p0=np.array(splev(float(a),tck));p3=np.array(splev(float(b),tck))
        v0=np.array(splev(float(a),tck,der=1));v1=np.array(splev(float(b),tck,der=1))
        d=float(b-a)
        spans.append([p0.tolist(),(p0+v0*d/3).tolist(),(p3-v1*d/3).tolist(),p3.tolist()])
    return spans


def svg_path(bezier):
    if not bezier:return ''
    f=lambda p:f'{p[0]:.3f},{p[1]:.3f}'
    s='M '+f(bezier[0][0])
    for row in bezier:s+=' C '+f(row[1])+' '+f(row[2])+' '+f(row[3])
    return s


def vectorize(path_id, observed_points,source_mask,blocked,cfg=VectorConfig(),gray=None):
    pieces,audit=separate_at_discontinuities(np.asarray(observed_points,float),cfg)
    candidates=[]
    for ii,(start,end,q) in enumerate(pieces):
        refinereport=None
        if gray is not None:
            q,refinereport=refine_dark_centers_subpixel(gray,q)
        try:fit=robust_cubic_spline(q,cfg)
        except (ValueError,TypeError,ZeroDivisionError):continue
        source,quality=source_support(fit['points'],fit['observed_xy'],source_mask,blocked,cfg)
        b=bezier_from_tck(fit['tck'])
        approved=fit['fit_accepted'] and quality['supported_sample_fraction']>=cfg.min_photo_support_ratio
        item={'curve_id':f'{path_id}_piece{ii}',
            'source_path_id':path_id,'source_point_range':[start,end],
            'fit_type':'cubic_parametric_B_spline','svg_path':svg_path(b),
            'bezier_cubic_spans':b,
            'source_anchored':bool(approved),'points_source_count':len(fit['observed_xy']),
            'subpixel_refinement':refinereport,
            'approx_source_arc_length_px':round(fit['length_source_px'],2),
            'spline_arc_length_px':round(fit['length_spline_px'],2),
            'fit_median_residual_px':round(fit['fit_median_residual_px'],3),
            'fit_p90_residual_px':round(fit['fit_p90_residual_px'],3),
            'photo_support':quality,
            'sampled_xy':np.round(fit['points'],2).tolist(),
            'sampled_source_supported':source.tolist(),
            'endpoints':[np.round(fit['points'][0],3).tolist(),np.round(fit['points'][-1],3).tolist()],
            'status':'SOURCE_CONSTRAINED_VISUAL_TRACK_CANDIDATE' if approved else 'REVIEW_ONLY_UNCERTAIN_FIT',
            'NOT_PHYSICAL_PARTICLE_ID':True}
        candidates.append(item)
    return candidates,audit


def _tangent_at_endpoint(z,end):
    xy=np.asarray(z['sampled_xy'],float)
    d=xy[0]-xy[min(len(xy)-1,8)] if end==0 else xy[-1]-xy[max(0,len(xy)-9)]
    n=np.linalg.norm(d)
    return d/max(n,1e-9)


def tentative_links(curves,photo,blocked,cfg=VectorConfig()):
    """Suggest *reversible* track association; never merge by endpoints alone.
    Require tangent agreement, source evidence, rival exclusion, and margin.
    """
    ports=[]
    good=[c for c in curves if c['source_anchored']]
    for k,c in enumerate(good):
        for end in (0,1):
            pos=np.asarray(c['endpoints'][end],float)
            ports.append((k,end,pos,_tangent_at_endpoint(c,end)))
    if len(ports)<2:return [],{'ports':len(ports),'pairs':0,'proposed':0}
    tree=cKDTree(np.array([q[2] for q in ports]))
    proposals=[]
    h,w=photo.shape
    photo_distance=distance_transform_edt(~np.asarray(photo,bool))
    rival_trees=[]
    for c in good:
        rival_trees.append(cKDTree(np.asarray(c['sampled_xy'],float)[::5]))
    for a,b in sorted(tree.query_pairs(cfg.max_proposed_gap_px)):
        ia,ea,pa,ta=ports[a];ib,eb,pb,tb=ports[b]
        if ia==ib:continue
        delta=pb-pa;length=np.linalg.norm(delta)
        if length<3:continue
        axis=delta/length
        ca=float(ta@axis);cb=float(tb@(-axis))
        if min(ca,cb)<cfg.min_endpoint_alignment_cos:continue
        path=np.linspace(pa,pb,max(10,int(np.ceil(length*2))))
        coords=np.rint(path).astype(int)
        if np.any(coords[:,0]<0)|np.any(coords[:,0]>=w)|np.any(coords[:,1]<0)|np.any(coords[:,1]>=h):continue
        xx,yy=coords[:,0],coords[:,1]
        if blocked[yy,xx].any():continue
        # Each proposed gap may be unsupported, but long gaps must not become
        # decorative connections. Require >25% nearby actual source signal.
        support=float(np.mean(photo_distance[yy,xx]<=2.5))
        if length>15 and support<.27:continue
        # Reject proposed jumps that traverse an already assigned *rival* curve
        # with a crossing; near rivals create an ambiguity rather than merge.
        # Candidate count is small but avoid allocating a global rival array.
        clash=False
        for i,ref_tree in enumerate(rival_trees):
            if i in (ia,ib):continue
            if np.min(ref_tree.query(path)[0])<cfg.max_rival_close_px:
                clash=True;break
        if clash:continue
        score=.45*min(ca,cb)+.4*support+.15*(1-length/cfg.max_proposed_gap_px)
        proposals.append({'a':a,'b':b,'curve_ids':[good[ia]['curve_id'],good[ib]['curve_id']],
                          'gap_px':round(float(length),2),'photo_corridor_support':round(support,3),
                          'score':round(float(score),4),
                          'inferred_link_xy':np.round(path,2).tolist(),
                          'status':'REVIEW_ONLY_NOT_VALIDATED_TRACK_IDENTITY'})
    # No automatically accepted connections in a crowded crossing. One
    # candidate per endpoint only when its score clearly beats alternatives.
    scores={}
    for edge in proposals:
        for port in (edge['a'],edge['b']):scores.setdefault(port,[]).append(edge['score'])
    for edge in proposals:
        ambiguous=any(len(scores[e])>1 and sorted(scores[e],reverse=True)[0]-sorted(scores[e],reverse=True)[1]<cfg.min_join_margin
                      for e in (edge['a'],edge['b']))
        mutual_best=all(edge['score']==max(scores[e]) for e in (edge['a'],edge['b']))
        if ambiguous: edge['decision']='ABSTAIN_COMPETING_CONTINUATION'
        elif not mutual_best: edge['decision']='ABSTAIN_NOT_MUTUAL_BEST'
        else: edge['decision']='TENTATIVE_FOR_EXPERT_REVIEW'
        edge.pop('a');edge.pop('b')
    return proposals,{'ports':len(ports),'pairs':len(proposals),
                      'tentative_for_review':sum(e['decision']=='TENTATIVE_FOR_EXPERT_REVIEW' for e in proposals),
                      'not_assigned_physical_identity':True}


def refine_dark_centers_subpixel(gray:np.ndarray,points:np.ndarray,
                                 max_shift_px:float=1.7):
    """Refine ordered ridge seeds across the local normal, not along the track.

    For isolated dark curves the intensity minimum typically lies BETWEEN
    pixels. Bilinear sampling plus constrained weighted-centroid yields a
    subpixel track center. Broad/ambiguous profiles, weak contrast and border
    points are left unchanged. A crossing is not resolved by this operation.
    """
    if gray.ndim!=2 or gray.dtype!=np.uint8:raise ValueError('uint8 grayscale photo needed')
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<9:raise ValueError('need ordered 2D centers')
    n=len(p);h,w=gray.shape
    radius=4
    a=np.maximum(0,np.arange(n)-radius)
    b=np.minimum(n-1,np.arange(n)+radius)
    tangent=p[b]-p[a]
    norm=np.linalg.norm(tangent,axis=1)
    tangent=tangent/np.maximum(norm[:,None],1e-6)
    normal=np.column_stack((-tangent[:,1],tangent[:,0]))
    offset=np.arange(-3.,3.01,.5)
    positions=p[:,None,:]+offset[None,:,None]*normal[:,None,:]
    coords_x=positions[...,0].astype(np.float32)
    coords_y=positions[...,1].astype(np.float32)
    samples=cv2.remap(gray.astype(np.float32),coords_x,coords_y,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    # Local continuum estimated from side bands, robust to small dark beads.
    bg=np.maximum(samples[:,0],samples[:,-1])
    depth=np.maximum(0.,bg[:,None]-samples)
    weight=depth**1.7
    mass=weight.sum(axis=1)
    mu=(weight*offset[None,:]).sum(axis=1)/np.maximum(1e-9,mass)
    sigma=np.sqrt((weight*(offset[None,:]-mu[:,None])**2).sum(axis=1)/np.maximum(mass,1e-9))
    # Avoid following ambiguous transverse crossings and random dark dust.
    enough=(bg-np.min(samples,axis=1))>=5.0
    good=(enough&(sigma<=1.75)&(np.abs(mu)<=max_shift_px)&(norm>.4)&
          (p[:,0]>=4)&(p[:,0]<w-4)&(p[:,1]>=4)&(p[:,1]<h-4))
    result=p.copy()
    result[good]+=mu[good,None]*normal[good]
    shifted=np.linalg.norm(result-p,axis=1)
    gray_new=cv2.remap(gray.astype(np.float32),result[:,0:1].astype(np.float32),result[:,1:2].astype(np.float32),cv2.INTER_LINEAR).reshape(-1)
    gray_old=cv2.remap(gray.astype(np.float32),p[:,0:1].astype(np.float32),p[:,1:2].astype(np.float32),cv2.INTER_LINEAR).reshape(-1)
    return result,{'candidate_points':n,'subpixel_refined_points':int(good.sum()),
                   'average_refined_displacement_px':round(float(shifted[good].mean()) if good.any() else 0.,3),
                   'mean_photo_intensity_improvement':round(float((gray_old[good]-gray_new[good]).mean()) if good.any() else 0.,3),
                   'not_GT_and_not_validated_physical_identity':True}