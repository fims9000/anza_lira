"""S111: long-gap photo checks and subpixel *inferred* Bézier refinement.

All original S109 cubic fragment geometry, source pixels and IDs are immutable.
Source-guided connector is STILL an unobserved hypothesis, not a photographed
line, even where individual bubble dots lie near it.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from .s110_global_association import (
    GlobalConfig, generate_candidates, prior_review_edges, endpoint_outward, sample_bezier, _make_tracks,
)

from .s111_conflict_optimizer import solve_exact

@dataclass(frozen=True)
class PhotoGuideConfig:
    long_gap_start_px: float = 80.0
    long_gap_required_support: float = .65
    long_gap_min_dark_side_difference: float = 4.0
    long_gap_min_side_support_margin: float = .18
    side_offsets_px: tuple[float, ...] = (-12.0, -8.0, 8.0, 12.0)
    max_shift_px: float = 2.5
    shift_step_px: float = .5
    max_neighbor_step_px: float = 1.0
    offset_penalty: float = .22
    transition_penalty: float = 1.10
    handle_regularization: float = .04
    max_handle_fraction: float = .62
    min_handle_fraction: float = .16
    min_improvement_photo_score: float = .03
    min_photo_score_after: float = .25
    max_photo_support_drop: float = .018
    max_centerline_shift_rms: float = 2.0
    confidence_reference_gap_px: float = 140.0


def _remap(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    xy=np.asarray(xy,np.float32)
    return cv2.remap(image.astype(np.float32),xy[...,0],xy[...,1],
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0.0)


def _trace_geometry(bezier: list, n: int):
    p=sample_bezier(bezier,n)
    derivative=np.gradient(p,axis=0)
    denom=np.maximum(np.linalg.norm(derivative,axis=1,keepdims=True),1e-6)
    tangent=derivative/denom
    normal=np.stack((-tangent[:,1],tangent[:,0]),axis=-1)
    return p,tangent,normal


def _score_path(path, dark, distance, width):
    p=np.asarray(path,np.float32)
    inrange=((p[:,0]>=2)&(p[:,0]<width[1]-2)&(p[:,1]>=2)&(p[:,1]<width[0]-2))
    s_dark=_remap(dark,p.reshape(-1,1,2)).reshape(-1)
    s_dist=_remap(distance,p.reshape(-1,1,2)).reshape(-1)
    # A blend of source-specific ridge support and actual grayscale contrast.
    # This is a *relative* visual alignment proxy, NOT true track precision.
    reward=(np.clip(s_dark,-5,35)*.06+
            np.clip(3.0-s_dist,0,3.0)*.60)
    reward[~inrange]=-100.
    return float(np.mean(reward)), float(np.mean(s_dist<=2.5)), float(np.mean(s_dark))


def long_gap_guard(edge: dict, dark, source_dist, normal, cfg: PhotoGuideConfig):
    path=np.asarray(edge['sampled_xy'],float)
    if len(path)!=len(normal):
        derivative=np.gradient(path,axis=0)
        den=np.maximum(np.linalg.norm(derivative,axis=1,keepdims=True),1e-6)
        normal=np.column_stack((-derivative[:,1]/den[:,0], derivative[:,0]/den[:,0]))
    if edge['gap_px']<=cfg.long_gap_start_px:
        return True,{'status':'SHORT_GAP_EXISTING_S110_CHECKS','long_gap':False}
    side=[]
    for offset in cfg.side_offsets_px:
        displaced=path+offset*normal
        photo_distance=_remap(source_dist,displaced.astype(np.float32).reshape(-1,1,2)).reshape(-1)
        side.append(float(np.mean(photo_distance<=2.5)))
    center_dist=_remap(source_dist,path.astype(np.float32).reshape(-1,1,2)).reshape(-1)
    center_support=float(np.mean(center_dist<=2.5))
    # The S110 dark-minus-side metric is independent evidence; a path alongside
    # a neighboring line cannot be auto-accepted just for matching tangents.
    side_margin=center_support-max(side,default=0.0)
    passed=(center_support>=cfg.long_gap_required_support and
            side_margin>=cfg.long_gap_min_side_support_margin and
            edge.get('source_dark_minus_side_mean',0)>=cfg.long_gap_min_dark_side_difference)
    return passed,{'status':'LONG_GAP_SIDE_BAND_PASS' if passed else 'LONG_GAP_AMBIGUOUS_PHOTO',
                   'long_gap':True,'central_source_fraction':round(center_support,4),
                   'side_source_fractions':[round(v,4) for v in side],
                   'best_side_margin':round(side_margin,4),
                   'dark_to_neighbor_margin':edge.get('source_dark_minus_side_mean')}


def _fit_anchored_cubic(points, p0, p1, ta, tb, cfg):
    """Least-squares handle-length fit constrained to fixed endpoint tangents.

    Only two scalar handle lengths are fitted, not arbitrary control points.
    Hence endpoints and the tangent *directions* are invariant. Any remaining
    unobserved area is a geometrical hypothesis with explicit provenance.
    """
    samples=np.asarray(points,float)
    n=len(samples)
    t=np.linspace(0,1,n)
    u=1-t
    a=3*t*u*u
    b=3*t*t*u
    p0=np.asarray(p0,float);p1=np.asarray(p1,float)
    ta=np.asarray(ta,float);tb=np.asarray(tb,float)
    baseline=((u**3+3*u*u*t)[:,None]*p0 + (3*t*t*u+t**3)[:,None]*p1)
    # Cubic bridge controls B1=p0+ta*h0, B2=p1+tb*h1.
    # In the original S110 `tb` is OUTWARD at destination, thus derivative at
    # the end equals -tb. This is deliberately consistent with S110.
    X=np.stack((a[:,None]*ta,b[:,None]*tb),axis=-1).reshape(-1,2)
    Y=(samples-baseline).reshape(-1)
    gap=float(np.linalg.norm(p1-p0))
    h0=gap*.33
    reg=np.sqrt(cfg.handle_regularization)
    A=np.vstack((X,reg*np.eye(2)))
    yy=np.r_[Y,reg*np.array([h0,h0])]
    h=np.linalg.lstsq(A,yy,rcond=None)[0]
    h=np.clip(h,cfg.min_handle_fraction*gap,cfg.max_handle_fraction*gap)
    bez=[p0.tolist(),(p0+ta*h[0]).tolist(),(p1+tb*h[1]).tolist(),p1.tolist()]
    return bez


def refine_bezier(edge, curves_by_id, dark, photo_dist, cfg=PhotoGuideConfig()):
    rec=dict(edge)
    src=np.asarray(edge['bezier_cubic'],float)
    n=max(48,int(ceil(edge['gap_px']*2.1)))
    path,tangent,normal=_trace_geometry(src,n)
    offsets=np.arange(-cfg.max_shift_px,cfg.max_shift_px+cfg.shift_step_px/4,
                      cfg.shift_step_px,dtype=np.float32)
    k=len(offsets); zero=int(np.argmin(np.abs(offsets)))
    candidate=(path[:,None,:]+offsets[None,:,None]*normal[:,None,:]).astype(np.float32)
    vals_dark=_remap(dark,candidate)
    vals_dist=_remap(photo_dist,candidate)
    gain=np.clip(vals_dark,-5,35)*.06+np.clip(3.0-vals_dist,0,3.0)*.60
    gain=gain-cfg.offset_penalty*offsets[None,:]**2
    # Do not move endpoints; do not leave input crop; don't chase far source.
    H,W=dark.shape
    outside=(candidate[...,0]<2)|(candidate[...,0]>=W-2)|(candidate[...,1]<2)|(candidate[...,1]>=H-2)
    gain[outside]=-1e6
    dp=np.full((n,k),-1e10,np.float64);back=np.zeros((n,k),np.int16)
    dp[0,zero]=gain[0,zero]
    step=np.abs(offsets[:,None]-offsets[None,:])
    penalty=cfg.transition_penalty*step**2
    penalty[step>cfg.max_neighbor_step_px]=1e7
    for i in range(1,n):
        choices=dp[i-1,:,None]-penalty
        index=np.argmax(choices,axis=0)
        dp[i]=gain[i]+choices[index,np.arange(k)]
        back[i]=index.astype(np.int16)
    idx=np.zeros(n,np.int32);idx[-1]=zero
    for i in range(n-1,0,-1):idx[i-1]=back[i,idx[i]]
    trace=candidate[np.arange(n),idx]
    rms=float(np.sqrt(np.mean(offsets[idx]**2)))
    p0=src[0];p1=src[-1]
    byid=curves_by_id
    a_id,b_id=edge['curve_ids']
    ea,eb=edge['ends']
    ta=endpoint_outward(byid[a_id],ea)
    tb=endpoint_outward(byid[b_id],eb)
    fitted=_fit_anchored_cubic(trace,p0,p1,ta,tb,cfg)
    old_score,old_fraction,old_dark=_score_path(path,dark,photo_dist,dark.shape)
    path_new=sample_bezier(fitted,n)
    new_score,new_fraction,new_dark=_score_path(path_new,dark,photo_dist,dark.shape)
    displacement=np.linalg.norm(path_new-path,axis=1)
    # Reject a fit if it is not demonstrably more source-aligned or it drifts.
    accepted=(rms<=cfg.max_centerline_shift_rms and
              float(displacement.max())<=cfg.max_shift_px and
              new_fraction+cfg.max_photo_support_drop>=old_fraction and
              new_score-old_score>=cfg.min_improvement_photo_score)
    rec['refinement']={'status':'PHOTO_ALIGNED_CUBIC_REFINED' if accepted else 'ORIGINAL_S110_BEZIER_RETAINED',
                       'mean_original_photo_score':round(old_score,4),
                       'mean_proposed_photo_score':round(new_score,4),
                       'source_support_before':round(old_fraction,4),
                       'source_support_after':round(new_fraction,4),
                       'mean_dark_before':round(old_dark,3),
                       'mean_dark_after':round(new_dark,3),
                       'tracer_offset_rms_px':round(rms,3),
                       'cubic_refit_max_displacement_px':round(float(displacement.max()),3),
                       'source_pixels_added_by_refinement':0,
                       'connector_remains_INFERRED_NOT_OBSERVED':True}
    rec['original_S110_bezier_cubic']=edge['bezier_cubic']
    if accepted:
        rec['bezier_cubic']=[[round(float(a),4) for a in p] for p in fitted]
        rec['sampled_xy']=np.round(path_new,2).tolist()
    return rec


def associate_s111(curves,photo,blocked,gray,expert_decisions=None,legacy_review=None,
                   global_cfg=None,guide_cfg=PhotoGuideConfig()):
    cfg=global_cfg or GlobalConfig(max_gap_px=140.,min_endpoint_cos=.95,
                      min_photo_support_long=.31,min_photo_support_very_long=.65,
                      auto_candidate_score=.83)
    cands,gdiag=generate_candidates(curves,photo,blocked,gray,cfg)
    # IMPORTANT: S110 used gap/max_gap_px as a score feature. Raising the
    # search radius therefore raised *all* scores and could auto-accept a
    # connection solely because the user changed a search parameter.
    # Recompute that single score term against a fixed 140px reference while
    # preserving the original score for a full audit.
    for edge in cands:
        length=float(edge['gap_px'])
        source_length_score=max(0.,1.-length/cfg.max_gap_px)
        calibrated_length_score=max(0.,1.-length/guide_cfg.confidence_reference_gap_px)
        edge['score_before_search_radius_calibration']=edge['score']
        edge['score']=round(float(edge['score'])+.14*(calibrated_length_score-source_length_score),5)
        edge['search_radius_score_calibrated']=True
    # Evaluate every extra-long continuation BEFORE matching, not after it.
    # Otherwise the solver could reserve the endpoint for an unsupported link
    # and then discard that link without reconsidering a better alternative.
    f=gray.astype(np.float32)
    dark=cv2.GaussianBlur(f,(0,0),5)-cv2.GaussianBlur(f,(0,0),.8)
    photo_dist=distance_transform_edt(~(np.asarray(photo,bool)&~np.asarray(blocked,bool))).astype(np.float32)
    long_filtered=0
    for edge in cands:
        if edge['gap_px']<=guide_cfg.long_gap_start_px:continue
        path=np.asarray(edge['sampled_xy'],float)
        derivative=np.gradient(path,axis=0)
        norm=np.maximum(np.linalg.norm(derivative,axis=1,keepdims=True),1e-6)
        normal=np.column_stack((-derivative[:,1]/norm[:,0],derivative[:,0]/norm[:,0]))
        allowed,guard=long_gap_guard(edge,dark,photo_dist,normal,guide_cfg)
        edge['long_gap_guard']=guard
        if not allowed:
            edge['auto_eligible']=False
            edge['s111_pre_matching_review_reason']='LONG_GAP_SIDE_BAND_UNSUPPORTED'
            long_filtered+=1
    legacy=prior_review_edges(curves,legacy_review or [],cands)
    cands.extend(legacy)
    cands.sort(key=lambda x:(-x['score'],x['candidate_id']))
    gdiag['legacy_s109_review_only_edges_added']=len(legacy)
    gdiag['long_gap_pre_matching_review_only_edges']=long_filtered
    scored,opt_diag=solve_exact(curves,cands,cfg,expert_decisions)
    tracks=_make_tracks(curves,scored)
    count={'fragments_in':len(curves),'global_track_candidates':len(tracks),
           'groups_with_multiple_fragments':sum(len(t['member_curve_ids'])>1 for t in tracks),
           'joined_curve_fragments':sum(len(t['member_curve_ids']) for t in tracks if len(t['member_curve_ids'])>1),
           'selected_link_count':sum(t['inferred_connector_count'] for t in tracks),
           'unresolved_candidate_pairs':sum(e['decision'] not in ('ALGORITHMIC_GROUPING_NOT_PHYSICAL_GT','EXPERT_CONFIRMED_SAME') for e in scored),
           'not_verified_true_particle_tracks':True}
    data={'tracks':tracks,'edge_proposals':scored,'association_report':count,
          'generator_audit':gdiag,'solver_audit':opt_diag}
    byid={c['curve_id']:c for c in curves}
    results=[];refined=0
    for edge in data['edge_proposals']:
        rec=dict(edge)
        selected=rec['decision'] in ('ALGORITHMIC_GROUPING_NOT_PHYSICAL_GT','EXPERT_CONFIRMED_SAME')
        if selected:
            if rec['gap_px']>=14:
                rec=refine_bezier(rec,byid,dark,photo_dist,guide_cfg)
                if rec['refinement']['status']=='PHOTO_ALIGNED_CUBIC_REFINED':refined+=1
            else:
                rec['refinement']={'status':'TOO_SHORT_FOR_REFIT',
                                   'connector_remains_INFERRED_NOT_OBSERVED':True}
        results.append(rec)
    data['tracks']=_make_tracks(curves,results)
    data['edge_proposals']=results
    count=data['association_report']
    count['global_track_candidates']=len(data['tracks'])
    count['selected_link_count']=sum(len(t['inferred_connectors']) for t in data['tracks'])
    count['groups_with_multiple_fragments']=sum(len(t['member_curve_ids'])>1 for t in data['tracks'])
    count['joined_curve_fragments']=sum(len(t['member_curve_ids']) for t in data['tracks'] if len(t['member_curve_ids'])>1)
    count['unresolved_candidate_pairs']=sum(e['decision'] not in ('ALGORITHMIC_GROUPING_NOT_PHYSICAL_GT','EXPERT_CONFIRMED_SAME') for e in results)
    data['s111_photo_guide_report']={
        'long_gap_candidates_screened_before_global_solve':long_filtered,
        'long_gap_rejected_ids':[e['candidate_id'] for e in cands if e.get('s111_pre_matching_review_reason')],
        'photo_guided_cubic_refinements':refined,
        'candidate_edges_total':len(results),
        'original_source_s109_curves_unchanged':True,
        'no_hidden_neural_training_or_GPU_inference':True,
        'physical_particle_identity_not_expert_verified':True}
    return data