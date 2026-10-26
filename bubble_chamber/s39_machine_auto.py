"""S39. Machine-only discovery of H50Z ridge hypotheses; never particle ground truth.

All thresholds and score semantics are fixed and explicit. Other photo's
machine-mask-trained AZUNet is an independent *model feature*, not an oracle.
Unknown pixels remain 255. Geometry follows local curves; Belyaev's hadroide
is intentionally NOT applied universally outside a certified stopping track.
"""
from __future__ import annotations

from collections import deque
from typing import Any
import cv2
import numpy as np
from scipy.ndimage import convolve
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

SCALES=(1.1,2.2,3.5)
SCHEMA='s39_machine_only_photo_hypotheses_v1'


def ridge_evidence(gray: np.ndarray) -> dict[str,np.ndarray]:
    """Dark, anisotropic, curved ridge score; no global constant brightness."""
    if gray.ndim!=2 or gray.dtype!=np.uint8 or min(gray.shape)<96:
        raise ValueError('Expected raw uint8 real/photo-backed grayscale')
    a=gray.astype(np.float32)/255.
    # A broad local intensity baseline compensates exposure, not geometry.
    darkness=np.maximum(cv2.GaussianBlur(a,(0,0),7.)-a,0.).astype('float32')
    responses=[]
    for sigma in SCALES:
        blurred=cv2.GaussianBlur(a,(0,0),sigma)
        xx=cv2.Sobel(blurred,cv2.CV_32F,2,0,ksize=3)*(sigma*sigma)
        yy=cv2.Sobel(blurred,cv2.CV_32F,0,2,ksize=3)*(sigma*sigma)
        xy=cv2.Sobel(blurred,cv2.CV_32F,1,1,ksize=3)*(sigma*sigma)
        radius=np.sqrt((xx-yy)**2+4*xy**2)
        major=(xx+yy+radius)*.5
        minor=(xx+yy-radius)*.5
        # A line valley has positive normal curvature and weak tangent curvature;
        # blob-like isotropic pits are discounted, not declared impossible tracks.
        anisotropy=np.exp(-.5*(np.abs(minor)/(np.abs(major)+1e-6)/.60)**2)
        score=np.maximum(major,0.)*anisotropy*np.sqrt(np.clip(darkness/.05,0,1))
        responses.append(score.astype('float32'))
    stack=np.stack(responses)
    merged=stack.max(0)
    if not np.isfinite(merged).all():
        raise ValueError('Invalid ridge map')
    return {'score':merged,'scales':stack,'darkness':darkness}


def fixed_budget_mask(score:np.ndarray, *, fraction:float=.035) -> np.ndarray:
    """At most the highest fraction of map pixels: UNSUPERVISED activation budget."""
    if score.ndim!=2 or not np.isfinite(score).all() or not .001<=fraction<=.25:
        raise ValueError('Invalid machine score / activation fraction')
    n=max(1,int(np.ceil(round(score.size*fraction,8))))
    flat=score.reshape(-1)
    if np.min(flat)==np.max(flat):
        return np.zeros_like(score,dtype=bool)  # No spatial evidence to rank.
    kth=float(np.partition(flat,-n)[-n])
    chosen=(flat>kth)
    left=n-int(chosen.sum())
    if kth>0 and left>0:
        # Stable coordinate-order tie-break, exactly equal budgets on valid maps.
        ties=np.flatnonzero(flat==kth)
        chosen[ties[:left]]=True
    return chosen.reshape(score.shape)


def raw_detector_mask(evidence:dict, *,fraction:float=.035) -> np.ndarray:
    """Hessian line positives require actual local darkening and exclude edges."""
    score=evidence['score'];dark=evidence['darkness']
    mask=fixed_budget_mask(score,fraction=fraction)&(dark>=.012)
    mask[:7,:]=False;mask[-7:,:]=False;mask[:,:7]=False;mask[:,-7:]=False
    return mask


def canny_score(gray:np.ndarray) -> np.ndarray:
    """Sobel gradient baseline (edge evidence rather than centerline)."""
    sm=cv2.GaussianBlur(gray.astype('float32')/255.,(0,0),1.1)
    gx=cv2.Sobel(sm,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(sm,cv2.CV_32F,0,1,ksize=3)
    return cv2.magnitude(gx,gy)


def contrast_baseline(gray:np.ndarray)->np.ndarray:
    a=gray.astype('float32')/255.
    return np.maximum(cv2.GaussianBlur(a,(0,0),7.)-a,0.)


def _trace_component(mask:np.ndarray)->np.ndarray|None:
    """Geodesic diameter of a junction-cut 8-connected skeleton component."""
    ys,xs=np.nonzero(mask)
    if len(xs)<18:return None
    points=list(zip(xs.tolist(),ys.tolist()))
    lookup={p:i for i,p in enumerate(points)}
    neighbors=[[] for _ in points]
    for i,(x,y) in enumerate(points):
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx==dy==0:continue
                j=lookup.get((x+dx,y+dy))
                if j is not None:neighbors[i].append(j)
    def bfs(start:int):
        d={start:0};parents={start:None};q=deque([start]);far=start
        while q:
            node=q.popleft()
            if d[node]>d[far]:far=node
            for nxt in neighbors[node]:
                if nxt not in d:
                    d[nxt]=d[node]+1;parents[nxt]=node;q.append(nxt)
        return far,parents,d
    ends=[i for i,items in enumerate(neighbors) if len(items)<=1]
    start=ends[0] if ends else 0
    candidate,_,_=bfs(start)
    target,parents,_=bfs(candidate)
    route=[];cur=target
    while cur is not None:
        route.append(points[cur]);cur=parents[cur]
    route=np.asarray(route[::-1],dtype='float64')
    if len(route)<19 or np.linalg.norm(route[0]-route[-1])<17:return None
    distances=np.linalg.norm(np.diff(route,axis=0),axis=1)
    u=np.concatenate(([0],np.cumsum(distances)))
    if u[-1]<20:return None
    k=min(96,max(12,int(u[-1]/3)))
    keep=np.unique(u,return_index=True)[1]
    if len(keep)<3:return None
    wanted=np.linspace(0,u[-1],k)
    return np.column_stack((np.interp(wanted,u[keep],route[keep,0]),
                            np.interp(wanted,u[keep],route[keep,1])))


def extract_ridges(evidence:dict,*,max_candidates=32) ->list[np.ndarray]:
    mask=raw_detector_mask(evidence)
    # Small closing connects adjacent bubbles; junction guards prevent a
    # single graph edge from being mislabeled as a completed physical event.
    mask=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))>0
    skel=skeletonize(mask).astype('uint8')
    count_neighbors=convolve(skel,np.ones((3,3),dtype='uint8'),mode='constant')-skel
    junction=(skel>0)&(count_neighbors>=4)
    blocked=cv2.dilate(junction.astype('uint8'),np.ones((3,3),dtype='uint8'))>0
    skel[blocked]=0
    nc,components,stats,_=cv2.connectedComponentsWithStats(skel,connectivity=8)
    pieces=[]
    for i in range(1,nc):
        if stats[i,cv2.CC_STAT_AREA]<18:continue
        x,y,w,h,_=stats[i]
        local=components[y:y+h,x:x+w]==i
        curve=_trace_component(local)
        if curve is None:continue
        curve[:,0]+=x;curve[:,1]+=y
        if np.any(curve<7) or np.any(curve[:,0]>=evidence['score'].shape[1]-7) or np.any(curve[:,1]>=evidence['score'].shape[0]-7):
            continue
        response=float(np.median(sample_bilinear(evidence['score'],curve)))
        pieces.append((response,curve))
    pieces.sort(key=lambda x:x[0],reverse=True)
    return [p for _,p in pieces[:max_candidates]]


def sample_bilinear(arr:np.ndarray,points:np.ndarray)->np.ndarray:
    p=np.asarray(points,dtype='float32')
    return cv2.remap(arr.astype('float32'),p[:,0][None,:],p[:,1][None,:],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)[0]


def line_evidence(gray:np.ndarray,pts:np.ndarray, maps:dict, *, az_map:np.ndarray|None=None,
                  prior_drafts:list[dict]|None=None, own_draft_id:str|None=None)->dict[str,Any]:
    pts=np.asarray(pts,dtype='float64')
    h,w=gray.shape
    if pts.ndim!=2 or pts.shape[1]!=2 or len(pts)<8 or not np.isfinite(pts).all() or np.any(pts<0) or np.any(pts[:,0]>=w) or np.any(pts[:,1]>=h):
        raise ValueError('Invalid candidate source polyline')
    g=np.gradient(pts,axis=0)
    norm=np.maximum(np.linalg.norm(g,axis=1),1e-6)
    normals=np.column_stack((-g[:,1],g[:,0]))/norm[:,None]
    center=sample_bilinear(gray,pts)
    flank=(sample_bilinear(gray,pts+6*normals)+sample_bilinear(gray,pts-6*normals))/2
    contrast=float(np.median(flank-center))
    scale_levels=maps['scales']
    # Compare scale-specific score to its OWN tile distribution, rather than
    # treating the scores across physical image sizes as calibrated probabilities.
    scale_support=[]
    for layer in scale_levels:
        thresh=float(np.percentile(layer,95))
        scale_support.append(float(np.mean(sample_bilinear(layer,pts)>=thresh)) if thresh>0 else 0.)
    az_center=az_delta=None
    if az_map is not None:
        if az_map.shape!=gray.shape or not np.isfinite(az_map).all() or az_map.min()<0 or az_map.max()>1:
            raise ValueError('Invalid AZ evidence')
        az_center=float(np.median(sample_bilinear(az_map,pts)))
        az_flank=(sample_bilinear(az_map,pts+6*normals)+sample_bilinear(az_map,pts-6*normals))/2
        az_delta=float(np.median(sample_bilinear(az_map,pts)-az_flank))
    source_support=[]
    if prior_drafts:
        for draft in prior_drafts:
            if draft.get('draft_id')==own_draft_id:continue
            q=np.asarray(draft['points_xy'],dtype=float)
            support=float(np.mean(cKDTree(q).query(pts)[0]<=4.5))
            if support>=.35:source_support.append((support,draft['source_method'],draft['draft_id']))
    source_support.sort(reverse=True)
    # An opaque review priority, NEVER a physical particle probability.
    agreement=float(source_support[0][0]) if source_support else 0.
    ridge_consensus=float(np.mean(np.asarray(scale_support)>=.35))
    priority=(np.clip(contrast/28.,0,1)*.34 + ridge_consensus*.31 +
              np.clip(max(az_delta or 0.,0.)/.09,0,1)*.15 + agreement*.20)
    return {
        'raw_center_minus_flank_8bit':round(contrast,3),
        'per_scale_top5pct_path_coverage':[round(v,3) for v in scale_support],
        'ridge_scale_consensus_fraction':round(ridge_consensus,3),
        'AZ_cross_photo_center_mean_UNCALIBRATED':round(az_center,4) if az_center is not None else None,
        'AZ_cross_photo_center_vs_flank_UNCALIBRATED':round(az_delta,4) if az_delta is not None else None,
        'S31_other_candidate_overlap_fraction':round(agreement,3),
        'matching_S31_draft_id':source_support[0][2] if source_support else None,
        'review_priority_NOT_PHYSICS_PROBABILITY':round(float(priority),4),
    }


def annotate_tile(gray:np.ndarray, original:list[dict], *,az_map:np.ndarray|None=None,
                  max_new:int=24)->tuple[list[dict],dict]:
    maps=ridge_evidence(gray)
    ridge_new=extract_ridges(maps,max_candidates=max_new*3)
    candidates=[]
    for d in original:
        pts=np.asarray(d['points_xy'],dtype=float)
        cues=line_evidence(gray,pts,maps,az_map=az_map,prior_drafts=original,own_draft_id=d['draft_id'])
        candidates.append({'draft_id':d['draft_id'],'source':'S31_EXISTING_MACHINE_DRAFT',
                           'source_method':d['source_method'],'points_xy':np.round(pts,2).tolist(),**cues})
    added=[]
    for j,pts in enumerate(ridge_new):
        cues=line_evidence(gray,pts,maps,az_map=az_map,prior_drafts=original)
        if cues['S31_other_candidate_overlap_fraction']>=.65:
            continue
        # Filter duplicate ridge fragments among new pieces.
        if any(np.mean(cKDTree(np.asarray(o['points_xy'])).query(pts)[0]<4)>.75 for o in added):
            continue
        added.append({'draft_id':f's39_ridge_{j:03d}','source':'S39_MULTISCALE_RIDGE_PROPOSAL',
                      'source_method':'Hessian_three_scale','points_xy':np.round(pts,2).tolist(),**cues})
    added=sorted(added,key=lambda x:x['review_priority_NOT_PHYSICS_PROBABILITY'],reverse=True)[:max_new]
    candidates.extend(added)
    # Trainable *provisional* machine positives are strictly a subset. No
    # implicit background=0; 255 means unknown/unobserved.
    positive=[c for c in candidates if
              c['raw_center_minus_flank_8bit']>=8 and
              c['ridge_scale_consensus_fraction']>=.34 and
              (c['S31_other_candidate_overlap_fraction']>=.35 or
               (c['AZ_cross_photo_center_vs_flank_UNCALIBRATED'] is not None and
                c['AZ_cross_photo_center_vs_flank_UNCALIBRATED']>=.018))]
    mask=np.full(gray.shape,255,dtype='uint8')
    local_positive=np.zeros_like(mask)
    for c in positive:
        pts=np.rint(np.asarray(c['points_xy'])).astype(np.int32)
        cv2.polylines(local_positive,[pts],False,1,2)
    mask[local_positive>0]=1
    # Conservative background probes only 8-10 pixels from accepted centerline
    # and outside *all* proposed line corridors; false negatives remain possible.
    dilation=cv2.dilate(local_positive,np.ones((21,21),dtype='uint8'))>0
    near=cv2.dilate(local_positive,np.ones((31,31),dtype='uint8'))>0
    all_proposals=np.zeros_like(mask)
    for c in candidates:
        pp=np.rint(np.asarray(c['points_xy'])).astype(np.int32)
        cv2.polylines(all_proposals,[pp],False,1,5)
    far_from_all=cv2.dilate(all_proposals,np.ones((11,11),dtype='uint8'))==0
    probe=(near & ~dilation & far_from_all & (maps['score']<np.percentile(maps['score'],75)))
    mask[probe]=0
    stat={'new_ridge_proposals':len(added),'original_drafts':len(original),
          'machine_consensus_positive_lines':len(positive),
          'positive_pixels_not_gt':int((mask==1).sum()),
          'local_offset_assumed_negative_pixels_not_gt':int((mask==0).sum()),
          'unknown_pixels':int((mask==255).sum()),
          'mean_priority_UNCALIBRATED':round(float(np.mean([c['review_priority_NOT_PHYSICS_PROBABILITY'] for c in candidates])),4) if candidates else None,
          'original_empty_tile_has_new_proposals':bool(not original and added),
          'physics_verified_count':0}
    return candidates,{'stats':stat,'pseudo_mask':mask,'ridge_map':maps['score']}
