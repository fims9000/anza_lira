"""S114 source-first research linelets from original 8-bit bubble chamber photos.

Distinct from a physical track classifier: long directional matched filters favor
persistent dark lines over unstructured grain. Graph spur pruning and B-spline
fitting produce per-fragment candidate geometry. A/B are arbitrary geometric
ends. Unobserved links are NEVER painted as photographed lines.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from typing import Any
import math
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class Config:
    orientation_step_deg: int = 15
    kernel_radius: int = 24
    longitudinal_sigma: float = 13.0
    lateral_sigma: float = 1.45
    flank_distance: float = 4.0
    response_threshold: float = 12.0
    min_binary_object_pixels: int = 30
    prune_spur_length_px: int = 18
    min_path_length_px: float = 45.0
    display_major_length_px: float = 85.0
    max_review_gap_px: float = 100.0
    min_gap_photo_support: float = .40
    gap_photo_threshold: float = 8.0


def directional_response(gray:np.ndarray,cfg:Config=Config())->np.ndarray:
    if gray.ndim!=2 or gray.dtype!=np.uint8:
        raise ValueError('Original 8-bit grayscale photograph required')
    if cfg.orientation_step_deg<1 or cfg.orientation_step_deg>30 or 180%cfg.orientation_step_deg:
        raise ValueError('Orientations must evenly partition 180 degrees')
    if gray.shape[0]<64 or gray.shape[1]<64:
        raise ValueError('Photograph tile too small')
    g=cv2.GaussianBlur(gray.astype(np.float32),(0,0),.7)
    radius=cfg.kernel_radius
    y,x=np.mgrid[-radius:radius+1,-radius:radius+1].astype(np.float32)
    out=np.zeros(gray.shape,np.float32)
    for a in range(0,180,cfg.orientation_step_deg):
        theta=math.radians(a)
        longitudinal=x*math.cos(theta)+y*math.sin(theta)
        lateral=-x*math.sin(theta)+y*math.cos(theta)
        sig=cfg.lateral_sigma; offset=cfg.flank_distance
        center=np.exp(-lateral*lateral/(2*sig*sig))
        shoulders=.5*(np.exp(-(lateral-offset)**2/(2*sig*sig))+
                       np.exp(-(lateral+offset)**2/(2*sig*sig)))
        window=np.exp(-longitudinal*longitudinal/(2*cfg.longitudinal_sigma**2))
        kernel=(center-shoulders)*window
        kernel-=kernel.mean()
        kernel/=max(float(kernel[kernel>0].sum()),1e-10)
        evidence=cv2.filter2D(-g,cv2.CV_32F,kernel,borderType=cv2.BORDER_REFLECT)
        np.maximum(out,evidence,out=out)
    if not np.isfinite(out).all():raise ValueError('Invalid directional response')
    return out


def _skeleton_graph(skeleton:np.ndarray):
    h,w=skeleton.shape
    ys,xs=np.nonzero(skeleton)
    index=np.full(skeleton.shape,-1,np.int32)
    index[ys,xs]=np.arange(len(xs),dtype=np.int32)
    coords=np.column_stack((xs,ys))
    neighbors=[]
    for x,y in coords:
        n=[]
        for dx,dy in ((-1,0),(0,-1),(0,1),(1,0),(-1,-1),(-1,1),(1,-1),(1,1)):
            xx=x+dx;yy=y+dy
            if not(0<=xx<w and 0<=yy<h and skeleton[yy,xx]):continue
            # No two-hop shortcut diagonal across orthogonal skeleton pixels.
            if dx and dy and (skeleton[y,xx] or skeleton[yy,x]):continue
            n.append(int(index[yy,xx]))
        neighbors.append(n)
    return coords,neighbors


def _prune_spurs(neighbors,limit:int)->np.ndarray:
    active=np.ones(len(neighbors),bool)
    for _ in range(8):
        degree=np.array([sum(active[j] for j in n) if active[i] else 0
                         for i,n in enumerate(neighbors)],dtype=np.int16)
        leaves=np.flatnonzero(active&(degree==1))
        doomed=set()
        for i in leaves:
            p=[int(i)];last=-1;cur=int(i)
            for _ in range(limit+2):
                nexts=[j for j in neighbors[cur] if active[j] and j!=last]
                if len(nexts)!=1:break
                last,cur=cur,nexts[0]
                if degree[cur]!=2:break
                p.append(cur)
            if degree[cur]>=3 and len(p)<=limit:doomed.update(p)
        if not doomed:break
        active[list(doomed)]=False
    return active


def _paths(coords,neighbors,active,min_length:float):
    degree=[sum(active[j] for j in nb) if active[i] else 0
            for i,nb in enumerate(neighbors)]
    seen=set();output=[]
    def edge(i,j):return (min(i,j),max(i,j))
    def walk(i,j):
        ids=[i];prev=i;here=j
        while True:
            e=edge(prev,here)
            if e in seen:break
            seen.add(e);ids.append(here)
            opts=[k for k in neighbors[here] if active[k] and k!=prev]
            if degree[here]!=2 or len(opts)!=1:break
            prev,here=here,opts[0]
            if here==i:break
        return ids
    def commit(ids):
        if len(ids)<3:return
        xy=coords[ids].astype(np.float64)
        length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())
        if length>=min_length:output.append((xy,length))
    for i in range(len(coords)):
        if not active[i] or degree[i]==2:continue
        for j in neighbors[i]:
            if active[j] and edge(i,j) not in seen:commit(walk(i,j))
    for i in range(len(coords)):
        if not active[i]:continue
        for j in neighbors[i]:
            if active[j] and edge(i,j) not in seen:commit(walk(i,j))
    return output


def _spline(xy:np.ndarray,length:float):
    if len(xy)<4:return xy
    cumulative=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    if cumulative[-1]<=0:return xy
    keep=np.r_[True,np.diff(cumulative)>1e-6]
    xy=xy[keep];u=(cumulative[keep]/cumulative[keep][-1])
    if len(xy)<4:return xy
    try:
        # Smoothing for a 1 px centerline: never claim subpixel source accuracy.
        tck,_=splprep([xy[:,0],xy[:,1]],u=u,s=len(xy)*.5,k=3)
        samples=max(3,int(length/2)+1)
        fitted=np.column_stack(splev(np.linspace(0,1,samples),tck))
        # The source path, not extrapolated spline controls, defines A and B.
        fitted[0]=xy[0]
        fitted[-1]=xy[-1]
        return fitted
    except (ValueError,TypeError):
        return xy


def reconstruct(gray:np.ndarray,protected:np.ndarray|None=None,cfg:Config=Config()):
    response=directional_response(gray,cfg)
    mask=response>=cfg.response_threshold
    if protected is not None:
        if protected.shape!=gray.shape:raise ValueError('Invalid cross mask dimensions')
        mask &= ~protected.astype(bool)
    # OpenCV component filtering is stable across scikit-image releases.
    count,components,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    valid=np.flatnonzero(stats[:,cv2.CC_STAT_AREA]>=cfg.min_binary_object_pixels)
    valid=valid[valid!=0]
    mask=np.isin(components,valid)
    skeleton=skeletonize(mask)
    coords,adj=_skeleton_graph(skeleton)
    active=_prune_spurs(adj,cfg.prune_spur_length_px)
    pieces=_paths(coords,adj,active,cfg.min_path_length_px)
    objects=[]
    for i,(pixel_path,length) in enumerate(pieces):
        fitted=_spline(pixel_path,length)
        fitted[:,0]=np.clip(fitted[:,0],0,gray.shape[1]-1)
        fitted[:,1]=np.clip(fitted[:,1],0,gray.shape[0]-1)
        # Integrate photometric response at actual observed skeleton points.
        p=np.rint(pixel_path).astype(int)
        evidence=float(np.mean(response[p[:,1],p[:,0]]))
        objects.append({'tracklet_id':f'S114_{i:04d}',
                        'observed_pixel_path_xy':pixel_path.round(2).tolist(),
                        'spline_xy':fitted.round(2).tolist(),
                        'start_A_xy':fitted[0].round(2).tolist(),
                        'end_B_xy':fitted[-1].round(2).tolist(),
                        'observed_arc_length_px':round(length,2),
                        'mean_directional_response':round(evidence,3),
                        'is_major_visual_candidate':bool(length>=cfg.display_major_length_px),
                        'physical_particle_id':None,'source_observation':'PIXEL_DERIVED',
                        'A_B_are_geometric_not_physical':True})
    objects.sort(key=lambda z:(-z['observed_arc_length_px'],z['start_A_xy']))
    return {'tracks':objects,'response':response,
            'mask':mask,'skeleton':skeleton,
            'report':{'source_pixels':int(gray.size),'skeleton_pixel_count':int(skeleton.sum()),
                      'candidate_piece_count':len(objects),
                      'major_candidate_pieces':sum(z['is_major_visual_candidate'] for z in objects),
                      'observed_candidate_length_px':round(sum(z['observed_arc_length_px'] for z in objects),2),
                      'source_first_not_trained_GPU':True,'not_GT':True}}


def _outward(track,end):
    p=np.asarray(track['spline_xy'],float)
    if end==0:p=p[::-1]
    lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    i=min(len(p)-1,max(1,int(np.searchsorted(lengths,12))))
    d=p[0]-p[i]
    return d/max(np.linalg.norm(d),1e-8)


def proposed_bridges(tracks,response:np.ndarray,cfg:Config=Config()):
    """Generate review-only continuations; *never* claim added observed pixels."""
    eligible=[t for t in tracks if t['is_major_visual_candidate']]
    if len(eligible)<2:return []
    ends=[(t,e,np.array(t['start_A_xy'] if e==0 else t['end_B_xy'],float))
          for t in eligible for e in (0,1)]
    near=cKDTree(np.array([t[2] for t in ends])).query_pairs(cfg.max_review_gap_px)
    candidates=[]
    for ai,bi in sorted(near):
        a,ea,p=ends[ai];b,eb,q=ends[bi]
        if a['tracklet_id']==b['tracklet_id']:continue
        delta=q-p;distance=float(np.linalg.norm(delta))
        if distance<5:continue
        axis=delta/distance
        ta=_outward(a,ea);tb=_outward(b,eb)
        c1=float(ta@axis);c2=float(tb@(-axis))
        if min(c1,c2)<.87:continue
        t=np.linspace(0,1,max(20,int(distance*2)))[:,None]
        # Cubic with matching tangents (entirely inferred between source pieces).
        h=distance/3.0
        controls=np.array([p,p+ta*h,q+tb*h,q])
        sampled=((1-t)**3*controls[0]+3*((1-t)**2)*t*controls[1]
                 +3*(1-t)*(t**2)*controls[2]+(t**3)*controls[3])
        xy=np.rint(sampled).astype(int)
        if np.any(xy[:,0]<0) or np.any(xy[:,1]<0) or np.any(xy[:,0]>=response.shape[1]) or np.any(xy[:,1]>=response.shape[0]):continue
        photo=float(np.mean(response[xy[:,1],xy[:,0]]>cfg.gap_photo_threshold))
        if photo<cfg.min_gap_photo_support:continue
        # Reject crossing another major candidate through the gap interior.
        collides=False
        for other in eligible:
            if other['tracklet_id'] in (a['tracklet_id'],b['tracklet_id']):continue
            s=np.asarray(other['spline_xy'],float)
            if len(s)==0:continue
            if np.min(cKDTree(s).query(sampled[5:-5])[0])<4:collides=True;break
        if collides:continue
        candidates.append({'candidate_id':f'gap_{len(candidates):04d}',
                           'tracklet_ids':[a['tracklet_id'],b['tracklet_id']],
                           'endpoint_indices':[ea,eb],
                           'geometric_gap_px':round(distance,2),
                           'tangent_alignment':[round(c1,3),round(c2,3)],
                           'directional_photo_support_fraction':round(photo,3),
                           'inferred_cubic_xy':controls.round(2).tolist(),
                           'inferred_samples_xy':sampled.round(2).tolist(),
                           'status':'REVIEW_ONLY_NOT_OBSERVED',
                           'is_observed':False})
    candidates.sort(key=lambda z:(-z['directional_photo_support_fraction'],z['geometric_gap_px']))
    return candidates


def render(gray,tracks,bridges=None,operator_points=None,include_short=False):
    img=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    for i,t in enumerate(tracks):
        if not include_short and not t['is_major_visual_candidate']:continue
        color=tuple(int(c) for c in cv2.cvtColor(np.uint8([[[int((i*35)%180),220,210]]]),cv2.COLOR_HSV2BGR)[0,0])
        pts=np.rint(t['spline_xy']).astype(np.int32)
        if len(pts)>1:cv2.polylines(img,[pts],False,color,2,cv2.LINE_AA)
        a=tuple(np.rint(t['start_A_xy']).astype(int));b=tuple(np.rint(t['end_B_xy']).astype(int))
        cv2.circle(img,a,4,(40,220,40),-1,cv2.LINE_AA)
        cv2.circle(img,b,4,(50,50,245),-1,cv2.LINE_AA)
    for gap in bridges or []:
        xy=np.rint(gap['inferred_samples_xy']).astype(int)
        for j in range(0,len(xy)-1,6):
            k=min(j+3,len(xy)-1)
            cv2.line(img,tuple(xy[j]),tuple(xy[k]),(0,170,255),2,cv2.LINE_AA)
    for p in operator_points or []:
        x,y=map(lambda z:int(round(z)),p['xy_local'])
        label=p['label'];col=(190,45,230) if label.startswith('f') else (255,200,60)
        cv2.circle(img,(x,y),16,col,2,cv2.LINE_AA)
        cv2.putText(img,label,(x+18,y-10),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(img,label,(x+18,y-10),cv2.FONT_HERSHEY_SIMPLEX,.65,col,1,cv2.LINE_AA)
    return img
