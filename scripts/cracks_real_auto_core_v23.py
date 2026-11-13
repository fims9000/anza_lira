"""Image-only segmentation, graph candidates, selective linking and paths.

No annotation masks or oracle endpoints are accepted by inference functions.
ANZA-inspired anisotropic image features are a *new lightweight surrogate*,
not the original pretrained ANZA operator. This is a real-image E2E prototype.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import math

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize, remove_small_objects


@dataclass(frozen=True)
class Trace:
    id: int
    points: np.ndarray
    mean_probability: float


@dataclass(frozen=True)
class Port:
    id: int
    trace_id: int
    point: tuple[int, int]
    tangent: tuple[float, float]
    confidence: float


@dataclass(frozen=True)
class Edge:
    first: int
    second: int
    distance: float
    geometry: float
    support: float
    score: float


def unit(v: np.ndarray) -> np.ndarray:
    v=np.asarray(v,dtype=np.float64)
    return v/max(float(np.linalg.norm(v)),1e-9)


def _shift_x(arr: np.ndarray, amount: int) -> np.ndarray:
    # Explicit nearest boundary, never np.roll's artificial left-right wrap.
    h,w=arr.shape
    xx=np.clip(np.arange(w)+amount,0,w-1)
    return arr[:,xx]


def anisotropic_features(image: np.ndarray) -> np.ndarray:
    """Return 10 per-pixel features capturing cross-stratal continuity loss.

    Images HxW float [0,1]. No dataset-specific labels or crop statistics.
    """
    a=np.asarray(image,np.float32)
    if a.ndim!=2 or not np.isfinite(a).all():
        raise ValueError('Expected finite grayscale HxW image')
    smooth=ndi.gaussian_filter(a,1.15)
    gy=ndi.sobel(smooth,axis=0)/8.
    gx=ndi.sobel(smooth,axis=1)/8.
    scale=max(float(np.percentile(np.abs(gy),95)),1e-4)
    gy=gy/scale
    gx=gx/scale
    layers=[a,smooth,np.abs(gy),np.abs(gx)]
    energy=np.sqrt(ndi.gaussian_filter(gy*gy,2.5))
    layers.append(np.clip(energy,0,5))
    for delta in (3,7,11):
        left=_shift_x(gy,-delta)
        right=_shift_x(gy,delta)
        numerator=ndi.gaussian_filter(left*right,(3.0,2.0))
        denom=np.sqrt(ndi.gaussian_filter(left*left,(3.,2.))*
                      ndi.gaussian_filter(right*right,(3.,2.)))+1e-5
        corr=np.clip(numerator/denom,-1,1)
        layers.append((1-corr)/2)
    # Informed by anisotropic geometry, but NOT a pretrained ANZA feature layer.
    structure=ndi.gaussian_filter(gx*gx,(2,3))
    cross=ndi.gaussian_filter(gx*gy,(2,3))
    vert=ndi.gaussian_filter(gy*gy,(2,3))
    anis=np.sqrt((structure-vert)**2+4*cross**2)/(
        structure+vert+1e-5)
    layers.append(np.clip(anis,0,1))
    layers.append(np.clip(ndi.gaussian_filter(np.abs(gx),(1,2)),0,5))
    return np.stack(layers,axis=-1).astype(np.float32)


def extract_traces(probability: np.ndarray, *, threshold: float, min_pixels=9,
                   border_margin=5) -> tuple[np.ndarray, tuple[Trace,...],tuple[Port,...]]:
    """Extract connected unbranched skeleton traces and outward terminal ports."""
    prob=np.asarray(probability,dtype=np.float32)
    if prob.ndim!=2 or not np.isfinite(prob).all():
        raise ValueError('Non-finite segmentation')
    mask=remove_small_objects(prob>=float(threshold),min_size=min_pixels)
    sk=skeletonize(mask)
    kernel=np.ones((3,3),np.uint8);kernel[1,1]=0
    degrees=ndi.convolve(sk.astype(np.uint8),kernel,mode='constant')
    junction=ndi.binary_dilation(sk & (degrees>=3),iterations=1)
    segments=sk & ~junction
    labels,n=ndi.label(segments,structure=np.ones((3,3),np.uint8))
    traces=[];ports=[]
    for idx,box in enumerate(ndi.find_objects(labels),start=1):
        if box is None:continue
        region=labels[box]==idx
        if int(region.sum())<min_pixels:continue
        degree=ndi.convolve(region.astype(np.uint8),kernel,mode='constant')
        ends=np.argwhere(region & (degree==1))
        if len(ends)!=2 or np.any(degree[region]>=3):continue
        path=[tuple(map(int,ends[0]))]
        last=None
        for _ in range(int(region.sum())-1):
            y,x=path[-1]
            neighbors=sorted((y+dy,x+dx) for dy in (-1,0,1)
                for dx in (-1,0,1) if (dy or dx) and
                0<=y+dy<region.shape[0] and 0<=x+dx<region.shape[1]
                and region[y+dy,x+dx] and (y+dy,x+dx)!=last)
            if len(neighbors)!=1 or neighbors[0] in path:break
            last=path[-1];path.append(neighbors[0])
        if len(path)!=int(region.sum()):continue
        offs=np.array([box[0].start,box[1].start])
        coords=np.asarray(path,int)+offs[None,:]
        if np.any(coords[0]<border_margin) or np.any(coords[-1]<border_margin) or \
           np.any(coords[0]>=np.array(prob.shape)-border_margin) or \
           np.any(coords[-1]>=np.array(prob.shape)-border_margin):
            continue
        tid=len(traces)
        trace=Trace(tid,coords,float(prob[coords[:,0],coords[:,1]].mean()))
        traces.append(trace)
        for side in (0,-1):
            i=0 if side==0 else len(coords)-1
            k=min(6,len(coords)-1)
            u=unit(coords[i].astype(float)-coords[k if side==0 else i-k].astype(float))
            ports.append(Port(len(ports),tid,tuple(coords[i]),tuple(map(float,u)),trace.mean_probability))
    return mask,tuple(traces),tuple(ports)


def propose_edges(ports: tuple[Port,...], *, min_gap=3.,max_gap=36.,
                  max_candidates=8,max_angle_deg=70.) -> tuple[tuple[int,int,float,float],...]:
    """Candidate generation from model-derived ports only; geometry never GT."""
    if not ports:return ()
    pts=np.asarray([p.point for p in ports],dtype=np.float64)
    tree=cKDTree(pts)
    cos_min=math.cos(math.radians(max_angle_deg))
    all_edges={}
    for i,port in enumerate(ports):
        pool=[]
        for j in tree.query_ball_point(pts[i],max_gap):
            if i==j or port.trace_id==ports[j].trace_id:continue
            d=float(np.linalg.norm(pts[j]-pts[i]))
            if d<min_gap:continue
            chord=(pts[j]-pts[i])/max(d,1e-6)
            start=float(np.dot(np.asarray(port.tangent),chord))
            finish=float(np.dot(np.asarray(ports[j].tangent),-chord))
            if min(start,finish)<cos_min:continue
            geo=(d/max_gap)+.7*(2-start-finish)
            pool.append((geo,j,d))
        for geo,j,d in sorted(pool)[:max_candidates]:
            pair=(min(i,j),max(i,j))
            old=all_edges.get(pair)
            if old is None or geo<old[3]:all_edges[pair]=(pair[0],pair[1],d,geo)
    return tuple(sorted(all_edges.values(),key=lambda e:(e[3],e[0],e[1])))


def _path_support(field:np.ndarray,a:tuple[int,int],b:tuple[int,int])->float:
    t=np.linspace(.12,.88,max(10,int(np.linalg.norm(np.array(a)-b)*2)))
    sample=np.array(a,float)[:,None]*(1-t)[None,:]+np.array(b,float)[:,None]*t[None,:]
    return float(ndi.map_coordinates(field,sample,order=1,mode='nearest').mean())


def score_edges(raw_edges,ports,probability, *, strategy:str='context',
                neighbor_probability:np.ndarray|None=None):
    """Frozen demonstration rule, no expert labels, with explicit NO_LINK gate."""
    if strategy not in ('nearest','geometry','context','context_neighbor'):
        raise ValueError(strategy)
    rows=[]
    for i,j,d,g in raw_edges:
        support=_path_support(probability,ports[i].point,ports[j].point)
        s2=_path_support(neighbor_probability,ports[i].point,ports[j].point) if \
            neighbor_probability is not None else support
        if strategy=='nearest': score=-d
        elif strategy=='geometry':score=-g
        elif strategy=='context':score=-g+1.3*support
        else: score=-g+.8*support+.5*s2
        rows.append(Edge(i,j,d,g,support,float(score)))
    return tuple(sorted(rows,key=lambda x:(-x.score,x.first,x.second)))


def accept_edges(ranked,ports,*,strategy,score_gate=-0.3,max_links=120):
    """Greedy disjoint port matching; returns NO_LINK for unused ports."""
    used=set();edges=[];used_trace_pairs=set()
    for e in ranked:
        if len(edges)>=max_links:break
        if e.first in used or e.second in used:continue
        pair=tuple(sorted((ports[e.first].trace_id,ports[e.second].trace_id)))
        if pair in used_trace_pairs:continue
        if strategy in ('context','context_neighbor') and e.score<score_gate:continue
        if strategy=='geometry' and e.geometry>1.1:continue
        if strategy=='nearest' and e.distance>18:continue
        used.update((e.first,e.second));used_trace_pairs.add(pair);edges.append(e)
    return tuple(edges)


def trace_path(probability:np.ndarray,start:tuple[int,int],goal:tuple[int,int],*,
               half_width=6,max_multiplier=2.0):
    """Dijkstra in a narrow corridor; no gold geometry; bounded detours."""
    first=np.asarray(start,dtype=int);last=np.asarray(goal,dtype=int)
    d=float(np.linalg.norm(first-last))
    if d<1:return ()
    pad=int(max(half_width+3,10))
    low=np.maximum(np.minimum(first,last)-pad,0)
    hi=np.minimum(np.maximum(first,last)+pad+1,np.array(probability.shape))
    crop=probability[low[0]:hi[0],low[1]:hi[1]]
    h,w=crop.shape
    if not h or not w:return ()
    yy,xx=np.indices(crop.shape)
    yx=np.stack([yy+low[0],xx+low[1]],axis=-1)
    direction=(last-first).astype(float)
    cross=np.abs(direction[1]*(yx[...,0]-first[0])-direction[0]*(yx[...,1]-first[1]))/d
    valid=cross<=half_width
    start_local=tuple(first-low);goal_local=tuple(last-low)
    valid[start_local]=True;valid[goal_local]=True
    # Edge cost contains soft image support and a reproducible geometric prior.
    cost=.25+2*(1-np.clip(crop,0,1))+.18*cross
    cost=np.where(valid,cost,np.inf)
    source=start_local;target=goal_local
    queue=[(0.,0,source)];best={source:(0.,0)};parent={}
    neigh=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    max_steps=int(max_multiplier*(d+2)+5)
    while queue:
        total,steps,p=heapq.heappop(queue)
        if best.get(p)!=(total,steps):continue
        if p==target:
            path=[p]
            while path[-1]!=source:path.append(parent[path[-1]])
            return tuple((int(v[0]+low[0]),int(v[1]+low[1])) for v in reversed(path))
        if steps>=max_steps:continue
        for dy,dx in neigh:
            y,x=p[0]+dy,p[1]+dx
            if 0<=y<h and 0<=x<w and valid[y,x]:
                step=math.sqrt(2) if dy and dx else 1.
                nxt=(y,x);val=(total+step*.5*(cost[p]+cost[nxt]),steps+1)
                if val<best.get(nxt,(float('inf'),10**9)):
                    best[nxt]=val;parent[nxt]=p
                    heapq.heappush(queue,(*val,nxt))
    return ()


def run_pipeline(probability:np.ndarray,*, threshold:float,strategy:str,
                 score_gate:float=-.3,neighbor_probability=None):
    mask,traces,ports=extract_traces(probability,threshold=threshold)
    raw=propose_edges(ports)
    ranked=score_edges(raw,ports,probability,strategy=strategy,
        neighbor_probability=neighbor_probability)
    accepted=accept_edges(ranked,ports,strategy=strategy,score_gate=score_gate)
    # Truth is never available in this function.
    enriched=mask.copy();completed=[]
    for e in accepted:
        path=trace_path(probability,ports[e.first].point,ports[e.second].point)
        if not path:continue
        aa=np.asarray(path)
        enriched[aa[:,0],aa[:,1]]=True
        completed.append({'p0':e.first,'p1':e.second,'score':e.score,
                          'distance':e.distance,'support':e.support,
                          'path':path})
    return {'raw_mask':mask,'completed_mask':enriched,'traces':traces,
            'ports':ports,'candidates':raw,'ranked':ranked,
            'selected':tuple(completed),
            'none_count':len(ports)-2*len(completed)}
