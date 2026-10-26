"""S67: pixel-only, crossing-aware 2-D curve *hypotheses*.

The inference boundary accepts image pixels only: no source simulator nodes,
particle identity, or stereo information. Matching branches across a crossing
is explicitly a reversible machine hypothesis, not a physical assertion.
"""
from __future__ import annotations

from collections import defaultdict
import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

from .s39_machine_auto import ridge_evidence, raw_detector_mask, _trace_component, sample_bilinear

METHOD = 's67_crossing_orientation_v1'
MAX_CURVES = 32
JOIN_RADIUS_PX = 8.5
MAX_DEFLECTION_DEG = 25.
MIN_UNIQUE_DIRECTION_MARGIN = .08


def _unit(v):
    d=float(np.linalg.norm(v))
    if not np.isfinite(d) or d<1e-8:
        return None
    return np.asarray(v,dtype=float)/d


def _endpoints(curve, component):
    """Endpoint and outward tangent (towards a missing junction)."""
    a=np.asarray(curve,float)
    if a.ndim!=2 or a.shape[1]!=2 or len(a)<8:
        raise ValueError('Invalid pixel-derived polyline')
    for k in (0,-1):
        end=a[k]
        inside=a[min(4,len(a)-1)] if k==0 else a[max(0,len(a)-5)]
        direction=_unit(end-inside)
        if direction is not None:
            yield (component,k,end,direction)


def _candidate_pairings(components,centers):
    """Pair only unambiguous *opposite* tangents at one junction.

    This local continuity rule cannot assert real physical identity at a
    crossing. It never sees any synthetic source IDs or labels.
    """
    proposals=[]
    cos_limit=float(np.cos(np.deg2rad(MAX_DEFLECTION_DEG)))
    for center in centers:
        ports=[]
        for i,p in enumerate(components):
            for port in _endpoints(p,i):
                d=center-port[2]
                if np.linalg.norm(d)>JOIN_RADIUS_PX:continue
                toward=_unit(d)
                if toward is not None and np.dot(port[3],toward)<-.15:continue
                ports.append(port)
        options=[]
        for n,a in enumerate(ports):
            for b in ports[n+1:]:
                if a[0]==b[0]:continue
                compatibility=float(-np.dot(a[3],b[3]))
                if compatibility<cos_limit:continue
                gap=float(np.linalg.norm(a[2]-center)+np.linalg.norm(b[2]-center))
                options.append((compatibility-.003*gap,a,b))
        for score,a,b in sorted(options,key=lambda x:(-x[0],x[1][0],x[2][0])):
            # If a different possible pairing is almost as strong, abstain.
            rival=[s for s,c,d in options if (c is not a or d is not b) and
                   (c[:2]==a[:2] or d[:2]==a[:2] or
                    c[:2]==b[:2] or d[:2]==b[:2])]
            if rival and score-max(rival)<MIN_UNIQUE_DIRECTION_MARGIN:continue
            proposals.append((score,(a[0],a[1]),(b[0],b[1]),tuple(center)))
    proposals.sort(reverse=True,key=lambda p:p[0])
    selected=[];used=set();links=defaultdict(set)
    for _,a,b,mid in proposals:
        if a in used or b in used:continue
        # No cycles: this module proposes only open 2-D centerline curves.
        graph=defaultdict(set)
        for x,y,_ in selected:
            graph[x[0]].add(y[0]);graph[y[0]].add(x[0])
        todo=[a[0]];seen=set()
        while todo:
            cur=todo.pop()
            if cur in seen:continue
            seen.add(cur);todo.extend(graph[cur]-seen)
        if b[0] in seen:continue
        selected.append((a,b,mid));used.update((a,b))
        links[a].add(b);links[b].add(a)
    return selected


def _assemble(components,pairs):
    port_link={}
    for a,b,mid in pairs:
        port_link[a]=(b,np.asarray(mid,float))
        port_link[b]=(a,np.asarray(mid,float))
    output=[];visited=set()
    for i in range(len(components)):
        if i in visited:continue
        # A chain has an unpaired endpoint. A standalone segment is a chain.
        starts=[k for k in (0,-1) if (i,k) not in port_link]
        if not starts:continue
        points=[];node=i;start=starts[0];local=set()
        while node not in local:
            local.add(node);visited.add(node)
            a=components[node]
            directed=a if start==0 else a[::-1]
            if len(points) and np.linalg.norm(points[-1]-directed[0])<.25:
                directed=directed[1:]
            points.extend(directed.tolist())
            exit_port=-1 if start==0 else 0
            nxt=port_link.get((node,exit_port))
            if nxt is None:break
            (other,other_port),mid=nxt
            if other in local:break
            points.append(mid.tolist())
            node=other;start=other_port
        if len(points)>=8:
            out=np.asarray(points,float)
            if np.all(np.isfinite(out)) and np.linalg.norm(out[0]-out[-1])>=17:
                output.append(out)
    # Any ambiguous/cyclic pieces are still kept individually, never discarded.
    for i,p in enumerate(components):
        if i not in visited:output.append(np.asarray(p,float))
    return output


def extract_instance_hypotheses(gray:np.ndarray,max_curves:int=MAX_CURVES):
    """Return pixel-derived 2-D polylines + audit counts, *never* truth labels."""
    if gray.ndim!=2 or gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('Require a uint8 192x192 image, no simulator geometry')
    if not isinstance(max_curves,int) or not 1<=max_curves<=32:
        raise ValueError('Bound the number of candidate instances')
    evidence=ridge_evidence(gray)
    mask=raw_detector_mask(evidence)
    mask=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))>0
    skel=skeletonize(mask).astype('uint8')
    degree=convolve(skel,np.ones((3,3),'uint8'),mode='constant')-skel
    junction=(skel>0)&(degree>=4)
    blocked=cv2.dilate(junction.astype('uint8'),np.ones((3,3),'uint8'))
    skel[blocked>0]=0
    n_j,j_labels,j_stats,j_cent=cv2.connectedComponentsWithStats(blocked,connectivity=8)
    centers=[np.asarray(j_cent[j],float) for j in range(1,n_j) if j_stats[j,cv2.CC_STAT_AREA]>=1]
    n,labels,stats,_=cv2.connectedComponentsWithStats(skel,connectivity=8)
    components=[]
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]<18:continue
        x,y,w,h,_=stats[i]
        piece=_trace_component(labels[y:y+h,x:x+w]==i)
        if piece is None:continue
        piece[:,0]+=x;piece[:,1]+=y
        if np.any(piece<7) or np.any(piece>=185):continue
        components.append(piece)
    # cap before quadratic junction search, while preserving strongest ridges.
    components=sorted(components,key=lambda a:-float(np.median(sample_bilinear(evidence['score'],a))))[:32]
    pairs=_candidate_pairings(components,centers)
    lines=_assemble(components,pairs)
    lines=sorted(lines,key=lambda a:(-float(np.median(sample_bilinear(evidence['score'],a))),-len(a)))[:max_curves]
    return {'polylines':lines,'pixel_junctions':len(centers),
            'prepaired_components':len(components),'provisional_continuity_pairs':len(pairs),
            'synthetic_or_native_GT_used':False,'physical_identity_verified':False,
            'method':METHOD}
