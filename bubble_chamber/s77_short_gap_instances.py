"""S77: guarded pixel-supported merging of *two already observed* short fragments.

This is an inference-only, source-blind hypothesis generator. It does not
identify physical tracks and never receives synthetic source coordinates.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from .s39_machine_auto import contrast_baseline, sample_bilinear

ARMS=('baseline','short_gap_strict','short_gap_relaxed')
# Every threshold was frozen before S77 CAL/TEST; never update using TEST results.
POLICIES={
 'short_gap_strict':dict(max_gap=5.5,min_cos=float(np.cos(np.deg2rad(18.))),
                         min_dark=.006,min_gap_vs_inside=.32),
 'short_gap_relaxed':dict(max_gap=8.5,min_cos=float(np.cos(np.deg2rad(25.))),
                         min_dark=.004,min_gap_vs_inside=.24),
}
MIN_GAP=1.6
RIVAL_MARGIN=.13
THIRD_CURVE_DISTANCE=2.5


def _piece(poly):
    a=np.asarray(poly,dtype=float)
    if a.ndim!=2 or a.shape[1]!=2 or len(a)<8 or not np.isfinite(a).all():
        raise ValueError('Expected a finite pixel-derived 2-D curve')
    return a


def _outward(a,port):
    tip=a[port]
    inner=a[min(4,len(a)-1)] if port==0 else a[max(0,len(a)-5)]
    v=tip-inner
    size=float(np.linalg.norm(v))
    return None if size<1e-8 else v/size


def _evidence_score(dark,curve_a,pa,curve_b,pb,policy,others):
    a,b=curve_a[pa],curve_b[pb]
    delta=b-a
    gap=float(np.linalg.norm(delta))
    if not MIN_GAP<=gap<=policy['max_gap']:return None
    ua=_outward(curve_a,pa);ub=_outward(curve_b,pb)
    if ua is None or ub is None:return None
    d=delta/gap
    cos_a=float(np.dot(ua,d));cos_b=float(np.dot(ub,-d))
    if min(cos_a,cos_b)<policy['min_cos']:return None
    # Require three genuine (possibly weak) pixel supports *inside the gap*.
    ts=np.array([.25,.50,.75],dtype=float)
    gap_samples=a[None,:]+ts[:,None]*delta
    if (np.any(gap_samples<7) or np.any(gap_samples[:,0]>=dark.shape[1]-7)
            or np.any(gap_samples[:,1]>=dark.shape[0]-7)):return None
    gaps=sample_bilinear(dark,gap_samples)
    interior_a=np.asarray(curve_a[1:5] if pa==0 else curve_a[-5:-1],float)
    interior_b=np.asarray(curve_b[1:5] if pb==0 else curve_b[-5:-1],float)
    support=np.concatenate((sample_bilinear(dark,interior_a),sample_bilinear(dark,interior_b)))
    gap_med=float(np.median(gaps)); inside=float(np.median(support))
    if gap_med<policy['min_dark'] or gap_med/max(inside,.01)<policy['min_gap_vs_inside']:
        return None
    if others:
        tree=cKDTree(np.concatenate(others))
        if float(np.min(tree.query(gap_samples)[0]))<THIRD_CURVE_DISTANCE:return None
    # This is NOT a track probability; every score component depends on pixels
    # and candidate geometry only, not synthetic labels or source identity.
    return float(.48*(cos_a+cos_b)/2+.37*min(gap_med/max(inside,.01),1.)
                 +.15*(1.-gap/policy['max_gap']))


def propose_pixel_gap_pairs(gray,curves,arm):
    """Enumerate safe local hypotheses; never asserts physical continuity."""
    if not isinstance(gray,np.ndarray) or gray.dtype!=np.uint8 or gray.shape!=(192,192):
        raise ValueError('Input must be 192x192 uint8 pixels')
    if arm not in ARMS:raise ValueError('Unfrozen S77 arm')
    if len(curves)>32:raise ValueError('Graph candidate budget exceeded')
    pieces=[_piece(x) for x in curves]
    if arm=='baseline' or len(pieces)<2:return []
    policy=POLICIES[arm];dark=contrast_baseline(gray)
    edges=[]
    for i,one in enumerate(pieces):
        for j in range(i+1,len(pieces)):
            two=pieces[j]
            others=[p for k,p in enumerate(pieces) if k not in (i,j)]
            for pa in (0,-1):
                for pb in (0,-1):
                    score=_evidence_score(dark,one,pa,two,pb,policy,others)
                    if score is not None:
                        edges.append((score,(i,pa),(j,pb)))
    # Score each endpoint across *all* potential partners and abstain on rivals.
    by_port=defaultdict(list)
    for score,a,b in edges:
        by_port[a].append(score);by_port[b].append(score)
    accepted=[]
    for score,a,b in edges:
        if any(sum(other>score-RIVAL_MARGIN-1e-12 for other in by_port[port])>1
               for port in (a,b)):
            continue
        accepted.append((score,a,b))
    accepted.sort(key=lambda x:(-x[0],x[1],x[2]))
    used=set();parent=list(range(len(pieces)));degree=[0]*len(pieces)
    def root(n):
        while n!=parent[n]:n=parent[n]
        return n
    selected=[]
    for score,a,b in accepted:
        if a in used or b in used or root(a[0])==root(b[0]):continue
        if degree[a[0]]>=2 or degree[b[0]]>=2:continue
        parent[root(b[0])]=root(a[0]);degree[a[0]]+=1;degree[b[0]]+=1
        used.update((a,b));selected.append((score,a,b))
    return selected


def repair_pixel_only(gray,curves,arm):
    """Return independently observable polylines, with reversible merge counts.

    The resulting objects are proposals, not verified particle identities.
    """
    pieces=[_piece(x).copy() for x in curves]
    matches=propose_pixel_gap_pairs(gray,pieces,arm)
    if not matches:return {'polylines':pieces,'merged_pairs':0,
                           'input_curves':len(pieces),'method':arm,
                           'physical_identity_verified':False}
    links={}
    for _,a,b in matches:links[a]=b;links[b]=a
    visited=set();rebuilt=[]
    for i in range(len(pieces)):
        if i in visited:continue
        # Chains cannot cycle. Begin at an unlinked endpoint if possible.
        starts=[p for p in (0,-1) if (i,p) not in links]
        if not starts: # an internal piece may appear earlier in another chain
            continue
        cur=i;entry=starts[0];assembled=[]
        while cur not in visited:
            visited.add(cur)
            path=pieces[cur] if entry==0 else pieces[cur][::-1]
            if assembled and np.linalg.norm(np.asarray(assembled[-1])-path[0])<.2:
                path=path[1:]
            assembled.extend(path.tolist())
            out_port=-1 if entry==0 else 0
            nxt=links.get((cur,out_port))
            if nxt is None:break
            other,other_port=nxt
            if other in visited:break
            target=pieces[other][other_port]
            gap=float(np.linalg.norm(np.asarray(assembled[-1])-target))
            if gap>1.6:
                # interpolated gap points preserve a continuous sampled curve;
                # inference *only* drew this bridge after image evidence check.
                left=np.asarray(assembled[-1])
                for f in np.arange(1.,gap,1.)/gap:
                    assembled.append((left+f*(target-left)).tolist())
            cur=other;entry=other_port
        rebuilt.append(np.asarray(assembled,float))
    for i,p in enumerate(pieces):
        if i not in visited:rebuilt.append(p)
    if len(rebuilt)!=len(pieces)-len(matches):
        raise AssertionError('Lost/duplicated source component while repairing')
    if len(rebuilt)>32:raise AssertionError('Bounded candidate count exceeded')
    return {'polylines':rebuilt,'merged_pairs':len(matches),
            'input_curves':len(pieces),'method':arm,
            'physical_identity_verified':False}
