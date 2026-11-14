"""V49 label-blind observed-boundary path decoders for controlled missing segments.

The route constructors use only predicted source/target candidates, the
observed post-intervention segmentation probability, and the intervention
rectangle. Hidden expert endpoints/centerline are not accepted as inputs.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import ndimage as ndi


def dense_polyline(points,shape):
    """Rasterize and connect piecewise linear coordinates at <=0.22 px steps."""
    a=np.asarray(points,np.float64)
    if a.ndim!=2 or a.shape[1]!=2 or len(a)<2 or not np.isfinite(a).all():
        raise ValueError("Path must be finite Nx2 with at least two vertices")
    out=[]
    h,w=shape
    for start,goal in zip(a[:-1],a[1:]):
        n=max(2,math.ceil(float(np.linalg.norm(goal-start))*5)+1)
        for p in np.linspace(start,goal,n):
            y,x=np.rint(p).astype(int)
            if 0<=y<h and 0<=x<w:
                pix=(int(y),int(x))
                if not out or out[-1]!=pix:
                    out.append(pix)
    return tuple(out)


def expected_rectangle_crossings(start,end,rect):
    """Return entry and exit at observed intervention box or fail closed."""
    a=np.asarray(start,float);b=np.asarray(end,float)
    if a.shape!=(2,) or b.shape!=(2,) or not np.isfinite([*a,*b]).all():
        raise ValueError("Invalid candidate endpoints")
    y0,y1,x0,x1=rect
    # Deterministic dense search; rectangle is known intervention geometry,
    # not an expert line, and is already used by original candidate SBPP.
    t=np.linspace(0.,1.,1001)
    p=a[None,:]+t[:,None]*(b-a)[None,:]
    hit=(p[:,0]>=y0)&(p[:,0]<y1)&(p[:,1]>=x0)&(p[:,1]<x1)
    if not hit.any():
        return None
    ii=np.flatnonzero(hit)
    return p[ii[0]],p[ii[-1]]


def estimate_observed_anchor(prob,inside,ring,endpoint,crossing,other):
    ys,xs=np.nonzero(ring)
    pts=np.stack([ys,xs],axis=1).astype(float)
    if not len(pts):
        return None
    ep=np.asarray(endpoint,float)
    cp=np.asarray(crossing,float)
    opp=np.asarray(other,float)
    d_cross=np.linalg.norm(pts-cp[None,:],axis=1)
    d_end=np.linalg.norm(pts-ep[None,:],axis=1)
    delta=opp-ep
    d_segment=max(np.dot(delta,delta),1e-8)
    projection=np.clip(((pts-ep)@delta)/d_segment,0,1)
    foot=ep[None,:]+projection[:,None]*delta[None,:]
    d_line=np.linalg.norm(pts-foot,axis=1)
    # Hard geometric constraints fixed before real dataset inspection.
    valid=(d_cross<=14)&(d_end<=30)&(d_line<=12)&(prob[ys,xs]>=.45)
    if not valid.any():
        return None
    score=5*prob[ys,xs]-.35*d_cross-.12*d_line-.025*d_end
    score[~valid]=-np.inf
    j=int(np.argmax(score))
    return tuple(map(int,(ys[j],xs[j])))


def boundary_anchors(prob,rect,start,end):
    p=np.asarray(prob,np.float32)
    if p.shape!=(96,96) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Invalid observed segmentation")
    cross=expected_rectangle_crossings(start,end,rect)
    if cross is None:
        return None
    y0,y1,x0,x1=rect
    inside=np.zeros_like(p,dtype=bool)
    inside[y0:y1,x0:x1]=True
    ring=ndi.binary_dilation(inside,iterations=3)&~inside
    first=estimate_observed_anchor(p,inside,ring,start,cross[0],end)
    last=estimate_observed_anchor(p,inside,ring,end,cross[1],start)
    if first is None or last is None or math.dist(first,last)<4.:
        return None
    return first,last,inside


def observed_tangent(prob,inside,anchor,forward):
    """Local principal axis of observed support EXCLUDING the erased ROI."""
    p=np.asarray(prob,np.float64)
    a=np.asarray(anchor,np.float64)
    direction=np.asarray(forward,np.float64)
    nd=max(np.linalg.norm(direction),1e-8)
    direction/=nd
    ys,xs=np.nonzero((p>=.55)&~inside)
    if not len(ys):
        return direction
    pts=np.column_stack([ys,xs]).astype(float)
    dist=np.linalg.norm(pts-a[None,:],axis=1)
    close=dist<=10
    pts=pts[close]
    if len(pts)<9:
        return direction
    weights=np.clip(p[ys[close],xs[close]]-.5,.05,1.)
    center=np.average(pts,axis=0,weights=weights)
    delta=pts-center
    cov=(delta*weights[:,None]).T@delta/max(weights.sum(),1)
    vals,vecs=np.linalg.eigh(cov)
    if vals[-1]<=1e-8 or vals[-1]/max(vals[0],1e-6)<1.3:
        return direction
    axis=vecs[:,-1]
    if np.dot(axis,direction)<0:axis=-axis
    return axis/max(np.linalg.norm(axis),1e-8)


def boundary_route(prob,rect,start,end,*,mode):
    """Candidate-only route; no hidden centerline or expert endpoints."""
    if mode not in ("boundary_straight","boundary_hermite"):
        raise ValueError("Unknown frozen route")
    start=np.asarray(start,float)
    end=np.asarray(end,float)
    anchors=boundary_anchors(prob,rect,start,end)
    if anchors is None:
        return dense_polyline([start,end],(96,96)),False
    a,b,inside=anchors
    a=np.asarray(a,float)
    b=np.asarray(b,float)
    if mode=="boundary_straight":
        return dense_polyline([start,a,b,end],(96,96)),True
    forward=b-a
    n=max(math.ceil(np.linalg.norm(forward)*5),60)
    m0=observed_tangent(prob,inside,a,forward)*(.30*np.linalg.norm(forward))
    m1=observed_tangent(prob,inside,b,forward)*(.30*np.linalg.norm(forward))
    t=np.linspace(0.,1.,n)
    h00=2*t**3-3*t**2+1
    h10=t**3-2*t**2+t
    h01=-2*t**3+3*t**2
    h11=t**3-t**2
    curve=(h00[:,None]*a+h10[:,None]*m0+
           h01[:,None]*b+h11[:,None]*m1)
    return dense_polyline(np.vstack([start,curve,end]),(96,96)),True


def hidden_path_success(scene,path):
    """Exactly original V26 hidden-centerline metric (scoring-only truth).

    Actual geometry is built before this function can see 'hidden_centerline'.
    """
    y0,y1,x0,x1=scene["rectangle"]
    true_global=np.asarray(scene["hidden_centerline"],int)
    origin=np.asarray(scene["origin"],int)
    true=true_global-origin[None,:]
    h,w=np.asarray(scene["visible_prob"]).shape
    inside=np.zeros((h,w),bool)
    for y,x in path:
        if y0<=y<y1 and x0<=x<x1 and 0<=y<h and 0<=x<w:
            inside[y,x]=True
    truth=np.zeros_like(inside)
    for y,x in true:
        if 0<=y<h and 0<=x<w and y0<=y<y1 and x0<=x<x1:
            truth[y,x]=True
    if not inside.any() or not truth.any():
        return False
    near_pred=ndi.binary_dilation(inside,iterations=2)
    near_true=ndi.binary_dilation(truth,iterations=2)
    precision=float(near_true[inside].mean())
    recall=float(near_pred[truth].mean())
    return bool(precision>=.8 and recall>=.8)


def score_new_paths(scene,top_proposal):
    """Report observed boundary support and exact independent path metric."""
    routes={}
    for mode in ("boundary_straight","boundary_hermite"):
        path,anchored=boundary_route(
            scene["visible_prob"],scene["rectangle"],
            top_proposal["source"],top_proposal["target"],mode=mode)
        routes[mode]={"success":bool(hidden_path_success(scene,path)),
                      "found_observed_anchors":bool(anchored),
                      "path_n_pixels":len(path)}
    return routes
