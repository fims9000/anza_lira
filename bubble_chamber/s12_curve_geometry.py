"""Turn a predicted single-track mask into a REVIEWABLE quadratic centerline.

No charge, momentum, ionization, 3D reconstruction or particle identity is
inferred. The fitted curve is in local, pixel-valued coordinates, with an
explicit invertible image transform. Ambiguous/damaged tracks abstain.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from typing import Any

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class QuadraticTrace:
    accepted_for_review: bool
    status: str
    components: int
    skeleton_pixels: int
    principal_origin_xy_px: tuple[float,float] | None
    u_axis_xy: tuple[float,float] | None
    normal_axis_xy: tuple[float,float] | None
    quadratic_v_of_u: tuple[float,float,float] | None
    u_extent_px: tuple[float,float] | None
    endpoints_xy_px: tuple[tuple[float,float],tuple[float,float]] | None
    rms_fit_px: float | None
    center_curvature_inv_px: float | None


def fit_single_track(mask: np.ndarray, *, min_points:int=12,
                     max_normalized_rms:float=.2) -> QuadraticTrace:
    z=np.asarray(mask)
    if z.ndim!=2 or z.size==0 or min_points<3:
        raise ValueError('Expected one [H,W] binary image and at least 3 points')
    active=np.asarray(z>0,dtype=np.uint8)
    n,labels,areas,_=cv2.connectedComponentsWithStats(active,8)
    components=int(n-1)
    if components<1:
        return QuadraticTrace(False,'blank',0,0,None,None,None,None,None,None,None,None)
    sizes=areas[1:,cv2.CC_STAT_AREA]
    best=1+int(np.argmax(sizes))
    chosen=(labels==best)
    thin=skeletonize(chosen)
    ij=np.column_stack(np.where(thin))
    length=len(ij)
    if length<min_points:
        return QuadraticTrace(False,'too_short',components,length,None,None,None,None,None,None,None,None)
    xy=ij[:,::-1].astype(np.float64)
    origin=xy.mean(axis=0)
    centered=xy-origin
    covariance=(centered.T@centered)/len(centered)
    evals,evecs=np.linalg.eigh(covariance)
    if evals[-1]<1e-8:
        return QuadraticTrace(False,'degenerate',components,length,None,None,None,None,None,None,None,None)
    axis=evecs[:,-1]
    if axis[0]<0 or (abs(axis[0])<1e-8 and axis[1]<0):axis=-axis
    normal=np.array([-axis[1],axis[0]])
    u=centered@axis
    v=centered@normal
    span=max(float(np.ptp(u)),1e-8)
    # Normalize for numerically stable fitting at full-resolution BMP size.
    q=u/span
    X=np.stack([q*q,q,np.ones_like(q)],axis=-1)
    coeff,*_=np.linalg.lstsq(X,v,rcond=1e-10)
    coeff_unscaled=np.array([coeff[0]/span**2,coeff[1]/span,coeff[2]])
    fitted=X@coeff
    rms=float(np.sqrt(np.mean((fitted-v)**2)))
    lo=float(np.min(u));hi=float(np.max(u))
    def end(position):
        vv=float(np.polyval(coeff_unscaled,position))
        point=origin+axis*position+normal*vv
        return (float(point[0]),float(point[1]))
    a,b,_=coeff_unscaled
    curvature=float(abs(2*a)/(1+b*b)**1.5)
    passed=(components==1 and rms/span<=max_normalized_rms)
    status=('ok' if passed else 'disconnected_requires_lira' if components>1 else 'nonquadratic_requires_review')
    return QuadraticTrace(passed,status,components,length,
         tuple(map(float,origin)),tuple(map(float,axis)),tuple(map(float,normal)),
         tuple(map(float,coeff_unscaled)),(lo,hi),(end(lo),end(hi)),rms,curvature)


def sample_fitted_curve(trace:QuadraticTrace,n:int=100)->np.ndarray:
    if trace.quadratic_v_of_u is None or trace.principal_origin_xy_px is None or trace.u_extent_px is None:
        raise ValueError('No valid curve coefficients')
    if n<2:raise ValueError('Need >=2 samples')
    u=np.linspace(*trace.u_extent_px,n)
    v=np.polyval(trace.quadratic_v_of_u,u)
    o=np.asarray(trace.principal_origin_xy_px)
    x=np.asarray(trace.u_axis_xy);normal=np.asarray(trace.normal_axis_xy)
    return o[None,:]+u[:,None]*x[None,:]+v[:,None]*normal[None,:]


def evaluate_curves(pred_masks:np.ndarray,true_masks:np.ndarray)->dict[str,Any]:
    """Offline synthetic-mask comparison. Truth never used to fit prediction."""
    if pred_masks.shape!=true_masks.shape or pred_masks.ndim!=3:
        raise ValueError('Need aligned [N,H,W] masks')
    if not len(pred_masks):
        raise ValueError('Need at least one image')
    statuses={}
    true_statuses={}
    accepted=0
    eligible_reference=0
    errs=[]; curvature=[];rms=[]
    for pred,true in zip(pred_masks,true_masks):
        fit_p=fit_single_track(pred)
        fit_t=fit_single_track(true)
        statuses[fit_p.status]=statuses.get(fit_p.status,0)+1
        true_statuses[fit_t.status]=true_statuses.get(fit_t.status,0)+1
        eligible_reference+=int(fit_t.accepted_for_review)
        accepted+=int(fit_p.accepted_for_review)
        # Never calculate a scientific error for an abstained disconnected or
        # nonquadratic candidate. Both masks must independently pass the same
        # geometry gate; otherwise reporting only the longest fragment would
        # make broken tracks look deceptively accurate.
        if not (fit_p.accepted_for_review and fit_t.accepted_for_review):
            continue
        a=np.array(fit_p.endpoints_xy_px);b=np.array(fit_t.endpoints_xy_px)
        # No directional arrow supplied by the 2D film; allow endpoint swap.
        dist1=np.linalg.norm(a-b,axis=1).mean()
        dist2=np.linalg.norm(a-b[::-1],axis=1).mean()
        errs.append(float(min(dist1,dist2)))
        curvature.append(float(abs(fit_p.center_curvature_inv_px-fit_t.center_curvature_inv_px)))
        rms.append(float(fit_p.rms_fit_px))
    return {'images':len(pred_masks),'status_counts':statuses,'reference_status_counts':true_statuses,
            'predicted_accepted_for_review':accepted,'eligible_reference_images':eligible_reference,
            'review_acceptance_rate':accepted/len(pred_masks),
            'paired_eligible_coverage':len(errs)/max(eligible_reference,1),
            'fitted_pairs':len(errs),'median_endpoint_error_px':float(np.median(errs)) if errs else None,
            'p90_endpoint_error_px':float(np.quantile(errs,.9)) if errs else None,
            'median_absolute_curvature_error_inv_px':float(np.median(curvature)) if curvature else None,
            'median_predicted_quadratic_rms_px':float(np.median(rms)) if rms else None,
            'metric_units':'2D synthetic image pixels only',
            'particle_physics_parameters_available':False}
