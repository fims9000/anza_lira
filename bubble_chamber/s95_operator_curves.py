"""S95: source-locked operator 2D curve guides, not dense track-mask truth.

Accepts explicit pixel-coordinate operator parabolas or measured same-track
points only when a named BMP source AND its SHA are independently aligned.
Human first/last points alone cannot determine unique parabola coefficients.
The produced image curves are GUIDE overlays and partial geometric references,
never physical trajectory/particle ID or a full pixel-segmentation GT.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree


SCHEMA="S95_EXPLICIT_OPERATOR_2D_CURVES_V1"


def _point(x,what):
    p=np.asarray(x,dtype=float)
    if p.shape!=(2,) or not np.isfinite(p).all():
        raise ValueError("Invalid finite 2D point for "+what)
    return p


def _bounded_xy(pts,source_shape):
    h,w=source_shape
    q=np.asarray(pts,dtype=float)
    if (q.ndim!=2 or q.shape[1]!=2 or not np.isfinite(q).all()
        or not (np.all(q[:,0]>=0) and np.all(q[:,0]<w)
                and np.all(q[:,1]>=0) and np.all(q[:,1]<h))):
        raise ValueError("Operator points outside exact source photo")


def _coefficients(values,name):
    arr=np.asarray(values,dtype=float)
    if arr.shape!=(3,) or not np.isfinite(arr).all():
        raise ValueError("Expected 3 finite polynomial coefficients: "+name)
    return arr


def _polyval_cba(coef,u):
    return coef[0]+coef[1]*u+coef[2]*u*u


def fit_measured_polyline(points, *, min_points=5, max_rms_px=5.):
    """Parametric x(u), y(u), fitted to an ordered same-track annotation.

    This is not an operator-supplied equation. When a 2D curve doubles back
    or spirals, quadratic rejection keeps original clicks as the guide.
    """
    p=np.asarray(points,dtype=float)
    if (p.ndim!=2 or p.shape[1]!=2 or len(p)<min_points
        or not np.isfinite(p).all()):
        raise ValueError("Need >=5 valid ordered operator measurements")
    steps=np.linalg.norm(np.diff(p,axis=0),axis=1)
    if np.any(steps<1e-5):
        raise ValueError("Repeated consecutive measured points")
    arc=np.r_[0.,np.cumsum(steps)]
    if arc[-1]<=0:
        raise ValueError("Degenerate operator measurements")
    u=arc/arc[-1]
    # Endpoints are expert-measured anchors: enforce them EXACTLY. A general
    # unconstrained least-squares quadratic can drift away from the start/end.
    # P(u)=(1-u)*start + u*end + bend*u*(1-u).
    base=(1.-u[:,None])*p[0]+u[:,None]*p[-1]
    basis=u*(1.-u)
    bend=(basis[:,None]*(p-base)).sum(axis=0)/max(float(basis@basis),1e-12)
    cx=np.array([p[0,0],p[-1,0]-p[0,0]+bend[0],-bend[0]])
    cy=np.array([p[0,1],p[-1,1]-p[0,1]+bend[1],-bend[1]])
    predicted=np.column_stack((_polyval_cba(cx,u),_polyval_cba(cy,u)))
    rms=float(np.sqrt(np.mean(np.sum((predicted-p)**2,axis=1))))
    return {
        "x_coefficients_c_b_a":cx.tolist(),
        "y_coefficients_c_b_a":cy.tolist(),
        "rms_to_measured_points_px":round(rms,5),
        "operator_measured_points":len(p),
        "quadratic_fit_usable":bool(rms<=max_rms_px),
        "original_measured_polyline_xy":p.tolist(),
        "fit_is_not_operator_supplied_equation":True,
    }


def sample_operator_record(row:dict,*,shape_yx,n:int=200,max_end_error_px:2.):
    """Sample a manually supplied curve under a strict explicit format.

    Accepted kinds:
      parametric_quadratic: x(u),y(u) c+b*u+a*u^2, u=[0,1].
      local_quadratic: v=a*u^2+b*u+c with orthonormal (u,v) axes.
      measured_points: operator same-track measured coordinates; quadratic
        only when fit RMS <= specified tolerance, otherwise retain polyline.
    """
    if not isinstance(row,dict) or not isinstance(n,int) or not 32<=n<=4096:
        raise ValueError("Invalid S95 operator curve record/sample count")
    if not isinstance(row.get("track_id"),str) or not row["track_id"].strip():
        raise ValueError("Operator curve has no track_id")
    model=row.get("kind")
    if model not in ("parametric_quadratic","local_quadratic","measured_points"):
        raise ValueError("Unsupported operator curve type; never infer a parabola from endpoints")
    start=_point(row.get("start_xy"),"operator start")
    end=_point(row.get("end_xy"),"operator end")
    u=np.linspace(0.,1.,n)
    fit=None
    if model=="parametric_quadratic":
        cx=_coefficients(row.get("x_coefficients_c_b_a"),"x")
        cy=_coefficients(row.get("y_coefficients_c_b_a"),"y")
        path=np.column_stack((_polyval_cba(cx,u),_polyval_cba(cy,u)))
        param={"x_coefficients_c_b_a":cx.tolist(),"y_coefficients_c_b_a":cy.tolist()}
        status="EXPLICIT_OPERATOR_QUADRATIC_SOURCE"
    elif model=="local_quadratic":
        origin=_point(row.get("principal_origin_xy"),"local origin")
        axis=_point(row.get("u_axis_xy"),"u-axis")
        normal=_point(row.get("normal_axis_xy"),"normal")
        if (abs(np.linalg.norm(axis)-1.)>.01 or
            abs(np.linalg.norm(normal)-1.)>.01 or
            abs(float(axis@normal))>.01):
            raise ValueError("Local quadratic requires orthonormal pixel axes")
        ab=_coefficients(row.get("quadratic_v_a_b_c"),"v(u)")
        lohi=np.asarray(row.get("u_start_end"),dtype=float)
        if lohi.shape!=(2,) or not np.isfinite(lohi).all() or lohi[0]==lohi[1]:
            raise ValueError("Local u_start_end is missing/degenerate")
        pos=np.linspace(*lohi,n)
        vv=ab[0]*pos*pos+ab[1]*pos+ab[2]
        path=origin[None,:]+pos[:,None]*axis[None,:]+vv[:,None]*normal[None,:]
        param={"principal_origin_xy":origin.tolist(),
               "u_axis_xy":axis.tolist(),"normal_axis_xy":normal.tolist(),
               "quadratic_v_a_b_c":ab.tolist(),"u_start_end":lohi.tolist()}
        status="EXPLICIT_OPERATOR_LOCAL_QUADRATIC_SOURCE"
    else:
        p=np.asarray(row.get("points_xy"),dtype=float)
        fit=fit_measured_polyline(p,max_rms_px=float(row.get("max_rms_px",5.)))
        if fit["quadratic_fit_usable"]:
            cx=np.asarray(fit["x_coefficients_c_b_a"])
            cy=np.asarray(fit["y_coefficients_c_b_a"])
            path=np.column_stack((_polyval_cba(cx,u),_polyval_cba(cy,u)))
            status="FITTED_QUADRATIC_FROM_OPERATOR_MEASUREMENTS"
        else:
            d=np.linalg.norm(np.diff(p,axis=0),axis=1)
            arc=np.r_[0,np.cumsum(d)]
            positions=np.linspace(0,arc[-1],n)
            path=np.column_stack([np.interp(positions,arc,p[:,i]) for i in (0,1)])
            status="MEASURED_POINTS_POLYLINE_NONQUADRATIC_NO_FORCED_PARABOLA"
        param=fit
    _bounded_xy(path,shape_yx)
    discrepancy=[
        float(np.linalg.norm(path[0]-start)),
        float(np.linalg.norm(path[-1]-end)),
    ]
    if max(discrepancy)>max_end_error_px:
        raise ValueError("Supplied quadratic and start/end disagree in source pixels")
    return {
        "track_id":row["track_id"],"curve_kind":model,
        "curve_status":status,
        "start_xy_source":start.tolist(),"end_xy_source":end.tolist(),
        "sampled_guide_xy_source":np.round(path,5).tolist(),
        "endpoint_discrepancy_px":np.round(discrepancy,5).tolist(),
        "model_parameters":param,
        "operator_geometry_reference_only":True,
        "not_a_dense_pixel_mask":True,
        "not_complete_particle_track_gt":True,
    }


def validate_operator_curve_package(payload:dict,*,source_bmp_name:str,
                                    source_bmp_sha256:str,
                                    source_shape_yx:tuple[int,int]|list[int]):
    """Always bind operator 2D data to EXACT photo hash, not same film prefix."""
    if not isinstance(payload,dict) or payload.get("schema")!=SCHEMA:
        raise ValueError("Unsupported S95 operator curve package schema")
    if (payload.get("source_bmp_name")!=source_bmp_name
        or payload.get("source_bmp_sha256")!=source_bmp_sha256
        or payload.get("image_shape_yx")!=list(source_shape_yx)):
        raise ValueError("Operator curve/source BMP name, hash or dimensions mismatch")
    if not isinstance(payload.get("explicit_photo_alignment_verified"),bool):
        raise ValueError("Explicit photo alignment decision required")
    if not payload["explicit_photo_alignment_verified"]:
        # The 070 visualMetro and H50Z film series must never be matched
        # just because coordinates look geometrically similar.
        raise ValueError("Operator curve photo/view identity not independently verified")
    rows=payload.get("curves")
    if not isinstance(rows,list):
        raise ValueError("Missing operator curves array")
    results=[sample_operator_record(item,shape_yx=source_shape_yx) for item in rows]
    ids=[r["track_id"] for r in results]
    if len(ids)!=len(set(ids)):
        raise ValueError("Repeated operator track_id in one source photo")
    return results


def operator_curve_support(candidate,curves,*,radius_px=3.,min_margin=.12):
    """Source coordinate identity support, descriptive only, no physics label.

    Reject close competitor guides; never assert a missing human-drawn line
    means a true negative.
    """
    path=np.asarray(candidate,dtype=float)
    if path.ndim!=2 or path.shape[1]!=2 or len(path)<3 or not np.isfinite(path).all():
        raise ValueError("Invalid candidate centerline")
    if not 0<radius_px<=30 or not 0<=min_margin<=1:
        raise ValueError("Invalid operator guide matching limits")
    scored=[]
    for c in curves:
        guide=np.asarray(c["sampled_guide_xy_source"],dtype=float)
        dist=cKDTree(guide).query(path)[0]
        fraction=float(np.mean(dist<=radius_px))
        # Same operator track can be covered by multiple S86 fragments.
        nearest=cKDTree(path).query(guide)[0]
        coverage=float(np.mean(nearest<=radius_px))
        scored.append((fraction,coverage,c["track_id"]))
    if not scored:
        return {"status":"NO_OPERATOR_CURVES","operator_track_id":None,
                "candidate_point_support":None,"operator_curve_coverage":None,
                "operator_support_is_not_physical_truth":True}
    ranked=sorted(scored,key=lambda p:(-p[0],-p[1],p[2]))
    best=ranked[0]
    other=ranked[1][0] if len(ranked)>1 else 0.
    decisive=(best[0]>=.50 and (len(ranked)==1 or best[0]-other>=min_margin))
    status=("NEAR_SINGLE_OPERATOR_GUIDE_REVIEW" if decisive else
            "AMBIGUOUS_MULTIPLE_OPERATOR_GUIDES" if best[0]>=.50 else
            "NOT_NEAR_OPERATOR_GUIDE_UNLABELED_NOT_NEGATIVE")
    return {
        "status":status,
        "operator_track_id":best[2] if decisive else None,
        "candidate_point_support":round(best[0],5),
        "operator_curve_coverage":round(best[1],5),
        "runner_up_support":round(other,5),
        "operator_support_is_not_physical_truth":True,
    }
