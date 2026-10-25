"""S29 non-certified geometrical/photometric measurements of proposed 2-D paths.

The measured input is a model candidate, not an expert validated particle track.
Outputs are PIXEL-SPACE proxies. No curvature-to-momentum calibration, 3-view
stereo matching, particle-ID or real annotation masks are created.
"""
from __future__ import annotations
import cv2
import numpy as np


def _as_path(points):
    pts=np.asarray(points,dtype='float64')
    if pts.ndim!=2 or pts.shape[1]!=2 or len(pts)<12 or not np.isfinite(pts).all():
        raise ValueError('Finite polyline with 12+ (x,y) points required')
    steps=np.linalg.norm(np.diff(pts,axis=0),axis=1)
    if np.max(steps)>50 or np.count_nonzero(steps>1e-5)<8:
        raise ValueError('Broken/degenerate candidate path')
    return pts


def parametric_quadratic_fit(points):
    """Fit both original-coordinate x(s), y(s), robust to vertical tracks.

    Normalized arclength u in [0,1]: x=a_x*u^2 + b_x*u + c_x, likewise y.
    This is a geometric approximation, NOT physically calibrated helix/impulse.
    """
    pts=_as_path(points)
    seg=np.linalg.norm(np.diff(pts,axis=0),axis=1)
    s=np.r_[0.,np.cumsum(seg)]
    length=float(s[-1]);u=s/length
    xc=np.polynomial.polynomial.polyfit(u,pts[:,0],2)
    yc=np.polynomial.polynomial.polyfit(u,pts[:,1],2)
    approx=np.stack([np.polynomial.polynomial.polyval(u,xc),np.polynomial.polynomial.polyval(u,yc)],axis=-1)
    residual=np.linalg.norm(approx-pts,axis=1)
    coef=np.stack([xc,yc],axis=0)
    return {'parameter':'normalized_arclength_u_0_to_1',
            'pixel_arclength':length,
            'x_coefficients_c_b_a':[float(v) for v in coef[0]],
            'y_coefficients_c_b_a':[float(v) for v in coef[1]],
            'fitted_start_xy':[float(v) for v in approx[0]],
            'fitted_end_xy':[float(v) for v in approx[-1]],
            'source_start_xy':[float(v) for v in pts[0]],
            'source_end_xy':[float(v) for v in pts[-1]],
            'rms_centerline_residual_px':float(np.sqrt(np.mean(residual**2))),
            'maximum_centerline_residual_px':float(residual.max()),
            'fitted_points_original_xy':approx.round(3).tolist()}


def photo_line_profile(gray:np.ndarray, points, *, lateral_radius:int=10, downsample:int=2, min_contrast:float=5.):
    """Estimate local darkness/width along unverified lines; preserve abstention.

    Reports pixel widths only in regions with an objectively darker center than
    both lateral shoulders. ``candidate_bubble_peaks`` is a signal descriptor,
    not an expert verified count of bubble objects or dE/dx.
    """
    if gray.ndim!=2 or gray.dtype!=np.uint8:
        raise ValueError('Expected source 8-bit grayscale photograph')
    pts=_as_path(points)[::downsample]
    if lateral_radius<5 or lateral_radius>20 or not 1<=downsample<=8 or len(pts)<6:
        raise ValueError('Invalid line-width sampling parameters')
    tang=np.gradient(pts,axis=0)
    norm=np.maximum(np.linalg.norm(tang,axis=1),1e-8)
    normals=np.stack([-tang[:,1]/norm,tang[:,0]/norm],axis=1)
    offsets=np.arange(-lateral_radius,lateral_radius+1,dtype=np.float32)
    xy=pts[:,None,:]+normals[:,None,:]*offsets[None,:,None]
    xs=xy[:,:,0].astype('float32');ys=xy[:,:,1].astype('float32')
    valid=((xs>=0)&(xs<gray.shape[1])&(ys>=0)&(ys<gray.shape[0])).all(axis=1)
    observed=cv2.remap(gray.astype('float32'),xs,ys,interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT,borderValue=255.)
    mid=lateral_radius
    # Local background from both shoulders. A nearby intersecting track may
    # contaminate it; this is why statistics are only diagnostic proxies.
    shoulders=np.concatenate([observed[:,:3],observed[:,-3:]],axis=1)
    bg=np.median(shoulders,axis=1)
    center=observed[:,mid]
    contrast=bg-center
    evidence=valid&(contrast>=min_contrast)
    widths=[]
    for k in np.flatnonzero(evidence):
        threshold=bg[k]-.5*contrast[k]
        line=observed[k]<=threshold
        left=mid;right=mid
        while left-1>=0 and line[left-1]:left-=1
        while right+1<len(line) and line[right+1]:right+=1
        widths.append(int(right-left+1))
    good_dark=np.maximum(contrast,0.)
    from scipy.signal import find_peaks
    signal=np.where(evidence,good_dark,0.)
    # Minimum 4 measured steps avoids double-counting tightly overlapping
    # samples; no physical microscope calibration or species inference.
    peaks,_=find_peaks(signal,distance=4,prominence=max(min_contrast,3.))
    gaps=[];count=0
    for yes in evidence:
        count=(count+1) if not yes else 0
        gaps.append(count)
    return {
        'sampling_lateral_radius_px':lateral_radius,
        'profiles_tested':int(len(pts)),
        'profiles_in_frame':int(valid.sum()),
        'profiles_with_dark_center':int(evidence.sum()),
        'fraction_with_dark_center':float(evidence.mean()),
        'maximum_unsupported_gap_samples':int(max(gaps,default=0)),
        'median_dark_center_contrast_8bit':float(np.median(contrast[evidence])) if evidence.any() else None,
        'median_apparent_width_px':float(np.median(widths)) if widths else None,
        'width_interquartile_range_px':[float(v) for v in np.quantile(widths,[.25,.75])] if widths else None,
        'candidate_bubble_peaks_unverified':int(len(peaks)),
        'physical_ionization_deduction_allowed':False,
        'source_is_expert_confirmed_track':False,
    }


def measure_candidate(gray:np.ndarray, candidate:dict)->dict:
    pts=candidate.get('path_points_original_xy')
    geometry=parametric_quadratic_fit(pts)
    profile=photo_line_profile(gray,pts)
    return {'schema':'s29_unconfirmed_2d_track_measurement_v1',
            'geometry':{k:v for k,v in geometry.items() if k!='fitted_points_original_xy'},
            'photometry':profile,
            'candidate_direction_deg':candidate.get('direction_angle_deg'),
            'candidate_hypothesis_score_uncalibrated':candidate.get('model_selection_score_UNCALIBRATED'),
            'expert_approved':False,'track_particle_ID':'UNKNOWN',
            'true_3d_momentum_or_mass_available':False,
            'coordinate_frame':'2D_CROP_PIXEL',
            'physics_use':'REVIEW_ONLY'}
