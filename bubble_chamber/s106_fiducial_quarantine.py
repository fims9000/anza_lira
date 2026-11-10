"""S106: reversible visual-only quarantine of operator-identified fiducial crosses.

The source bitmap, original S88 mask and S105 identities remain unchanged.
A masked pixel is unknown for particle-trace interpretation, NOT certified
background. A real trace may overlap a fiducial; never delete source evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class FiducialConfig:
    search_radius_px: int = 305
    suppression_radius_px: int = 290
    min_line_length_px: int = 115
    max_center_distance_px: float = 20.0
    min_angle_separation_deg: float = 40.0
    max_angle_separation_deg: float = 135.0
    mask_half_width_px: float = 14.0
    hough_threshold: int = 45

    def validate(self):
        if not (100 <= self.search_radius_px <= 500
                and 80 <= self.suppression_radius_px <= self.search_radius_px
                and 40 <= self.min_line_length_px < self.search_radius_px*2
                and 2 <= self.max_center_distance_px <= 40
                and 15 <= self.min_angle_separation_deg <= 90
                and 90 <= self.max_angle_separation_deg <= 180
                and 2 <= self.mask_half_width_px <= 30
                and 10 <= self.hough_threshold <= 100):
            raise ValueError('invalid fiducial geometry budget')


def _axes_for_mark(gray:np.ndarray,xy:tuple[float,float],cfg:FiducialConfig):
    """Propose two real long dark-line axes near an independently clicked mark."""
    h,w=gray.shape
    cx,cy=map(float,xy)
    if not(0<=cx<w and 0<=cy<h):
        return None, 'OUTSIDE_CROP'
    radius=cfg.search_radius_px
    x0=max(0,int(math.floor(cx-radius)))
    y0=max(0,int(math.floor(cy-radius)))
    x1=min(w,int(math.ceil(cx+radius+1)))
    y1=min(h,int(math.ceil(cy+radius+1)))
    sub=gray[y0:y1,x0:x1]
    if min(sub.shape)<150:
        return None,'INSUFFICIENT_SOURCE_AREA'
    blurred=cv2.GaussianBlur(sub,(3,3),0)
    edges=cv2.Canny(blurred,30,85)
    raw=cv2.HoughLinesP(edges,1,np.pi/180,threshold=cfg.hough_threshold,
                        minLineLength=cfg.min_line_length_px,maxLineGap=20)
    if raw is None:
        return None,'NO_HOUGH_CANDIDATES'
    eligible=[]
    for X0,Y0,X1,Y1 in raw[:,0]:
        ax=float(X0+x0);ay=float(Y0+y0)
        bx=float(X1+x0);by=float(Y1+y0)
        dx=bx-ax;dy=by-ay
        length=math.hypot(dx,dy)
        dist=abs(dx*(cy-ay)-dy*(cx-ax))/max(1,length)
        if dist>cfg.max_center_distance_px:
            continue
        ux=dx/length;uy=dy/length
        # Keep only a substantial line spanning both sides of the operator mark.
        t0=(ax-cx)*ux+(ay-cy)*uy
        t1=(bx-cx)*ux+(by-cy)*uy
        both=min(abs(min(t0,t1)),abs(max(t0,t1)))
        if t0*t1>=0 or both<48:
            continue
        orientation=float(math.atan2(uy,ux)%math.pi)
        eligible.append({'angle_rad':orientation,'length':length,'distance':dist,
                         'score':length-2*dist,'endpoints':(ax,ay,bx,by)})
    if not eligible:
        return None,'NO_TWO_SIDED_AXES'
    eligible=sorted(eligible,key=lambda q:-q['score'])[:40]
    best=None
    for i,a in enumerate(eligible):
        for b in eligible[i+1:]:
            delta=abs(a['angle_rad']-b['angle_rad'])
            sep=math.degrees(min(delta,math.pi-delta))
            if not cfg.min_angle_separation_deg<=sep<=min(90,cfg.max_angle_separation_deg):
                continue
            # Keep both axes comparably strong; orthogonal angles are preferred.
            value=min(a['length'],b['length']) + 0.05*(a['length']+b['length']) + 0.5*sep
            if best is None or value>best[0]:
                best=(value,a,b)
    if best is None:
        return None,'ONLY_ONE_DIRECTION_OR_AMBIGUOUS'
    return (best[1],best[2]),'TWO_AXES_SOURCE_SUPPORTED'


def fiducial_quarantine(gray:np.ndarray,marks:list[dict],origin:tuple[int,int,int,int],
                         cfg:FiducialConfig=FiducialConfig()):
    """Mask only inferred calibration-cross corridors; no source pixels change."""
    cfg.validate()
    if gray.dtype!=np.uint8 or gray.ndim!=2:
        raise ValueError('expected uint8 source grayscale ROI')
    if len(origin)!=4 or origin[2]-origin[0]!=gray.shape[1] or origin[3]-origin[1]!=gray.shape[0]:
        raise ValueError('crop does not match original coordinate frame')
    if len(marks)>32:
        raise ValueError('too many fiducials in one bounded ROI')
    h,w=gray.shape
    out=np.zeros((h,w),dtype=bool)
    yy,xx=np.indices(gray.shape,dtype=np.float32)
    details=[]
    for mark in marks:
        if not str(mark.get('label','')).startswith('f'):
            raise ValueError('fiducial-only marks required')
        xy=mark.get('xy_original_px')
        if xy is None or len(xy)!=2:
            raise ValueError('missing source-bound fiducial coordinates')
        cx=float(xy[0])-origin[0];cy=float(xy[1])-origin[1]
        if not math.isfinite(cx) or not math.isfinite(cy):
            raise ValueError('nonfinite fiducial position')
        axes,status=_axes_for_mark(gray,(cx,cy),cfg)
        entry={'label':mark['label'],'xy_source_px':[float(xy[0]),float(xy[1])],
               'status':status,'not_ground_truth':True}
        if axes is not None:
            dist=np.hypot(xx-cx,yy-cy)
            local=np.zeros((h,w),bool)
            for a in axes:
                theta=a['angle_rad'];dx=math.cos(theta);dy=math.sin(theta)
                normal=np.abs(-(xx-cx)*dy+(yy-cy)*dx)
                local|=(normal<=cfg.mask_half_width_px)&(dist<=cfg.suppression_radius_px)
            out|=local
            entry['axes_deg']=[round(math.degrees(a['angle_rad']),2) for a in axes]
            entry['quarantine_area_px']=int(local.sum())
        details.append(entry)
    return out,details


def suppress_visual_only(original_gray:np.ndarray,visual_bgr:np.ndarray,
                         observed_mask:np.ndarray,quarantine_mask:np.ndarray):
    if original_gray.shape!=observed_mask.shape or original_gray.shape!=quarantine_mask.shape:
        raise ValueError('mask shape mismatch')
    if visual_bgr.shape!=(*original_gray.shape,3):
        raise ValueError('overlay shape mismatch')
    if not (original_gray.dtype==visual_bgr.dtype==np.uint8):
        raise ValueError('invalid image types')
    clean=visual_bgr.copy()
    source=cv2.cvtColor(original_gray,cv2.COLOR_GRAY2BGR)
    clean[quarantine_mask]=source[quarantine_mask]
    unquarantined=observed_mask.astype(bool)&~quarantine_mask
    return clean,unquarantined
