"""Reproducible interchangeable 2-D line extractors for S85 integration.

These are historical/classical baselines, not current H50Z trained models.
LSD/Hough favor straight structures; the existing Hessian ridge extractor is
more appropriate for curved thin tracks. Detector output is image-only.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from .interop_api import validate_pixels, MAX_LINES
from .s39_machine_auto import ridge_evidence, extract_ridges


def _long_segment(a,b):
    pa=np.asarray(a,float);pb=np.asarray(b,float)
    if np.linalg.norm(pa-pb)<18:return None
    n=max(8,min(70,int(np.linalg.norm(pa-pb)/2)+1))
    p=np.linspace(pa,pb,n)
    if np.any(p<0) or np.any(p>191):return None
    return p


def _segments_to_lines(raw,limit):
    # Stable score/order, independent of OpenCV's arbitrary output ordering.
    options=[]
    for a,b in raw:
        c=_long_segment(a,b)
        if c is None:continue
        # Canonicalize orientation (x then y) deterministically for logging.
        if tuple(c[0]) > tuple(c[-1]):c=c[::-1].copy()
        options.append((float(np.linalg.norm(c[-1]-c[0])),c))
    options.sort(key=lambda x:(-x[0],tuple(np.round(x[1][0],3)),tuple(np.round(x[1][-1],3))))
    chosen=[]
    for _,c in options:
        if any(float(np.linalg.norm(c[0]-p[0])+np.linalg.norm(c[-1]-p[-1]))<7 for p in chosen):
            continue
        chosen.append(c)
        if len(chosen)>=limit:break
    return chosen


@dataclass(frozen=True)
class HessianRidgeDetector:
    max_lines: int = 12
    detector_id: str = 'hessian-ridge-v1'

    def detect(self,gray):
        validate_pixels(gray)
        if not 1<=self.max_lines<=MAX_LINES:raise ValueError('Invalid line budget')
        maps=ridge_evidence(gray)
        return extract_ridges(maps,max_candidates=self.max_lines)


@dataclass(frozen=True)
class OpenCVLSDDetector:
    max_lines: int = 12
    detector_id: str = 'opencv-lsd-v1'

    def detect(self,gray):
        validate_pixels(gray)
        if not 1<=self.max_lines<=MAX_LINES:raise ValueError('Invalid line budget')
        # OpenCV LSD provides straight line segments; no GT or trained score.
        finder=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        result=finder.detect(np.array(gray,copy=True))
        raw=[] if result[0] is None else [(x[0,:2],x[0,2:]) for x in result[0]]
        return _segments_to_lines(raw,self.max_lines)


@dataclass(frozen=True)
class OpenCVHoughDetector:
    max_lines: int = 12
    detector_id: str = 'opencv-hough-p-v1'

    def detect(self,gray):
        validate_pixels(gray)
        if not 1<=self.max_lines<=MAX_LINES:raise ValueError('Invalid line budget')
        edges=cv2.Canny(np.array(gray,copy=True),50,115,L2gradient=True)
        h=cv2.HoughLinesP(edges,1,np.pi/180,threshold=13,minLineLength=18,maxLineGap=6)
        raw=[] if h is None else [(x[0,:2],x[0,2:]) for x in h]
        return _segments_to_lines(raw,self.max_lines)


DETECTOR_BASELINES=(HessianRidgeDetector,OpenCVLSDDetector,OpenCVHoughDetector)


@dataclass(frozen=True)
class S80ConsensusAugmenter:
    """Existing multiscale ridge extras; unvalidated for a *swapped* detector.

    Needs its own DETECTOR+S80-trained/calibrated scorer before use. The S80
    source method was evaluated originally only with its own 2-D detector.
    """
    augmenter_id: str = 's80-consensus-add6'

    def propose(self,gray,original):
        validate_pixels(gray)
        from .s80_multiscale_instances import augment_pixel_only
        res=augment_pixel_only(gray,list(original),'s80_consensus_add6')
        return res['polylines']
