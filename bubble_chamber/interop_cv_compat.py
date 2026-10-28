"""Versioned OpenCV 4.x detector interop for both (N,4) and (N,1,4) outputs.

IMPORTANT: detector_id is NEW (v2) because the implementation differs from
frozen S85 interop v1. Never reuse v1 training or CAL with this detector.
Historic experiments import interop_detectors.py, left BYTE-IDENTICAL.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from .interop_api import validate_pixels, MAX_LINES
from .interop_detectors import _segments_to_lines


def normalize_cv_lines(raw):
    """Convert OpenCV LSD/Hough formats to endpoint pairs without modifying XY."""
    if raw is None:
        return []
    array=np.asarray(raw,dtype=np.float64)
    if array.size==0:
        return []
    if array.ndim not in (2,3) or array.shape[-1]!=4 or not np.isfinite(array).all():
        raise ValueError('OpenCV lines must be finite Nx4 or Nx1x4')
    return [(x[:2],x[2:]) for x in array.reshape(-1,4)]


@dataclass(frozen=True)
class OpenCVLSDCompatibleDetector:
    """New, detector-specific training/calibration required (v2)."""
    max_lines: int = 12
    detector_id: str = 'opencv-lsd-cvshape-v2'

    def detect(self,gray):
        validate_pixels(gray)
        if not 1<=self.max_lines<=MAX_LINES:raise ValueError('Invalid candidate budget')
        lines=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(np.array(gray,copy=True))
        return _segments_to_lines(normalize_cv_lines(lines[0]),self.max_lines)


@dataclass(frozen=True)
class OpenCVHoughCompatibleDetector:
    """New, detector-specific training/calibration required (v2)."""
    max_lines: int = 12
    detector_id: str = 'opencv-hough-p-cvshape-v2'

    def detect(self,gray):
        validate_pixels(gray)
        if not 1<=self.max_lines<=MAX_LINES:raise ValueError('Invalid candidate budget')
        edges=cv2.Canny(np.array(gray,copy=True),50,115,L2gradient=True)
        lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=13,minLineLength=18,maxLineGap=6)
        return _segments_to_lines(normalize_cv_lines(lines),self.max_lines)
