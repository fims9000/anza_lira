"""OpenCV 4/5 compatible LSD line detector for PRIVATE BMP review.

This is a *versioned* decoder of OpenCV LSD segment arrays, not a change to
the frozen original opencv-lsd-v1 baseline from S85 research receipts.

OpenCV variants may return (N,1,4), (N,4) or an empty None. Both forms encode
(x0, y0, x1, y1). No trained checkpoint, native GT or auto-joining is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from .interop_api import validate_pixels, MAX_LINES
from .interop_detectors import _segments_to_lines


@dataclass(frozen=True)
class OpenCVLSDDetectorCompat:
    max_lines: int = 12
    detector_id: str = 'opencv-lsd-v2-opencv4-5-compatible'

    def detect(self, gray: np.ndarray):
        validate_pixels(gray)
        if type(self.max_lines) is not int or not 1 <= self.max_lines <= MAX_LINES:
            raise ValueError('Invalid bounded line candidate budget')
        detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        result = detector.detect(np.array(gray, copy=True))
        if not isinstance(result, (tuple, list)) or not result:
            raise ValueError('OpenCV LSD returned no expected line-array tuple')
        raw = result[0]
        if raw is None:
            return []
        segments = np.asarray(raw, dtype=np.float64)
        if segments.size == 0:
            return []
        if segments.ndim not in (2, 3) or segments.size % 4 != 0 or \
           not np.isfinite(segments).all():
            raise ValueError('Unexpected nonfinite or malformed OpenCV LSD segments')
        segments = segments.reshape(-1,4)
        return _segments_to_lines([(s[:2],s[2:]) for s in segments],self.max_lines)
