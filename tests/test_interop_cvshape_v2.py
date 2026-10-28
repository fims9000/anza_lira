"""OpenCV 4.x compatibility without modifying frozen S85 interop v1 records."""
import cv2
import numpy as np
import pytest
from bubble_chamber.interop_api import process_image,ScoreBinding
from bubble_chamber.interop_cv_compat import (
    normalize_cv_lines,OpenCVLSDCompatibleDetector,OpenCVHoughCompatibleDetector
)


def test_numpy_line_normalization_does_not_change_coordinates():
    base=np.asarray([[15.,20.,80.,20.],[15.,70.,80.,73.]],dtype=np.float32)
    a=normalize_cv_lines(base)
    b=normalize_cv_lines(base.reshape(2,1,4))
    assert len(a)==len(b)==2
    assert all(np.array_equal(x[0],y[0]) and np.array_equal(x[1],y[1]) for x,y in zip(a,b))
    assert normalize_cv_lines(None)==[]
    assert normalize_cv_lines(np.empty((0,1,4),np.float32))==[]
    with pytest.raises(ValueError):normalize_cv_lines(np.full((2,4),np.nan))
    with pytest.raises(ValueError):normalize_cv_lines(np.ones((2,2),np.float32))


@pytest.mark.parametrize('shape',[(2,4),(2,1,4)])
def test_lsd_compatible_detector_handles_both_shapes(monkeypatch,shape):
    data=np.asarray([[20.,50.,100.,50.],[20.,90.,100.,94.]],np.float32).reshape(shape)
    class Fake:
        def detect(self,_):return (data,None,None,None)
    monkeypatch.setattr(cv2,'createLineSegmentDetector',lambda *_:Fake())
    detector=OpenCVLSDCompatibleDetector()
    gray=np.full((192,192),200,np.uint8)
    lines=detector.detect(gray)
    assert len(lines)==2
    assert {tuple(p[0]) for p in lines}=={(20.,50.),(20.,90.)}
    assert {tuple(np.round(p[-1],3)) for p in lines}=={(100.,50.),(100.,94.)}
    b=ScoreBinding('TOY_ONLY',detector.detector_id,lambda rows:np.ones(len(rows)),
                   cutoff=1.,strategy='none')
    result=process_image(gray,detector,b)
    assert len(result.lines)==2 and not result.joins
    assert not result.physical_auto_joins_authorized


@pytest.mark.parametrize('shape',[(2,4),(2,1,4)])
def test_hough_compatible_detector_handles_both_shapes(monkeypatch,shape):
    data=np.asarray([[20,50,100,50],[20,90,100,94]],np.int32).reshape(shape)
    monkeypatch.setattr(cv2,'HoughLinesP',lambda *a,**k:data)
    detector=OpenCVHoughCompatibleDetector()
    lines=detector.detect(np.full((192,192),200,np.uint8))
    assert len(lines)==2
    assert {tuple(p[0]) for p in lines}=={(20.,50.),(20.,90.)}
    assert {tuple(np.round(p[-1],3)) for p in lines}=={(100.,50.),(100.,94.)}


def test_v1_calibration_cannot_be_used_with_v2_detector():
    det=OpenCVLSDCompatibleDetector()
    old=ScoreBinding('S85_TOY_ONLY_LEGACY','opencv-lsd-v1',lambda rows:np.ones(len(rows)),.7)
    with pytest.raises(ValueError,match='Detector/score'):
        process_image(np.full((192,192),210,np.uint8),det,old)
