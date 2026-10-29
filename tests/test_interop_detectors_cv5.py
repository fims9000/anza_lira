"""Generated-only OpenCV 4/5 segment-shape compatibility; no native H50Z."""
import cv2
import numpy as np
import pytest

from bubble_chamber import interop_detectors_cv5 as compat


def photo():
    result=np.full((192,192),200,np.uint8)
    cv2.line(result,(22,50),(160,51),25,2)
    return result


def test_real_installed_opencv_detects_or_abstains_without_crash():
    image=photo()
    old=image.copy()
    output=compat.OpenCVLSDDetectorCompat(max_lines=12).detect(image)
    assert isinstance(output,list)
    assert len(output)<=12
    assert np.array_equal(image,old)
    assert all(np.asarray(x).ndim==2 and x.shape[1]==2 and np.isfinite(x).all() for x in output)


@pytest.mark.parametrize('shape',[(2,4),(2,1,4)])
def test_opencv_4_and_5_segment_array_shapes_normalized(monkeypatch,shape):
    pixels=photo()
    raw=np.asarray([[20,80,160,81],[30,120,170,121]],np.float32).reshape(shape)
    class Fake:
        def detect(self,x):
            assert np.array_equal(x,pixels)
            return (raw,None,None,None)
    monkeypatch.setattr(compat.cv2,'createLineSegmentDetector',lambda _:Fake())
    items=compat.OpenCVLSDDetectorCompat(max_lines=12).detect(pixels)
    assert len(items)==2
    assert all(q.shape[1]==2 and len(q)>=8 for q in items)


def test_none_and_nonfinite_fail_closed(monkeypatch):
    class Fake:
        def __init__(self,value):self.value=value
        def detect(self,x):return (self.value,None,None,None)
    monkeypatch.setattr(compat.cv2,'createLineSegmentDetector',lambda _:Fake(None))
    assert compat.OpenCVLSDDetectorCompat().detect(photo())==[]
    monkeypatch.setattr(compat.cv2,'createLineSegmentDetector',
                        lambda _:Fake(np.array([[1.,2.,float('nan'),4.]])))
    with pytest.raises(ValueError,match='nonfinite'):
        compat.OpenCVLSDDetectorCompat().detect(photo())
    with pytest.raises(ValueError):
        compat.OpenCVLSDDetectorCompat(max_lines=999).detect(photo())
