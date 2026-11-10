import cv2
import numpy as np
import pytest

from bubble_chamber.s106_fiducial_quarantine import fiducial_quarantine,suppress_visual_only


def _scene(with_second_axis=True):
    a=np.full((450,450),206,dtype=np.uint8)
    cv2.line(a,(60,390),(390,60),50,6,cv2.LINE_AA)
    if with_second_axis:
        cv2.line(a,(60,60),(390,390),50,6,cv2.LINE_AA)
    cv2.line(a,(30,100),(170,100),80,2,cv2.LINE_AA)
    return a


def test_detects_two_arms_without_claiming_negative_source():
    source=_scene()
    mark=[{'label':'f4','xy_original_px':[225,225]}]
    mask,details=fiducial_quarantine(source,mark,(0,0,450,450))
    assert details[0]['status']=='TWO_AXES_SOURCE_SUPPORTED'
    assert mask[225,225]
    assert mask[160,160] and mask[160,290]
    assert not mask[100,30]
    overlay=cv2.cvtColor(source,cv2.COLOR_GRAY2BGR)
    overlay[:]=(0,0,255)
    clean,remaining=suppress_visual_only(source,overlay,np.ones_like(source,bool),mask)
    assert np.all(clean[mask,0]==source[mask])
    assert np.array_equal(remaining,~mask)


def test_single_axis_not_automatically_suppressed():
    source=_scene(False)
    mask,details=fiducial_quarantine(source,[{'label':'f4','xy_original_px':[225,225]}],(0,0,450,450))
    assert details[0]['status']!='TWO_AXES_SOURCE_SUPPORTED'
    assert not mask.any()


def test_no_mark_no_suppression():
    source=_scene()
    mask,details=fiducial_quarantine(source,[],(0,0,450,450))
    assert not mask.any() and details==[]


def test_invalid_crop_and_untrusted_label():
    with pytest.raises(ValueError,match='crop'):
        fiducial_quarantine(_scene(),[],(0,0,350,450))
    with pytest.raises(ValueError,match='fiducial-only'):
        fiducial_quarantine(_scene(),[{'label':'start','xy_original_px':[225,225]}],(0,0,450,450))
