import numpy as np
import cv2
import pytest
from bubble_chamber.s56_global_path_assembly import Edge
from bubble_chamber.s65_endpoint_guard import (
    endpoint_termination_ratio, edge_termination_risk,retain_by_endpoint_guard,
    choose_cal_only,THRESHOLD_GRID)


def geometry():
    def p(x,y):return np.column_stack((np.linspace(x,x+35,35),np.full(35,y)))
    return [p(20,60),p(70,60),p(20,125)]


def photo():
    gray=np.full((192,192),210,dtype=np.uint8)
    cv2.line(gray,(20,60),(55,60),40,2)
    cv2.line(gray,(70,60),(105,60),40,2)
    cv2.line(gray,(20,125),(55,125),40,2)
    return gray


def test_feature_port_orientation_and_pixels_only():
    im=photo();p=geometry()[0]
    a=endpoint_termination_ratio(im,p,-1)
    b=endpoint_termination_ratio(im,p[::-1],0)
    assert np.isfinite(a) and np.isclose(a,b)
    assert 0<=a<15
    with pytest.raises(ValueError):endpoint_termination_ratio(im.astype(float),p,-1)
    with pytest.raises(ValueError):endpoint_termination_ratio(im,p,1)


def test_filter_never_adds_new_proposals():
    gray=photo();curves=geometry()
    chosen=[Edge(0,1,-1,0,.9),Edge(0,2,0,-1,.91)]
    for threshold in THRESHOLD_GRID:
        result=retain_by_endpoint_guard(gray,curves,chosen,threshold)
        assert all(e in chosen for e in result)
        assert result==[e for e in chosen if edge_termination_risk(gray,curves,e)<=threshold]
    assert retain_by_endpoint_guard(gray,curves,chosen,None)==chosen
    with pytest.raises(ValueError):retain_by_endpoint_guard(gray,curves,chosen,.333)


def test_calibration_uses_frozen_grid_and_emits_fallback():
    input={None:{'correct':10,'noncorrect':20,'scenes':32}}
    input.update({v:{'correct':5,'noncorrect':15,'scenes':32} for v in THRESHOLD_GRID})
    assert choose_cal_only(input)['fallback_is_unfiltered_original_S64']
    input[.7]={'correct':7,'noncorrect':8,'scenes':32}
    assert choose_cal_only(input)['selected_CAL_only']==.7
    with pytest.raises(ValueError):choose_cal_only({None:input[None]})
