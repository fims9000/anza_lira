import cv2
import numpy as np
import pytest
from bubble_chamber.s82_local_instance_support import (
 ARMS,local_dark_stroke_support,thresholds_from_source_CAL,filter_extra_candidates,choose_CAL_only)


def line(y):
    return np.column_stack((np.linspace(30,161,55),np.full(55,y)))


def test_support_finds_dark_thin_line_and_order_invariance():
    gray=np.full((192,192),206,np.uint8)
    true=line(89)
    cv2.line(gray,(30,89),(161,89),28,2)
    signal=local_dark_stroke_support(gray,true)
    blank=local_dark_stroke_support(gray,line(111))
    assert signal>.7 and blank<.1,(signal,blank)
    assert np.isclose(signal,local_dark_stroke_support(gray,true[::-1]),atol=1e-6)
    assert np.isfinite(signal)


def test_input_guards_and_no_oracle_argument():
    image=np.full((192,192),207,np.uint8)
    with pytest.raises(ValueError):local_dark_stroke_support(image.astype(float),line(90))
    with pytest.raises(ValueError):local_dark_stroke_support(image[:100],line(90))
    with pytest.raises(TypeError):local_dark_stroke_support(image,line(90),gt=[1,2])
    assert local_dark_stroke_support(image,np.column_stack((np.linspace(3,10,20),np.full(20,3))))==0.


def test_pixel_only_thresholds_and_full_source_identity():
    th=thresholds_from_source_CAL([.2,.4,.6,.8])
    assert 0<=th['s82_local_q25']<=th['s82_local_q50']<=1
    assert all(x==1 for x in thresholds_from_source_CAL([]).values())
    with pytest.raises(ValueError):thresholds_from_source_CAL([float('nan')])
    gray=np.full((192,192),200,np.uint8)
    a,b=line(89),line(111)
    cv2.line(gray,(30,111),(161,111),20,2)
    plus=filter_extra_candidates(gray,[a],[a,b],'s82_local_q50',th)
    assert plus['polylines'][0] is a
    assert plus['extra_accepted']==1 and plus['extra_considered']==1
    no=filter_extra_candidates(gray,[a],[a,b],'baseline',th)
    assert no['polylines']==[a] and no['extra_accepted']==0
    with pytest.raises(ValueError):filter_extra_candidates(gray,[a],[a.copy(),b],'s82_local_q25',th)


def test_CAL_gate_prevents_false_join_growth_and_can_select():
    base=dict(selected_synthetic_correct_links=2,synthetic_noncorrect_selected=3,
              complete_three_piece_paths_synthetic=1,shape_true_fragments_matched=12,
              solver_failure=0,extra_accepted=0)
    rows={a:{c:base.copy() for c in ('clean','blur_sigma_11')} for a in ARMS}
    assert choose_CAL_only(rows)['CAL_selected']=='baseline'
    rows['s82_local_q25']['blur_sigma_11'].update(
        selected_synthetic_correct_links=5,shape_true_fragments_matched=15,
        synthetic_noncorrect_selected=4)
    assert choose_CAL_only(rows)['CAL_selected']=='baseline'
    rows['s82_local_q25']['blur_sigma_11']['synthetic_noncorrect_selected']=3
    assert choose_CAL_only(rows)['CAL_selected']=='s82_local_q25'
