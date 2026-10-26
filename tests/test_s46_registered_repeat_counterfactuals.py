import numpy as np
import cv2
import pytest
from scripts.research.bubble_chamber.s46_registered_repeat_counterfactuals import (
    inject,generate_path,challenge_once,SCENARIOS,perpendicular_shift)


def test_exact_same_geometry_is_used_in_each_synthetic_control():
    base=np.full((160,160),215,dtype=np.uint8)
    path=np.column_stack([np.linspace(15,145,70),np.full(70,85)])
    before=base.copy()
    one=inject(base,path)
    assert np.array_equal(base,before)
    assert (one[85,70]<base[85,70])
    assert (inject(base,perpendicular_shift(path,8.))[85,70]==base[85,70])
    assert (inject(base,perpendicular_shift(path,8.))[93,70]<base[93,70])
    assert set(SCENARIOS)=={'single_view_only','identical_registered_repeat','normal_shifted_eight_px_repeat'}


def test_valid_single_vs_repeat_native_background_synthetic():
    a=np.full((192,192),205,dtype=np.uint8)
    p=np.column_stack([np.linspace(25,165,80),np.full(80,78)])
    one=challenge_once(a,a,p,'single_view_only')
    duplicate=challenge_once(a,a,p,'identical_registered_repeat')
    assert not one['registered_repeat_advisory']
    assert duplicate['registered_repeat_advisory']
    assert one['known_synthetic_geometry_not_native_GT']


def test_generate_path_avoids_synthetic_known_fiducial_region():
    rng=np.random.default_rng(11)
    zone=[{'photo':'005','center_xy_PRIVATE':[240,235],
           'machine_ambiguity_radius_px':35,'arms_angles_rad_PRIVATE':[],
           'arm_half_span_px':0,'arm_tolerance_px':5}]
    p=generate_path(rng,zone,[0,0],'005')
    assert p.shape==(100,2)
    from bubble_chamber.s40_fiducial_safety import fiducial_guard
    assert not fiducial_guard(p,[0,0],'005',zone)['fiducial_uncertainty_any']


def test_invalid_scenario_rejected():
    im=np.full((192,192),205,dtype='uint8')
    p=np.column_stack([np.linspace(25,155,40),np.full(40,70)])
    with pytest.raises(ValueError):challenge_once(im,im,p,'native_GT')
