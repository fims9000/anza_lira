import cv2
import numpy as np
import pytest
from bubble_chamber.s81_quality_gated_instances import (
 ARMS,QUANTILES,fine_coarse_ratio,thresholds_from_CAL_clean,activated,choose_cal_only)


def test_quality_pixels_not_labels_and_frozen_quantiles():
    a=np.full((192,192),192,dtype=np.uint8)
    a[40:100:3,40:150:3]=10
    b=cv2.GaussianBlur(a,(0,0),1.6)
    assert np.isfinite(fine_coarse_ratio(a)) and np.isfinite(fine_coarse_ratio(b))
    with pytest.raises(ValueError):fine_coarse_ratio(a.astype(float))
    with pytest.raises(ValueError):fine_coarse_ratio(a[:80,:80])
    assert set(QUANTILES)==set(ARMS[1:])


def test_cal_threshold_calibration_and_guard():
    t=thresholds_from_CAL_clean(list(np.arange(16,dtype=float)))
    assert 0<=t['s81_gate_q20']<=t['s81_gate_q40']
    assert activated(1.,1.) and not activated(1.1,1.)
    with pytest.raises(ValueError):thresholds_from_CAL_clean([1,2])
    with pytest.raises(ValueError):activated(float('nan'),1.)


def test_strict_CAL_promotion_does_not_increase_wrong():
    b={'selected_synthetic_correct_links':2,'synthetic_noncorrect_selected':5,
       'shape_true_fragments_matched':12,'complete_three_piece_paths_synthetic':1,
       'solver_failure':0,'quality_gate_triggered':0}
    data={arm:{c:dict(b) for c in ('clean','blur_sigma_11')} for arm in ARMS}
    assert choose_cal_only(data)['CAL_selected']=='baseline'
    data['s81_gate_q20']['blur_sigma_11'].update(selected_synthetic_correct_links=4,
        synthetic_noncorrect_selected=6,shape_true_fragments_matched=14)
    assert choose_cal_only(data)['CAL_selected']=='baseline'
    data['s81_gate_q20']['blur_sigma_11']['synthetic_noncorrect_selected']=5
    assert choose_cal_only(data)['CAL_selected']=='s81_gate_q20'
