import inspect
import numpy as np
import pytest
from bubble_chamber.s79_pixel_gate_rescue import ARMS
from scripts.research.bubble_chamber.s79_calibrate_then_test_gate_rescue import (
    infer_arms_from_pixels,choose_on_CAL,CAL_SEEDS,TEST_SEEDS,N_CAL,N_TEST,SCHEMA,PROTOCOL,OPERATOR)


def test_frozen_source_seeds_different_from_S78_and_test():
    a=set(CAL_SEEDS.values())|set(sum((list(s) for s in TEST_SEEDS.values()),[]))
    assert len(a)==6 and a.isdisjoint({20267801,20267811,20267803,20267804,20267813,20267814})
    assert N_CAL==16 and N_TEST==24
    assert 'S79_FROZEN_PIXEL_SUPPORTED_GATE_RESCUE' in PROTOCOL.name and OPERATOR.name=='s79_pixel_gate_rescue.py'
    assert 'SYNTHETIC_ONLY' in SCHEMA


def test_pixel_inference_has_no_ref_argument():
    assert list(inspect.signature(infer_arms_from_pixels).parameters)==['gray','scorer','cutoff','graph_arm']
    with pytest.raises(TypeError):infer_arms_from_pixels(np.zeros((192,192),np.uint8),None,.84,'s39_original',synthetic_truth=[])


def test_cal_gate_fails_when_wrong_joins_increase():
    base={'selected_synthetic_correct_links':8,'synthetic_noncorrect_selected':7,
          'complete_three_piece_paths_synthetic':1,'solver_failure':0}
    q={a:{'clean':dict(base),'blur_sigma_11':dict(base)} for a in ARMS}
    assert choose_on_CAL(q)['CAL_selected']=='baseline'
    q['rescue_strict']['blur_sigma_11']['selected_synthetic_correct_links']=10
    assert choose_on_CAL(q)['CAL_selected']=='rescue_strict'
    q['rescue_strict']['clean']['synthetic_noncorrect_selected']=8
    assert choose_on_CAL(q)['CAL_selected']=='baseline'
