import inspect
import pytest
import numpy as np
from scripts.research.bubble_chamber.s78_calibrate_then_test_exclusive_graph import (
    infer_arms_from_pixels,CAL_SEEDS,TEST_SEEDS,N_CAL,N_TEST,choose_on_CAL,
    SCHEMA,ARMS)


def test_frozen_seeds_are_new_and_disjoint():
    hist={20267701,20267711,20267703,20267704,20267713,20267714}
    fresh=set(CAL_SEEDS.values())|set(TEST_SEEDS['005'])|set(TEST_SEEDS['006'])
    assert len(fresh)==6 and not fresh.intersection(hist)
    assert (N_CAL,N_TEST)==(16,24)


def test_blind_inference_interface_has_no_source_gt():
    assert list(inspect.signature(infer_arms_from_pixels).parameters)==['gray','scorer','cutoff','graph_arm']
    assert 'SYNTHETIC_ONLY' in SCHEMA
    with pytest.raises(TypeError):infer_arms_from_pixels(np.zeros((192,192),np.uint8),None,.7,'s39_original',source_nodes=[])


def test_cal_only_selection_explicit_failure_falls_back():
    base={'selected_synthetic_correct_links':10,'synthetic_noncorrect_selected':15,
          'complete_three_piece_paths_synthetic':2,'solver_failure':0}
    controls={'baseline':{'clean':dict(base),'blur_sigma_11':dict(base)},
              'alternative_strict':{'clean':dict(base),'blur_sigma_11':dict(base)},
              'alternative_relaxed':{'clean':dict(base),'blur_sigma_11':dict(base)}}
    assert choose_on_CAL(controls)['CAL_selected']=='baseline'
    controls['alternative_strict']['blur_sigma_11']['selected_synthetic_correct_links']=12
    assert choose_on_CAL(controls)['CAL_selected']=='alternative_strict'
    controls['alternative_strict']['clean']['synthetic_noncorrect_selected']=16
    assert choose_on_CAL(controls)['CAL_selected']=='baseline'
