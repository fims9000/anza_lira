"""S68 CAL-only gates and no-truth leakage contracts."""
import inspect
import numpy as np
import pytest
from scripts.research.bubble_chamber.s68_benchmark_branch_preservation import (
    choose_calibration_only, infer_pixel_only, ARMS, CAL_SEEDS, TEST_SEEDS,
)


def test_calibration_gate_prevents_risky_promotion():
    v={'selected_synthetic_correct_links':10,'synthetic_noncorrect_selected':3,
       'scene_any_noncorrect':2,'complete_three_piece_paths_synthetic':1}
    a={name:v.copy() for name in ARMS}
    a[ARMS[2]]['selected_synthetic_correct_links']=12
    a[ARMS[2]]['synthetic_noncorrect_selected']=4
    assert choose_calibration_only(a)['CAL_only_selected']==ARMS[0]
    a[ARMS[2]]['synthetic_noncorrect_selected']=3
    assert choose_calibration_only(a)['CAL_only_selected']==ARMS[2]
    a[ARMS[2]]['complete_three_piece_paths_synthetic']=0
    assert choose_calibration_only(a)['CAL_only_selected']==ARMS[0]
    del a[ARMS[3]]
    with pytest.raises(ValueError):choose_calibration_only(a)


def test_fresh_seed_lineage_and_inference_signature():
    assert set(CAL_SEEDS.values()).isdisjoint({v for xs in TEST_SEEDS.values() for v in xs})
    assert len(set(v for xs in TEST_SEEDS.values() for v in xs))==4
    parameters=set(inspect.signature(infer_pixel_only).parameters)
    assert parameters=={'gray','scorer','cutoff','arm'}
    assert not parameters&{'source_frames','truth','scene','synthetic_gt','nodes'}
    with pytest.raises(ValueError):
        infer_pixel_only(np.zeros((192,192),np.float32), lambda x:None,.5,ARMS[2])
    with pytest.raises(ValueError):
        infer_pixel_only(np.zeros((192,192),np.uint8),lambda x:None,.5,'invalid')
