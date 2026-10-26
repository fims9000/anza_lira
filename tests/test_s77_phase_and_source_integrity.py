"""Immutable CAL before TEST, baseline identity, experiment ID and source-only safety."""
from pathlib import Path
import numpy as np
import pytest
from bubble_chamber.s77_short_gap_instances import ARMS,repair_pixel_only
from scripts.research.bubble_chamber.s77_calibrate_then_test_short_gaps import (
 CAL_SEEDS,TEST_SEEDS,N_CAL,N_TEST,select_from_CAL,
 infer_arms_from_pixels,EXPERIMENT_ID,LOCKED_S72_CAL_SHA256,digest,SOURCE_CAL)


def _stats(correct,wrong,paths=0):
    return {'selected_synthetic_correct_links':correct,
            'synthetic_noncorrect_selected':wrong,
            'complete_three_piece_paths_synthetic':paths,
            'solver_failure':0}


def test_calibration_gates_freeze_and_require_both_conditions():
    cases={arm:{'clean':_stats(10,7,2),'blur_sigma_11':_stats(3,7,0)} for arm in ARMS}
    assert select_from_CAL(cases)['CAL_selected']=='baseline'
    cases['short_gap_strict']['blur_sigma_11']=_stats(5,7,0)
    assert select_from_CAL(cases)['CAL_selected']=='short_gap_strict'
    cases['short_gap_strict']['clean']=_stats(10,8,2)
    assert select_from_CAL(cases)['CAL_selected']=='baseline'
    cases['short_gap_strict']['clean']=_stats(9,7,2)
    assert select_from_CAL(cases)['CAL_selected']=='short_gap_strict'
    with pytest.raises(ValueError):select_from_CAL({})


def test_infer_arms_never_uses_source_identity(monkeypatch):
    import scripts.research.bubble_chamber.s77_calibrate_then_test_short_gaps as m
    gray=np.full((192,192),189,dtype=np.uint8)
    source={'polylines':[],'chosen':[],'pair_count':0,'solver_success':True,
            'physics_auto_joins_authorized':False}
    seen=[]
    def fake(pixels,scorer,cutoff,method):
        seen.append((pixels.shape,method))
        return source
    monkeypatch.setattr(m.s68,'infer_pixel_only',fake)
    out=infer_arms_from_pixels(gray,lambda rows:np.zeros(len(rows)),.8,'s39_original')
    assert set(out)==set(ARMS)
    assert all(z['physics_auto_joins_authorized'] is False and z['chosen']==[] for z in out.values())
    assert seen==[((192,192),'s39_original')]
    with pytest.raises(TypeError):infer_arms_from_pixels(gray,lambda x:[],.8,'s39_original',GT=[])


def test_source_freeze_and_new_scene_ids():
    assert digest(SOURCE_CAL)==LOCKED_S72_CAL_SHA256
    assert EXPERIMENT_ID=='ANZA-LIRA-BC-S77-20261026'
    assert (N_CAL,N_TEST)==(16,24)
    assert 'phase_scene_count_per_seed' in (Path(__file__).resolve().parents[1]/'scripts/research/bubble_chamber/s77_calibrate_then_test_short_gaps.py').read_text()
    test={seed for v in TEST_SEEDS.values() for seed in v}
    assert test.isdisjoint(set(CAL_SEEDS.values()))
    prior={20267603,20267604,20267613,20267614,
           20267503,20267504,20267513,20267514}
    assert test.isdisjoint(prior)
    assert sorted(test)==[20267703,20267704,20267713,20267714]
