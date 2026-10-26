from bubble_chamber.s60_margin_calibration import MARGINS,select_margin
import pytest

def counts():
    return {str(m):dict(scenes=64,correct_selected=70,wrong_selected=5,
          any_wrong_scenes=5,full_paths_selected=20,solver_failed_scenes=0)
            for m in MARGINS}

def test_frozen_selection_rejects_small_gain_and_new_false_scenes():
    x=counts();x[str(.02)]['correct_selected']=77
    x[str(.01)].update(correct_selected=80,any_wrong_scenes=6)
    assert select_margin(x)['selected_margin_CAL_only']==.04

def test_frozen_selection_uses_cal_only_and_deterministic_tiebreaks():
    x=counts();x[str(.02)].update(correct_selected=80,any_wrong_scenes=5,full_paths_selected=22)
    x[str(.03)].update(correct_selected=80,any_wrong_scenes=4,full_paths_selected=21)
    assert select_margin(x)['selected_margin_CAL_only']==.03
    x[str(.03)]['solver_failed_scenes']=1
    assert select_margin(x)['selected_margin_CAL_only']==.02

def test_incomplete_or_invalid_cal_rejected():
    x=counts();x.pop(str(.01))
    with pytest.raises(ValueError):select_margin(x)
    x=counts();x[str(.01)]['scenes']=63
    with pytest.raises(ValueError):select_margin(x)
