from pathlib import Path
import numpy as np
import pytest
from scripts.research.bubble_chamber.s63_calibrate_then_new_test_pixel_curves import (
    SCHEMA,CAL_SEEDS,TEST_SEEDS,SCENES,scene_metrics)


def test_new_seeds_and_cal_test_phases_are_disjoint():
    assert set(sum((list(x) for x in CAL_SEEDS.values()),[])).isdisjoint(
      set(sum((list(x) for x in TEST_SEEDS.values()),[])))
    assert SCENES=={'cal':12,'test':24}
    assert 'SYNTHETIC_ONLY' in SCHEMA


def test_predicted_before_evaluation_and_all_four_arms():
    a=np.full((192,192),200,dtype=np.uint8)
    nodes=[np.column_stack((np.linspace(25,160,31),np.full(31,20.+i*20))) for i in range(7)]
    x=scene_metrics(a,nodes)
    assert len(x)==4
    assert all(z['proposal_count']<=32 for z in x.values())
    assert all(z['synthetic_decoy_fragment_usable'] in (0,1) for z in x.values())
