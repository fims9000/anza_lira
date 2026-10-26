"""Evaluation-only S69 scope and seed controls."""
import inspect
import numpy as np
import pytest
from scripts.research.bubble_chamber.s69_replay_endpoint_audit import (
    FRESH_SEEDS,CAL_SEEDS,S68_HISTORIC_SEEDS,
    paired_delta,totals,sample,run,
)


def test_fresh_locked_seeds_disjoint():
    cal=set(CAL_SEEDS.values());history={v for t in S68_HISTORIC_SEEDS.values() for v in t}
    fresh={v for t in FRESH_SEEDS.values() for v in t}
    assert len(fresh)==4 and not(fresh&history) and not(fresh&cal)
    assert set(inspect.signature(sample).parameters)=={'tiles','templates','seed','n','scorer','cutoff'}
    assert set(inspect.signature(run).parameters)=={'visualmetro','tiles','phase','calibration'}


def test_paired_delta_grouped_by_scene_not_pixels():
    old=[dict(selected_synthetic_correct_links=0,selected_synthetic_wrong_endpoint=1,
              synthetic_noncorrect_selected=1,scene_any_noncorrect=1,
              complete_three_piece_paths_synthetic=0) for _ in range(4)]
    new=[dict(selected_synthetic_correct_links=1,selected_synthetic_wrong_endpoint=0,
              synthetic_noncorrect_selected=0,scene_any_noncorrect=0,
              complete_three_piece_paths_synthetic=1) for _ in range(4)]
    out=paired_delta(old,new,9)
    assert out['selected_synthetic_correct_links']['mean_delta_per_scene']==1
    assert out['selected_synthetic_correct_links']['scene_bootstrap_95pct']==[1.,1.]
    assert out['synthetic_noncorrect_selected']['mean_delta_per_scene']==-1
    with pytest.raises(ValueError):paired_delta(old,[],9)
