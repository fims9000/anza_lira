import numpy as np
import pytest
from bubble_chamber.s30_trajectory_identity import gate_candidates,mutual_coverage


def candidate(y,score=.8,support=.8):
    t=np.linspace(0,100,61)
    return {'path_points_original_xy':np.column_stack([t,y(t) if callable(y) else t*0+y]).tolist(),
            'model_selection_score_UNCALIBRATED':score,
            'fraction_steps_above_local_signal_0_15':support}


def test_duplicate_centerlines_not_counted_as_two_tracks():
    a=candidate(0,.9);b=candidate(.2,.85)
    r=gate_candidates([a,b]);assert r['unique_path_count']==1
    assert r['rows'][1]['duplicate_of']==0
    assert r['decision']=='SINGLE_PROVISIONAL_REVIEW' and not r['physical_join_authorized']


def test_branching_fork_abstains_and_never_approves():
    a=candidate(0,.9);b=candidate(lambda t:.23*t,.88)
    r=gate_candidates([a,b]);assert r['unique_path_count']==2
    assert r['decision']=='AMBIGUOUS_REVIEW'
    assert r['expert_truth_added']==0 and not r['physical_join_authorized']


def test_weak_evidence_abstains():
    r=gate_candidates([candidate(0,.8,.01)])
    assert r['decision']=='LOW_EVIDENCE_REVIEW'


def test_bad_geometry_rejected():
    with pytest.raises(ValueError):mutual_coverage({'path_points_original_xy':[[0,0]]},candidate(0))
    with pytest.raises(ValueError):gate_candidates([candidate(0,support=-1)])
    with pytest.raises(ValueError):gate_candidates([candidate(0)],duplicate_coverage=1.2)
