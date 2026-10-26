"""Public data-free S37 contracts and calibration regression tests."""
import numpy as np
import pytest
from bubble_chamber.s37_risk_calibration import (
    cp_upper,review_selection,scene_stats,candidate_cutoffs,calibrate_review_risk,public_stats)

def fake_scenes(n=24):
    return [{'pairs':[{'left_index':i,'right_index':j,'label_SYNTHETIC_ONLY':int(i==j)}
                      for i in range(2) for j in range(3)]} for _ in range(n)]

def test_small_calibration_cannot_claim_two_percent_risk():
    assert cp_upper(0,24,.05)>.1
    assert cp_upper(0,320,.05/13)<.02
    assert cp_upper(0,320,.05/13)>0
    assert cp_upper(1,320,.05/13)>cp_upper(0,320,.05/13)


def test_abstain_when_calibration_too_small():
    scenes=fake_scenes(24)
    # Scores place both real synthetic pairs first, no false joins.
    s=np.tile([.97,.2,.1,.1,.96,.2],24)
    d=calibrate_review_risk(np.linspace(.01,.99,100),scenes,s,target_scene_false_risk=.02)
    assert d.all_abstain and d.threshold==float('inf') and d.scope.endswith('NOT_PHYSICAL_SAFETY')


def test_nonempty_calibration_possible_with_320_synthetic_scenes():
    scenes=fake_scenes(320)
    scores=np.tile([.97,.1,.15,.1,.96,.2],320)
    train=np.concatenate([np.linspace(.2,.8,30),np.linspace(.8,.95,70)])
    d=calibrate_review_risk(train,scenes,scores,target_scene_false_risk=.02)
    assert d.threshold<float('inf') and d.calibration_correct>0
    assert d.observed_false_scenes==0 and d.upper_scene_false_probability<.02


def test_risk_adaptation_uses_no_test_labels():
    scenes=fake_scenes(320)
    s=np.tile([.97,.98,.1,.1,.96,.20],320)
    train=np.linspace(.1,.97,200)
    d=calibrate_review_risk(train,scenes,s)
    # Adversarial false high-scoring link on EVERY scene must force abstain.
    assert d.all_abstain
    assert scene_stats(scenes,s,float('inf'))['accepted_suggestions']==0


def test_mutual_best_preserves_no_duplicate_and_margin():
    rows=fake_scenes(1)[0]['pairs']
    assert review_selection(rows,[.97,.18,.1,.20,.95,.2],.90)==((0,0),(1,1))
    assert review_selection(rows,[.97,.96,.1,.20,.95,.2],.90)==()
    with pytest.raises(ValueError): review_selection(rows,[1,2],.5)
    with pytest.raises(ValueError):review_selection(rows,[.5]*6,float('nan'))


def test_public_metrics_strips_private_per_scene_counts():
    s=fake_scenes(5)
    x=scene_stats(s,np.tile([.97,.12,.11,.12,.98,.1],5),.8)
    assert x['correct_links']==10 and x['wrong_links']==0
    assert 'per_scene_counts_private' not in public_stats(x)
    assert not public_stats(x).get('physics_auto_joins_authorized',False)


def test_source_threshold_order_is_predeclared():
    a=np.linspace(.01,.99,100)
    assert candidate_cutoffs(a)==candidate_cutoffs(a)
    with pytest.raises(ValueError):candidate_cutoffs([float('nan')]*100)
