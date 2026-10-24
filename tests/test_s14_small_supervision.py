from dataclasses import replace
import numpy as np
import pytest
from bubble_chamber.s14_small_supervision import (
    LabelBudgetProtocol, _initial_indices, propose_uncertain_reviews, propose_balanced_reviews,
    _frames_split, _row_id, _candidate_digest,
)


def rows(n=60):
    return [{'frame_id':f'070_{i//4:03d}','film_id':'070','view':(i%3)+1,
             'scene_id':f'070_{i//4:03d}:v{i%3}', 'cut_index':0,
             'left_id':f'e{i//2}', 'right_id':f'e{i}',
             'truth_same_track':i%3==0,
             'geometry_generic':[float(i%7),float(i/8)],
             'geometry_axial_fuzzy':[float(i%7),float(i/8),float(i%3)]} for i in range(n)]


def test_uncertainty_selector_no_truth_leak_and_spread():
    x=rows(80);prob=np.asarray([.5 if i<30 else .999 for i in range(len(x))])
    chosen=propose_uncertain_reviews(x,prob,{0,1},10,max_per_frame=2)
    assert 0 not in chosen and 1 not in chosen and len(set(chosen))==10
    assert max(sum(x[i]['frame_id']==f for i in chosen) for f in {r['frame_id'] for r in x})<=2
    y=[dict(r,truth_same_track=not r['truth_same_track']) for r in x]
    assert chosen==propose_uncertain_reviews(y,prob,{0,1},10,max_per_frame=2)


def test_identity_does_not_include_labels():
    x=rows(3);copy=[dict(r,truth_same_track=not r['truth_same_track']) for r in x]
    assert _candidate_digest(x)==_candidate_digest(copy)
    assert _row_id(x[1])==_row_id(copy[1])
    assert _candidate_digest(x)!=_candidate_digest(x[:2])


def test_proposal_input_rejects_invalid_scores():
    x=rows(2)
    for prob in [np.array([.5]), np.array([.2,np.nan]),np.array([-.1,.2])]:
        with pytest.raises(ValueError):propose_uncertain_reviews(x,prob,set(),1)
    with pytest.raises(ValueError):propose_uncertain_reviews(x,np.array([.3,.9]),set(),0)


def test_split_rejects_overlaps_and_missing():
    x=rows(16);frames=sorted({r['frame_id'] for r in x})
    with pytest.raises(ValueError,match='overlap'):
        _frames_split(x,frames,{'train':frames[:2],'val':frames[1:3],'test':frames[3:]})
    with pytest.raises(ValueError,match='incomplete'):
        _frames_split(x,frames,{'train':frames[:1],'val':frames[1:2],'test':frames[2:3]})


def test_initial_review_seed_no_label_oracle():
    x=rows(60);a=_initial_indices(x,20,42)
    assert a==_initial_indices(x,20,42)
    changed=[dict(r,truth_same_track=not r['truth_same_track']) for r in x]
    assert a==_initial_indices(changed,20,42)


def test_budget_validation():
    LabelBudgetProtocol().validate()
    with pytest.raises(ValueError):replace(LabelBudgetProtocol(),budgets=(20,20,80)).validate()


def test_balanced_selector_no_oracle_and_frame_diversity():
    x=rows(100)
    p=np.linspace(.01,.99,100)
    chosen=propose_balanced_reviews(x,p,set(range(4)),12,max_per_frame=2)
    assert len(chosen)==12 and len(set(chosen))==12 and not (set(chosen)&set(range(4)))
    assert max(sum(x[i]['frame_id']==f for i in chosen) for f in {r['frame_id'] for r in x})<=2
    mutated=[dict(r,truth_same_track=not r['truth_same_track']) for r in x]
    assert chosen==propose_balanced_reviews(mutated,p,set(range(4)),12,max_per_frame=2)
    with pytest.raises(ValueError):propose_balanced_reviews(x,p,set(),101)
