import numpy as np
import pytest
from bubble_chamber.s24_consensus_review import (
    AuditConfig, conservative_consensus, validate_scene_matching,
    split_validation_frames, unlabeled_audit_indices, candidate_error_bound,
    audit_labeled_subset, route_provisional_disagreements, replay_review_answers,
    zero_error_review_sample_size,
)


def rows():
    result=[]
    for frame in range(12):
        for j in range(6):
            result.append({'scene_id':f'frame{frame}:scene','frame_id':f'frame{frame}',
                           'left_id':f'L{j//2}','right_id':f'R{j%3}',
                           'truth_same_track':(j==0 or j==4)})
    return result


def test_consensus_safe_and_validates_matching():
    rr=rows()[:6]
    g=np.array([True,False,False,False,True,False])
    h=np.array([True,False,False,False,False,True])
    assert conservative_consensus(rr,g,h).tolist()==[True,False,False,False,False,False]
    bad=np.array([True,True,False,False,False,False])
    with pytest.raises(ValueError,match='inconsistent'):
        conservative_consensus(rr,g,bad)
    with pytest.raises(ValueError,match='duplicate'):
        validate_scene_matching(rr+[rr[0]],np.zeros(7,bool))


def test_audit_partition_frame_disjoint_and_label_blind():
    rr=rows();a,b=split_validation_frames(rr,43)
    assert set(a).isdisjoint(b)
    assert {rr[i]['frame_id'] for i in a}.isdisjoint({rr[i]['frame_id'] for i in b})
    chosen=unlabeled_audit_indices(rr,20,43)
    flipped=[{**r,'truth_same_track':not r['truth_same_track']} for r in rr]
    assert chosen==unlabeled_audit_indices(flipped,20,43)
    assert np.array_equal(a,split_validation_frames(flipped,43)[0])


def test_exact_risk_bound_and_no_release_claim():
    assert candidate_error_bound(0,0) is None
    assert candidate_error_bound(2,2)==1.
    assert .018 < candidate_error_bound(0,160)< .019
    assert candidate_error_bound(0,80)> .03
    assert candidate_error_bound(0,300)< .01
    with pytest.raises(ValueError):candidate_error_bound(1,0)
    rr=rows()
    x=np.zeros(len(rr),bool);x[0]=True
    z=audit_labeled_subset(rr,x,list(range(10)))
    assert z['correct_accepted']==1
    assert z['physical_certification_possible'] is False


def test_disagreement_queue_never_reads_truth_and_review_is_confirmed_only():
    rr=rows()[:6]
    g=np.array([True,False,False,False,True,False])
    h=np.array([False,False,False,False,False,True])
    scores1=np.array([.9,.1,.1,.1,.8,.1])
    scores2=np.array([.1,.1,.1,.1,.1,.95])
    ordered=route_provisional_disagreements(rr,g,h,scores1,scores2,12)
    flipped=[{**r,'truth_same_track':not r['truth_same_track']} for r in rr]
    ordered2=route_provisional_disagreements(flipped,g,h,scores1,scores2,12)
    assert ordered==ordered2
    replay=replay_review_answers(rr,conservative_consensus(rr,g,h),ordered,(1,2,3,5))
    assert replay['5']['human_pairs_reviewed']==3
    assert replay['5']['manually_verified_correct_joins_added']<=2
    assert replay['5']['provisional_false_joins']==0
    assert replay['5']['automated_physical_merge_authorized'] is False


def test_bad_inputs_fail_closed():
    with pytest.raises(ValueError):AuditConfig(training_gaps=(1,5)).validate()
    with pytest.raises(ValueError):split_validation_frames(rows()[:8],42)
    with pytest.raises(ValueError):unlabeled_audit_indices(rows(),200,42)
    with pytest.raises(ValueError):route_provisional_disagreements(rows()[:6],
        np.zeros(6,bool),np.zeros(6,bool),np.array([float('nan')]*6),np.zeros(6),12)
    with pytest.raises(ValueError):replay_review_answers(rows()[:6],np.zeros(6,bool),[0,0],(2,))


def test_claim_limit_requires_many_independent_accepted_reviews():
    assert zero_error_review_sample_size(.01)==299
    assert zero_error_review_sample_size(.005)==598
    with pytest.raises(ValueError):zero_error_review_sample_size(0.0)
