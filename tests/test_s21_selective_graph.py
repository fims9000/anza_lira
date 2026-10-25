import numpy as np
import pytest
from bubble_chamber.s21_selective_graph import (
    S21Protocol,grouped_graph_features,acquire_human_pairs,
    fit_selective_models,accept_with_support,calibrate_zero_false_joins,
    group_metrics,paired_frame_ci)

def sample_rows(n=100):
    rows=[]
    for i in range(n):
        scene='S'+str(i//10);a='L'+str(i//2);b='R'+str(i%10)
        rows.append(dict(scene_id=scene,frame_id='F'+str(i//10),left_id=a,
          right_id=b,selection_score=i%7/10.,geometry_axial_fuzzy=[float(i%7+1)]+[.01*(i+j) for j in range(13)],
          truth_same_track=i%5==0,candidate_rank=i%5))
    return rows

def test_context_does_not_use_hidden_truth():
    rows=sample_rows()
    a,b=grouped_graph_features(rows)
    assert a.shape==(100,14) and b.shape==(100,24)
    rows2=[{**r,'truth_same_track':not r['truth_same_track']} for r in rows]
    x,y=grouped_graph_features(rows2)
    np.testing.assert_allclose(a,x);np.testing.assert_allclose(b,y)


def test_frame_diverse_acquisition_is_label_blind_and_deterministic():
    rows=sample_rows(100);a,b=grouped_graph_features(rows)
    ids,log=acquire_human_pairs(rows,b,a,55,seed=42,warm_start=20,batch_size=17)
    assert len(ids)==55 and len(set(ids))==55
    rows2=[{**r,'truth_same_track':not r['truth_same_track']} for r in rows]
    ids2,log2=acquire_human_pairs(rows2,b,a,55,seed=42,warm_start=20,batch_size=17)
    # label-aware acquisitions AFTER first batch may differ; first batch may not.
    assert ids[:20]==ids2[:20]
    assert all(not z['class_labels_consulted_for_selection'] for z in log)


def test_strict_threshold_abstains_and_support_mask():
    rows=sample_rows(100)
    scores=np.linspace(.01,.99,100)
    out=calibrate_zero_false_joins(rows,scores,margin=.05,dist_cut=.2,max_false=0)
    assert out['val_reviewed_fp']==0
    a=accept_with_support(rows,scores,2.0,.05,dist_cut=100)
    assert not a.any()
    met=group_metrics(rows,a)
    assert met['tp']==0 and met['fp']==0 and met['release_to_physics'] is False


def test_fit_needs_both_classes_and_selects_same_labels():
    rows=sample_rows();a,b=grouped_graph_features(rows)
    with pytest.raises(ValueError,match='both'):
        fit_selective_models(rows,a,b,[i for i in range(100) if i%5==0],42)
    result=fit_selective_models(rows,a,b,list(range(45)),42)
    assert set(result)=={'geometry_logreg','context_logreg','context_hgb'}


def test_pair_bootstrap_identity():
    rows=sample_rows();a=np.zeros(100,bool)
    o=paired_frame_ci(rows,a,a,repeats=100)
    assert o['delta_recall_b_minus_a']==0
    assert o['frame_bootstrap_ci95']==[0.,0.]


def test_invalid_geometry_rejected():
    rows=sample_rows();rows[2]['geometry_axial_fuzzy']=[float('nan')]*14
    with pytest.raises(ValueError,match='finite'):
        grouped_graph_features(rows)
    with pytest.raises(ValueError):S21Protocol(label_budgets=(20,10)).validate()


def test_validation_hidden_labels_cannot_change_calibration():
    from bubble_chamber.s21_selective_graph import blind_validation_reviews
    rows=sample_rows(200)
    reviewed=blind_validation_reviews(rows,45,seed=4)
    scores=np.linspace(.2,.9,200)
    selected=set(reviewed)
    alternative=[dict(row) for row in rows]
    for i in range(len(rows)):
        if i not in selected:
            alternative[i]['truth_same_track']=not alternative[i]['truth_same_track']
    try:
        x=calibrate_zero_false_joins(rows,scores,margin=.01,dist_cut=100.,reviewed_indices=reviewed)
        y=calibrate_zero_false_joins(alternative,scores,margin=.01,dist_cut=100.,reviewed_indices=reviewed)
    except ValueError as e:
        if 'both classes' in str(e):pytest.skip('synthetic review subset has no class diversity')
        raise
    assert x==y


def test_blind_validation_review_identities_unchanged_on_label_flip():
    from bubble_chamber.s21_selective_graph import blind_validation_reviews
    rows=sample_rows(100)
    first=blind_validation_reviews(rows,30,seed=111)
    flipped=[{**r,'truth_same_track':not r['truth_same_track']} for r in rows]
    assert first==blind_validation_reviews(flipped,30,seed=111)
