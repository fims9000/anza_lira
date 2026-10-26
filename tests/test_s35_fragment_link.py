"""Data-free geometry, safe rejection and synthetic source-lock contracts for S35."""
import copy
from hashlib import sha256
from pathlib import Path
import json
import cv2
import numpy as np
import pytest
from bubble_chamber.s35_fragment_link import (
    GEOMETRY_FIELDS,PHOTO_FIELDS,ANZA_FIELDS,optimal_endpoints,
    pair_features,prepare_evidence,calibrated_negative_threshold,
    select_review_links,feature_matrix)
from bubble_chamber.s35_simulated_gaps import frame_split,extract_track_templates,make_scene


def line(y=95,n=80,lo=15,hi=85):
    return np.stack((np.linspace(lo,hi,n), np.linspace(y,y+0.7,n)),axis=1)


def test_nearest_endpoints_unchanged_by_path_reversal():
    a=line(lo=20,hi=81)
    b=line(lo=99,hi=166)
    x=optimal_endpoints(a,b)
    y=optimal_endpoints(a[::-1],b[::-1])
    assert np.isclose(x['gap'],y['gap'])
    assert np.allclose(x['endpoint_a'],y['endpoint_a'])
    assert np.allclose(x['endpoint_b'],y['endpoint_b'])
    assert x['gap']>16


def test_source_photo_features_do_not_contain_truth_or_auto_join():
    gray=np.full((192,192),190,dtype=np.uint8)
    cv2.line(gray,(20,95),(84,95),80,2)
    cv2.line(gray,(102,95),(165,95),80,2)
    a=line(lo=20,hi=84);b=line(lo=102,hi=165)
    image=prepare_evidence(gray,anza_map=np.full(gray.shape,.5,np.float32))
    f=pair_features(a,b,image)
    for k in GEOMETRY_FIELDS+PHOTO_FIELDS+ANZA_FIELDS:
        assert np.isfinite(f[k]),k
    assert f['source_evidence_status']=='UNVERIFIED_MODEL_AND_PHOTOMETRY'
    assert not any('truth' in k.lower() or 'physics' in k.lower() for k in f)
    assert f['endpoint_gap_px']>16
    with pytest.raises(ValueError):prepare_evidence(gray,anza_map=np.full(gray.shape,.9,np.float32)[1:])


def test_reject_foreign_and_corrupt_polyline_without_silent_clipping():
    gray=np.full((192,192),200,np.uint8)
    e=prepare_evidence(gray)
    a=line();b=line(lo=95,hi=172)
    with pytest.raises(ValueError):pair_features(a*float('nan'),b,e)
    with pytest.raises(ValueError):pair_features(a,b+900,e)
    with pytest.raises(ValueError):feature_matrix([{'endpoint_gap_px':2.}],'not a tuple')


def test_empirical_calibration_bound_on_calibration_only():
    scores=np.array([.85,.70,.6,.51,.49,.2,.1,.04,.01,.0, .66,.32],float)
    labels=np.array([0]*10+[1,1],int)
    t=calibrated_negative_threshold(scores,labels,max_fpr=0.0)
    assert np.sum(scores[:10]>=t)==0
    t=calibrated_negative_threshold(scores,labels,max_fpr=.2)
    assert np.sum(scores[:10]>=t)<=2
    with pytest.raises(ValueError):calibrated_negative_threshold(scores,np.zeros(len(scores)))
    with pytest.raises(ValueError):calibrated_negative_threshold(scores,labels,max_fpr=.99)


def test_conservative_pair_selection_abstains_and_never_approves():
    rows=[{'left_index':i,'right_index':j} for i in range(2) for j in range(3)]
    uncertain=select_review_links(rows,[.91,.90,.20,.60,.59,.30],threshold=.50,margin=.04)
    assert uncertain['suggested_pairs']==[]
    assert uncertain['human_verified_pairs_created']==0
    assert uncertain['physics_auto_joins_authorized'] is False
    certain=select_review_links(rows,[.97,.15,.12,.20,.96,.05],threshold=.90)
    assert set(certain['suggested_pairs'])=={(0,0),(1,1)}
    assert certain['physics_auto_joins_authorized'] is False
    with pytest.raises(ValueError):select_review_links(rows,[.5]*5,threshold=.5)
    with pytest.raises(ValueError):select_review_links(rows+rows[:1],[.5]*7,threshold=.5)


def test_synthetic_truth_only_for_injected_fragments(tmp_path):
    tile=tmp_path/'fake_005.png'
    image=np.full((512,512),200,dtype=np.uint8)
    assert cv2.imwrite(str(tile),image)
    u=np.linspace(0,1,15)
    sources=[{'points':np.column_stack((1000*u,100*u*u)),'frame':'070_001','id':'x'},
             {'points':np.column_stack((1000*u,-80*u*u)),'frame':'070_002','id':'y'}]
    a=make_scene(tile,sources,seed=123456,scenario='crossing')
    b=make_scene(tile,sources,seed=123456,scenario='crossing')
    assert np.array_equal(a.gray,b.gray)
    assert a.correct_pairs=={(0,0),(1,1)}
    assert len(a.left)==2 and len(a.right)==3
    assert a.simulated_measurement_error_px>0
    assert not np.array_equal(a.gray,image[:192,:192])
    assert a.extra_masks['injected_visible_strokes'].any()
    assert a.extra_masks['injected_false_decoy'].any()


def test_source_frames_split_disjoint_and_deterministic():
    fake={'frames':[{'frame_id':f'070_{i:03d}','tracks':[]} for i in range(100)]}
    a,b=frame_split(fake)
    assert len(a)==75 and len(b)==25
    assert not a&b and a|b=={x['frame_id'] for x in fake['frames']}
    assert frame_split(fake)==(a,b)


def test_no_human_authentic_track_claims_in_generator():
    import inspect
    from bubble_chamber import s35_simulated_gaps as s
    source=inspect.getsource(s)
    assert 'SyntheticFragmentScene' in source and 'correct_pairs' in source
    assert 'native H50Z photo is unlabeled' in source
