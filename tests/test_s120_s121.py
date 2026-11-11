import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from s120_features import photo_features,FEATURE_NAMES
from s120_reviewer import group_scores
from bubble_chamber.s121_fullframe_continuation import associate_fullframe,ContinuationConfig


def _example(gap_y=80,signal=True):
    image=np.full((200,240),215,np.uint8)
    xx1=np.arange(20,80,dtype=np.float64)
    xx2=np.arange(110,190,dtype=np.float64)
    pts1=np.column_stack((xx1,np.full(len(xx1),80.)))
    pts2=np.column_stack((xx2,np.full(len(xx2),float(gap_y))))
    cv2.line(image,(20,80),(79,80),70,1)
    cv2.line(image,(110,gap_y),(189,gap_y),70,1)
    if signal and gap_y==80:
        # Real, faint source response through the gap (not inferred from GT).
        cv2.line(image,(80,80),(109,80),120,1)
    curves=[{'candidate_id':'a','observed_pixel_path_xy':pts1.tolist()},
            {'candidate_id':'b','observed_pixel_path_xy':pts2.tolist()}]
    tile=[{'tile_id':'T00000','xyxy':[0,0,240,200],'curves':curves}]
    groups=[{'global_group_id':'ga','source_curve_keys':['T00000:a']},
            {'global_group_id':'gb','source_curve_keys':['T00000:b']}]
    return image,tile,groups


def test_photo_features_stable_when_path_reversed():
    image,tile,_=_example()
    x=np.asarray(tile[0]['curves'][0]['observed_pixel_path_xy'])
    a=photo_features(image,x)
    b=photo_features(image,x[::-1])
    assert len(a)==len(FEATURE_NAMES)==len(b)
    assert np.all(np.isfinite(a))
    np.testing.assert_allclose(a,b,atol=.15)


def test_photo_features_detects_source_line_evidence():
    image,tile,_=_example()
    real=photo_features(image,tile[0]['curves'][0]['observed_pixel_path_xy'])
    plain=photo_features(np.full_like(image,215),tile[0]['curves'][0]['observed_pixel_path_xy'])
    idx=FEATURE_NAMES.index('mean_flank_contrast_8')
    assert real[idx]>plain[idx]+20


def test_s120_group_score_preserves_source_and_id():
    gs=[{'global_group_id':'same','source_curve_keys':['a','b']},
        {'global_group_id':'low','source_curve_keys':['c']}]
    r=group_scores(gs,{'a':.95,'b':.6,'c':.1},{'a':100,'b':80,'c':10})
    assert r['same']['decision']=='AUTO_TRACK_LIKE_NOT_GT'
    assert r['low']['decision']=='DEPRIORITIZED_NOT_DELETED'
    assert [x['global_group_id'] for x in gs]==['same','low']


def test_s121_source_photo_supports_clean_gap_only():
    image,tile,groups=_example()
    r=associate_fullframe(tile,{},groups,image)
    assert r['report']['accepted_provisional_links']==1
    assert r['report']['provisional_s121_groups']==1
    e=next(q for q in r['proposals'] if q['decision'].startswith('AUTO_'))
    assert e['source_observed_bridge_pixels']==0
    assert e['inferred_geometry_only']


def test_s121_blank_gap_never_auto_links():
    image,tile,groups=_example(signal=False)
    r=associate_fullframe(tile,{},groups,image)
    assert r['report']['accepted_provisional_links']==0
    assert r['report']['provisional_s121_groups']==2


def test_s121_parallel_jump_rejected():
    image,tile,groups=_example(gap_y=86)
    r=associate_fullframe(tile,{},groups,image)
    assert r['report']['accepted_provisional_links']==0
    assert r['report']['provisional_s121_groups']==2


def test_s121_does_not_bridge_inside_same_existing_group():
    image,tile,groups=_example()
    groups=[{'global_group_id':'ga','source_curve_keys':['T00000:a','T00000:b']}]
    r=associate_fullframe(tile,{},groups,image)
    assert len(r['groups'])==1
    assert r['report']['accepted_provisional_links']==0


def test_s121_rejects_missing_original_gray_and_wrong_source_geometry():
    image,tile,groups=_example()
    with pytest.raises(ValueError,match='grayscale'):
        associate_fullframe(tile,{},groups,image.astype(np.float32))
    tile[0]['curves'][0]['observed_pixel_path_xy'][0]=[-5.,80.]
    with pytest.raises(ValueError,match='outside'):
        associate_fullframe(tile,{},groups,image)
