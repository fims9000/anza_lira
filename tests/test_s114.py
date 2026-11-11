import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pytest
from bubble_chamber.s114_source_geometry import Config,reconstruct,proposed_bridges,render
from bubble_chamber.s114_supplement import select_uncovered_tracklets


def fixture_image(seed=1):
    rng=np.random.default_rng(seed)
    g=(195+rng.normal(0,2.5,(288,320))).clip(0,255).astype(np.uint8)
    for y in range(20,269,4):
        x=int(110+ .00085*(y-140)**2)
        cv2.circle(g,(x,y),2,25,-1)
    cv2.line(g,(216,15),(235,265),26,3)
    cv2.line(g,(25,260),(68,215),10,3)
    cv2.line(g,(25,215),(68,260),10,3)
    return g


def test_reconstruction_source_first_and_path_endpoints():
    gray=fixture_image()
    protected=np.zeros_like(gray,np.uint8)
    cv2.circle(protected,(47,237),28,1,-1)
    result=reconstruct(gray,protected)
    assert result['report']['major_candidate_pieces']>=1
    # Most of the recognized main tracklet pixels must stay away from the known cross.
    for tr in result['tracks']:
        pts=np.asarray(tr['observed_pixel_path_xy'],float)
        assert all(protected[int(p[1]),int(p[0])]==0 for p in pts)
        assert tr['physical_particle_id'] is None
        assert tr['A_B_are_geometric_not_physical']
        assert len(tr['start_A_xy'])==2 and len(tr['end_B_xy'])==2
    overlay=render(gray,result['tracks'],operator_points=[{'label':'f4','xy_local':[47,237]}])
    assert overlay.shape == (*gray.shape,3)


def test_negative_uniform_gray_is_not_a_line():
    gray=np.full((220,220),196,np.uint8)
    res=reconstruct(gray)
    assert res['report']['candidate_piece_count']==0
    assert res['report']['skeleton_pixel_count']==0


def test_non8bit_or_invalid_cross_mask_abstains():
    with pytest.raises(ValueError):reconstruct(np.ones((100,100),np.float32))
    with pytest.raises(ValueError):reconstruct(np.zeros((100,100),np.uint8),np.zeros((20,20),np.uint8))


def test_supplement_does_not_duplicate_s111_source_curves():
    tr={'tracklet_id':'a','observed_arc_length_px':110.0,'mean_directional_response':22.0,
        'observed_pixel_path_xy':[[float(i),60.] for i in range(20,131)],
        'spline_xy':[[20.,60.],[130.,60.]],'is_major_visual_candidate':True,
        'start_A_xy':[20.,60.],'end_B_xy':[130.,60.]}
    old={'source_anchored':True,'sampled_xy':[[float(i),60.] for i in range(20,131)]}
    assert select_uncovered_tracklets([tr],[old])==[]
    tr2=dict(tr,tracklet_id='b',observed_pixel_path_xy=[[float(i),110.] for i in range(20,131)])
    found=select_uncovered_tracklets([tr,tr2],[old])
    assert len(found)==1 and found[0]['tracklet_id']=='b'
    assert found[0]['status']=='SOURCE_FIRST_NEW_TRACKLET_REVIEW_ONLY'


def test_review_bridges_cannot_be_observed():
    res=reconstruct(fixture_image(seed=1))
    gaps=proposed_bridges(res['tracks'],res['response'])
    assert all(g['status']=='REVIEW_ONLY_NOT_OBSERVED' and g['is_observed'] is False for g in gaps)
