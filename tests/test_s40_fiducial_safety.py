"""No manual GT required; ensure known f-marks cannot be silently called tracks."""
from __future__ import annotations
import numpy as np
import pytest
from bubble_chamber.s40_fiducial_safety import registered_fiducials,fiducial_guard,guard_proposals,guard_graph


def mock():
    return {'H50Z_030_01_005.bmp': {'original_width':1024,'original_height':1024,
        'keypoints':[{'label':'f1','x_px':212,'y_px':280,'width_px':46},
                     {'label':'start','x_px':222,'y_px':280,'width_px':46}]}}


def test_start_is_not_negative_fiducial():
    z=registered_fiducials(mock(),'005')
    assert [x['label'] for x in z]==['f1']


def test_nearest_point_on_segment_not_just_polyline_endpoints():
    z=registered_fiducials(mock(),'005')
    r=fiducial_guard([[20,180],[420,180]],(0,100),'005',z)
    assert r['fiducial_mark_zone_overlap'] and r['nearby_fiducial_labels']==['f1']


def test_origin_and_photo_are_provenance_protected():
    z=registered_fiducials(mock(),'005')
    assert not fiducial_guard([[20,180],[420,180]],(0,100),'006',z)['fiducial_mark_zone_overlap']
    assert not fiducial_guard([[20,180],[420,180]],(0,700),'005',z)['fiducial_mark_zone_overlap']


def test_far_curve_never_hidden():
    z=registered_fiducials(mock(),'005')
    assert not fiducial_guard([[600,600],[700,601]],(0,0),'005',z)['fiducial_mark_zone_overlap']


def test_invalid_fiducials_fail_closed():
    x=mock();x['H50Z_030_01_005.bmp']['keypoints'][0]['x_px']=99999
    with pytest.raises(ValueError,match='out of bounds'):registered_fiducials(x,'005')
    with pytest.raises(ValueError,match='photo'):registered_fiducials(mock(),'007')


def test_duplicates_fail_closed():
    d=mock();d['H50Z_030_01_005.bmp']['keypoints'].append(d['H50Z_030_01_005.bmp']['keypoints'][0].copy())
    with pytest.raises(ValueError,match='Duplicate'):registered_fiducials(d,'005')


def test_proposal_flags_do_not_mutate_machine_coordinates():
    z=registered_fiducials(mock(),'005')
    p=[{'draft_id':'d1','points_xy':[[20,180],[420,180]]}]
    result=guard_proposals(p,(0,100),'005',z)
    assert result[0]['fiducial_mark_zone_overlap']
    assert p==[{'draft_id':'d1','points_xy':[[20,180],[420,180]]}]


def test_graph_never_automerge_even_unsupported_schema():
    z=registered_fiducials(mock(),'005')
    g={'nodes':[{'draft_id':'d1','points_xy':[[20,180],[420,180]]}],
       'relations':[{'totally_unknown_names':['d1','d2']}]}
    r=guard_graph(g,(0,100),'005',z)
    assert r['candidate_edges'][0]['status']=='UNKNOWN_SCHEMA_ABSTAIN'
    assert r['fiducial_safe_auto_joins']==0


def test_detect_two_calibration_cross_axes_on_artificial_X():
    import cv2
    from bubble_chamber.s40_fiducial_safety import infer_fiducial_cross_arms
    gray=np.full((512,512),210,dtype=np.uint8)
    cv2.line(gray,(50,50),(460,460),35,3)
    cv2.line(gray,(50,460),(460,50),35,3)
    angles=infer_fiducial_cross_arms(gray,(255,255))
    assert len(angles)==2
    assert all(0<=a<np.pi for a in angles)
    assert abs(abs(angles[1]-angles[0])-np.pi/2)<.15


def test_arm_lookalike_flags_long_X_but_not_arbitrary_line():
    z=registered_fiducials(mock(),'005')
    z[0]['arms_angles_rad_PRIVATE']=[np.pi/4,3*np.pi/4]
    z[0]['arm_half_span_px']=220
    z[0]['arm_tolerance_px']=9
    a=fiducial_guard([[75,143],[350,418]],(0,0),'005',z)
    assert a['fiducial_arm_lookalike'] and a['fiducial_uncertainty_any']
    b=fiducial_guard([[75,400],[350,410]],(0,0),'005',z)
    assert not b['fiducial_arm_lookalike']


def test_graph_flags_dangerous_endpoint_and_bridge():
    z=registered_fiducials(mock(),'005')
    nodes=[{'draft_id':'near','points_xy':[[200,270],[202,274]]},
           {'draft_id':'far','points_xy':[[400,400],[420,420]]}]
    g={'nodes':nodes,'candidate_edges':[{'i':0,'j':1,'hermite_bridge_pixels_PROPOSAL_ONLY':[[250,300],[400,400]]}],
       'unambiguous_ranked_edges_UNVERIFIED':[{'i':0,'j':1}]}
    r=guard_graph(g,(0,0),'005',z)
    assert r['fiducial_risk_edges']==1
    assert r['fiducial_risk_among_previously_ranked_edges']==1
    assert r['candidate_edges'][0]['status']=='FIDUCIAL_RISK_ABSTAIN'
    assert r['fiducial_safe_auto_joins']==0


def test_bridge_alone_can_create_risk():
    z=registered_fiducials(mock(),'005')
    nodes=[{'draft_id':'nearleft','points_xy':[[100,280],[110,280]]},
           {'draft_id':'nearright','points_xy':[[320,280],[350,280]]}]
    r=guard_graph({'nodes':nodes,'candidate_edges':[{'i':0,'j':1,'hermite_bridge_pixels_PROPOSAL_ONLY':[[110,280],[310,280]]}]},(0,0),'005',z)
    assert r['candidate_edges'][0]['bridge_fiducial_risk']
    assert r['fiducial_risk_edges']==1
