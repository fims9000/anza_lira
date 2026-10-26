"""2D crossing is an uncertainty, never a false-particle label."""
from __future__ import annotations
import numpy as np
import pytest
from bubble_chamber.s51_topology_collision import (
    collision_advisory,central_bridge_points,min_point_to_segments_distance,
    full_graph_diagnostic)


def straight(x1,y1,x2,y2,n=15):
    return np.stack((np.linspace(x1,x2,n),np.linspace(y1,y2,n)),axis=1)


def graph(obstacle=True):
    nodes=[{'points_xy':straight(10,50,40,50).tolist()},
           {'points_xy':straight(70,50,110,50).tolist()}]
    if obstacle: nodes.append({'points_xy':straight(55,25,55,75).tolist()})
    else:nodes.append({'points_xy':straight(55,120,60,120).tolist()})
    bridge=straight(40,50,70,50,21).tolist()
    return {'nodes':nodes,'candidate_edges':[{'i':0,'j':1,
            'hermite_bridge_pixels_PROPOSAL_ONLY':bridge,
            'physics_join_authorized':False}], 'physics_auto_links_performed':0}


def test_exact_middle_collision_and_noncollision():
    a=graph(True);b=graph(False)
    x=collision_advisory(a['nodes'],a['candidate_edges'][0])
    y=collision_advisory(b['nodes'],b['candidate_edges'][0])
    assert x['potential_third_line_overlap'] is True
    assert x['native_particle_relation_truth']=='UNKNOWN'
    assert x['physical_auto_join_authorized'] is False
    assert not y['potential_third_line_overlap']
    assert full_graph_diagnostic(a)['collision_warning_edges']==1


def test_exact_segment_distance_detects_between_sparse_clicks():
    points=np.array([[55.,50.],[55.,51.]])
    line=np.array([[30.,50.],[80.,50.]])
    assert min_point_to_segments_distance(points,line)==0.
    assert len(central_bridge_points(straight(20,50,70,50,21)))==61


def test_reversing_bridge_not_change_warning():
    a=graph(True);z=a['candidate_edges'][0]
    result=collision_advisory(a['nodes'],z)
    z['hermite_bridge_pixels_PROPOSAL_ONLY']=z['hermite_bridge_pixels_PROPOSAL_ONLY'][::-1]
    assert result['potential_third_line_overlap']==collision_advisory(a['nodes'],z)['potential_third_line_overlap']


def test_fail_closed_for_broken_native_or_unauthorized():
    a=graph(True);a['candidate_edges'][0]['physics_join_authorized']=True
    with pytest.raises(ValueError):full_graph_diagnostic(a)
    a=graph(True);a['nodes'][2]['points_xy']=[[-1.,10.],[15.,10.]]
    with pytest.raises(ValueError):full_graph_diagnostic(a)
    a=graph(True);a['candidate_edges'][0]['i']=0;a['candidate_edges'][0]['j']=0
    with pytest.raises(ValueError):full_graph_diagnostic(a)
