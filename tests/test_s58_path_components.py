import numpy as np
import pytest
from bubble_chamber.s58_assemble_path_hypotheses import assemble_components


def segment(x0,x1,y,reverse=False):
    xs=np.linspace(x0,x1,31)
    p=np.column_stack((xs,np.full(31,y)))
    if reverse:p=p[::-1]
    return {'points_xy':p.tolist()}


def test_machine_path_preserves_visible_and_predicted_segments_in_order():
    nodes=[segment(20,45,90),segment(60,85,90,True),segment(100,125,90)]
    records=assemble_components(nodes,[[0,1],[1,2]])
    assert len(records)==1
    p=records[0]
    assert p['node_count']==3 and p['join_count']==2
    assert p['path_status']=='UNKNOWN_PHYSICAL_IDENTITY'
    assert p['physics_auto_joins_authorized'] is False
    assert [x['kind'].startswith('OBSERVED') for x in p['segments_PRIVATE']]==[True,False,True,False,True]
    assert [x['kind'].startswith('PREDICTED_GAP') for x in p['segments_PRIVATE']].count(True)==2
    assert [x['graph_node'] for x in p['segments_PRIVATE'] if 'graph_node' in x]==[0,1,2]
    assert p['segments_PRIVATE'][1]['points_xy_PRIVATE'][0][0] == pytest.approx(45,abs=.1)
    assert p['segments_PRIVATE'][1]['points_xy_PRIVATE'][-1][0] == pytest.approx(60,abs=.1)


def test_duplicate_ports_and_cycle_rejected():
    nodes=[segment(20,45,90),segment(60,85,90),segment(100,125,90)]
    with pytest.raises(ValueError):assemble_components(nodes,[[0,1],[0,1]])
    with pytest.raises(ValueError):assemble_components(nodes,[[0,1],[0,2],[1,2]])


def test_empty_result_preserves_unknown_and_no_join_claims():
    nodes=[segment(20,45,90),segment(60,85,90)]
    assert assemble_components(nodes,[])==[]
    with pytest.raises(ValueError):assemble_components(nodes,[[0,7]])
