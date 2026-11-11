import numpy as np
from bubble_chamber.s115_junction_graph import associate_observed_junctions, JunctionConfig


def _path(arr,i):
    return {'tracklet_id':str(i), 'observed_pixel_path_xy':arr}


def test_x_junction_keeps_two_observed_crossing_curves():
    sk=np.zeros((101,101),bool)
    sk[50,15:86]=True;sk[15:86,50]=True
    arms=[_path([[x,50] for x in range(15,51)],'west'),
          _path([[x,50] for x in range(50,86)],'east'),
          _path([[50,y] for y in range(15,51)],'north'),
          _path([[50,y] for y in range(50,86)],'south')]
    r=associate_observed_junctions(arms,sk)
    assert r['report']['accepted_source_junction_joins']==2,r['report']
    groups={frozenset(c['source_tracklet_ids']) for c in r['curves']}
    assert groups=={frozenset(('west','east')),frozenset(('north','south'))}
    assert all(c['particle_identity'] is None for c in r['curves'])


def test_branch_alone_does_not_follow_right_angle():
    sk=np.zeros((101,101),bool)
    sk[50,15:86]=True;sk[50:86,50]=True
    arms=[_path([[x,50] for x in range(15,51)],'west'),
          _path([[x,50] for x in range(50,86)],'east'),
          _path([[50,y] for y in range(50,86)],'south')]
    r=associate_observed_junctions(arms,sk)
    assert len(r['joins'])==1
    assert {r['joins'][0]['a_tracklet_id'],r['joins'][0]['b_tracklet_id']}=={'west','east'}


def test_source_only_no_observed_shortcut_over_missing_core():
    sk=np.zeros((101,101),bool);sk[50,15:48]=True;sk[50,53:86]=True
    arms=[_path([[x,50] for x in range(15,48)],'west'),
          _path([[x,50] for x in range(53,86)],'east')]
    r=associate_observed_junctions(arms,sk)
    assert r['report']['accepted_source_junction_joins']==0
    assert len(r['curves'])==2


def test_input_ids_must_be_unique():
    import pytest
    with pytest.raises(ValueError,match='Duplicate'):
        associate_observed_junctions([_path([[1,1],[2,1]],'x'),_path([[3,1],[4,1]],'x')],np.ones((6,6),bool))


def test_ambiguous_two_straight_continuations_abstain():
    sk=np.zeros((101,101),bool)
    sk[50,15:86]=True
    a=_path([[x,50] for x in range(15,51)],'west')
    b=_path([[x,50] for x in range(50,86)],'east_a')
    c=_path([[x,50] for x in range(50,86)],'east_b')
    r=associate_observed_junctions([a,b,c],sk)
    assert r['report']['accepted_source_junction_joins']==0
    assert r['report']['abstained_ambiguous_junction_clusters']==1
    assert len(r['curves'])==3


def test_source_only_disallows_parallel_neighbor_shortcut():
    sk=np.zeros((101,101),bool)
    sk[48,15:86]=True;sk[52,15:86]=True
    a=_path([[x,48] for x in range(15,51)],'a')
    b=_path([[x,52] for x in range(49,86)],'b')
    result=associate_observed_junctions([a,b],sk)
    assert result['report']['accepted_source_junction_joins']==0
    assert all(c['particle_identity'] is None for c in result['curves'])


def test_no_cycles_and_one_port_one_join():
    sk=np.zeros((81,81),bool)
    sk[40,10:71]=True
    a=_path([[x,40] for x in range(10,41)],'A')
    b=_path([[x,40] for x in range(40,71)],'B')
    r=associate_observed_junctions([a,b],sk)
    used=set()
    for e in r['joins']:
        for node in ((e['i'],e['a_end']),(e['j'],e['b_end'])):
            assert node not in used
            used.add(node)
    assert len(r['joins'])==1


def test_internal_fragment_is_not_lost_when_it_has_two_connections():
    """Regression: a middle fragment sorted first must not break a 3-arm chain."""
    sk=np.zeros((96,140),bool);sk[50,10:131]=True
    center=_path([[x,50] for x in range(50,91)],'center')
    left=_path([[x,50] for x in range(10,51)],'left')
    right=_path([[x,50] for x in range(90,131)],'right')
    r=associate_observed_junctions([center,left,right],sk)
    assert r['report']['accepted_source_junction_joins']==2
    assert r['report']['output_curve_count']==1
    assert len(r['curves'][0]['source_tracklet_ids'])==3
    assert len(r['curves'][0]['junction_join_ids'])==2
    assert set(r['curves'][0]['source_tracklet_ids'])=={'left','center','right'}
