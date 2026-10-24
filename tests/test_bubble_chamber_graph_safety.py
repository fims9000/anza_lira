import numpy as np
import pytest
from bubble_chamber.graph_safety import GraphSafetyProtocol,select_global_links,count_links,run_graph_safety_study


def _r(k,a,sa,b,sb,truth=False,scene='s0'):
    return {'scene_id':scene,'pair':[a,sa,b,sb],'truth_same_track':truth}


def test_ambiguous_endpoint_defers_two_competing_edges():
    rows=[_r(0,0,'B',1,'A',True),_r(1,0,'B',2,'A',False)]
    accepted,reasons=select_global_links(rows,np.array([.95,.93]),threshold=.9,margin=.05)
    assert not accepted.any()
    assert 'ambiguous_margin' in reasons


def test_unique_mutual_endpoint_and_chain_are_accepted():
    rows=[_r(0,0,'B',1,'A',True),_r(1,1,'B',2,'A',True)]
    accepted,_=select_global_links(rows,np.array([.96,.92]),threshold=.9,margin=.05)
    assert accepted.tolist()==[True,True]
    metrics=count_links(rows,accepted)
    assert metrics['tp']==2 and metrics['fp']==0


def test_candidate_endpoint_cannot_be_reused():
    rows=[_r(0,0,'B',1,'A'),_r(1,0,'B',2,'A')]
    accepted,reasons=select_global_links(rows,np.array([.99,.80]),threshold=.7,margin=.05)
    assert accepted.sum()==1
    assert not accepted[1]


def test_graph_cycle_is_rejected():
    rows=[_r(0,0,'B',1,'A',True),_r(1,1,'B',2,'A',True),_r(2,2,'B',0,'A',True)]
    accepted,reasons=select_global_links(rows,np.array([.98,.95,.90]),threshold=.8,margin=.02)
    assert accepted.sum()==2
    assert reasons[2]=='cycle'


def test_score_calibration_never_uses_real_film_labels():
    p=GraphSafetyProtocol(train_scenes=15,val_scenes=15,test_scenes=15)
    x=run_graph_safety_study(p)
    assert x['status']=='SYNTHETIC_PAIR_NONE_GRAPH_SAFETY_ONLY'
    assert x['allow_real_auto_linking'] is False
    assert x['selection_status'].startswith('VAL_GATE_')
    assert x['test']['graph_constrained']['real_automatic_merges']==0
    assert x['test']['graph_constrained']['accepted_edges'] <= x['test']['local_probability_only']['accepted_edges']
