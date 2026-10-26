"""S55 offline physics demo never promotes simulator scores to true physics."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import pytest
from scripts.research.bubble_chamber.s55_build_private_demo import combine
from scripts.research.bubble_chamber.s55_audit_native_support import inference
from scripts.research.bubble_chamber.s55_native_shape_simulator import SCHEMA


def fake_public_private_graph():
    tiles=[];new=[];serial=0
    for i in range(39):
        count=567//39 + (i < 567%39)
        edges=182//39 + (i < 182%39)
        tiles.append({'crop_id':f'p{i}','photo':'005',
          'candidates':[{'draft_id':f'c{i}.{j}','points':[[1,2],[3,4]],'status':'NONE'}
                        for j in range(count)],
          'graph_edges':[{'i':0,'j':1,'native_relation_truth':'UNKNOWN',
                          'physical_auto_join_authorized':False,'risky':True} for _ in range(edges)]})
        new.append({'crop_id':f'p{i}','photo':'005',
          'candidate_advisories':[{'draft_id':f'c{i}.{j}',
             'status':'UNKNOWN_UNSCORABLE_SOURCE_CONTEXT' if serial+j<141 else 'UNKNOWN_WITHIN_SIMULATOR_SUPPORT_NOT_PHYSICS_TRUTH',
             'native_track_identity':'UNKNOWN',
             'candidate_preserved':True,'physical_auto_join_authorized':False}
             for j in range(count)]})
        serial+=count
    return ({'native_GT_count':0,'physical_auto_links':0,
             'summary':{'machine_candidates':567,'graph_edges':182,'risky_edges':76},
             'tiles':tiles},
            {'native_expert_GT_count':0,'physical_auto_joins':0,'tiles':new})


def test_combine_is_non_mutating_and_never_grants_physics_gt():
    old,new=fake_public_private_graph(); before=deepcopy(old)
    merged=combine(old,new)
    assert old==before
    assert merged['native_GT_count']==0 and merged['physical_auto_links']==0
    assert merged['s55_support_counts']['UNKNOWN_UNSCORABLE_SOURCE_CONTEXT']==141
    assert sum(len(t['candidates']) for t in merged['tiles'])==567
    assert all(e['native_relation_truth']=='UNKNOWN' and not e['physical_auto_join_authorized']
               for t in merged['tiles'] for e in t['graph_edges'])


def test_private_builder_fails_if_native_physics_auto_join_is_injected():
    old,new=fake_public_private_graph()
    old['tiles'][0]['graph_edges'][0]['physical_auto_join_authorized']=True
    with pytest.raises(ValueError): combine(old,new)
    old,new=fake_public_private_graph()
    new['tiles'][0]['candidate_advisories'][0]['native_track_identity']='particle'
    with pytest.raises(ValueError): combine(old,new)


def test_s55_portable_inference_refuses_wrong_source_or_native_truth():
    model=dict(schema=SCHEMA,trained_on_source_photo='005',target_photo='006',
               native_true_track_labels=0,can_autojoin_native_tracks=False,
               feature_names=['x','y'],mean=[0.,0.],sd=[1.,1.],weights=[0.,0.],intercept=0.)
    score,z=inference(np.array([[1.,2.]]),model,source='005',target='006')
    assert score[0]==.5 and z[0]==2.
    model['can_autojoin_native_tracks']=True
    with pytest.raises(ValueError):inference(np.array([[1.,2.]]),model,source='005',target='006')
