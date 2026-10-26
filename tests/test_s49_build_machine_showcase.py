"""Private offline viewer keeps hypotheses machine-only with provenance locks."""
import pytest
from scripts.research.bubble_chamber.s49_build_machine_showcase import (
    build_payload, TEMPLATE)


def make_payload():
    tiles=[]; details=[];graphs=[]
    for i in range(39):
        crop=f'tile_{i}'
        tile={'crop_id':crop,'photo_id':'005' if i<21 else '006',
              'proposals':[{'draft_id':f'c{i}','points_xy':[[1,1],[2,2]],
                            'source_method':'geometry'}]}
        d={'crop_id':crop,'candidate_evidence_PRIVATE':[
            {'draft_id':f'c{i}','registration_status':'POSE_ONLY_POSSIBLE_REPEAT_AMBIGUOUS'}],
           'graph_evidence_PRIVATE':[{'i':0,'j':1,'union_advisory':True,
                                     'physical_auto_join_authorized':False}]}
        g={'crop_id':crop,'candidate_edges':[{'i':0,'j':1,'physics_join_authorized':False,
                 'hermite_bridge_pixels_PROPOSAL_ONLY':[[1,1],[2,2]],
                 'geometry_rank_UNCALIBRATED':.8}]}
        tiles.append(tile);details.append(d);graphs.append(g)
    original={'native_expert_gt_created':0,'tiles':tiles}
    private={'machine_crops_PRIVATE':details}
    source_graph={'graphs':graphs}
    agg={'S48_native_machine':{'machine_candidate_counts':{'all_original_machine_candidates_preserved':39},
                                'graph_counts':{'edge_count':39,'union_any':39}}}
    return original,private,source_graph,agg


def test_all_private_machine_groups_preserved_no_GT():
    args=make_payload()
    r=build_payload(*args)
    assert len(r['tiles'])==39 and r['summary']['risky_edges']==39
    assert r['physical_auto_links']==0 and r['native_GT_count']==0
    assert not r['allowed_to_use_as_heldout_photo_training_GT']
    assert r['tiles'][0]['graph_edges'][0]['risky']


def test_public_html_template_is_offline_and_has_interactive_controls():
    html=TEMPLATE.read_text(encoding='utf-8')
    assert html.count('__INLINE_PAYLOAD__')==1
    assert 'https://' not in html and 'http://' not in html
    assert 'URL.createObjectURL' in html and 'showGraphRiskOnly' in html
    assert 'fetch(' not in html


def test_fail_closed_on_missing_native_provenance_or_auto_join():
    a,b,c,d=make_payload()
    a['tiles'][0]['crop_id']='other'
    with pytest.raises(ValueError):build_payload(a,b,c,d)
    a,b,c,d=make_payload()
    c['graphs'][0]['candidate_edges'][0]['physics_join_authorized']=True
    with pytest.raises(ValueError):build_payload(a,b,c,d)
    a,b,c,d=make_payload()
    a['native_expert_gt_created']=1
    with pytest.raises(ValueError):build_payload(a,b,c,d)
