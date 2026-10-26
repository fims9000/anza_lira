"""S52 only extends existing machine-only risk annotations."""
from __future__ import annotations
import pytest
from scripts.research.bubble_chamber.s52_build_machine_showcase import join_advisories, build_template


def sources():
    old={'native_GT_count':0,'physical_auto_links':0,
         'summary':{'crops':1,'machine_candidates':2,'graph_edges':1,'risky_edges':0},
         'tiles':[{'crop_id':'tile0','photo':'005',
             'graph_edges':[{'i':0,'j':1,'risky':False,'bridge':[[2,2],[5,5]]}],
             'candidates':[{'draft_id':'x','points':[[2,2],[5,5]]}]}]}
    new={'native_no_physics_gt':True,'details_PRIVATE':[{'crop_id':'tile0','photo':'005',
         'edges_PRIVATE':[{'i':0,'j':1,'old_S49_warning':False,
                           'collision_warning':True,'native_track_relation_truth':'UNKNOWN',
                           'physics_auto_join_authorized':False}]}]}
    return old,new


def test_composes_new_warning_without_mutating_original():
    a,b=sources()
    new=join_advisories(a,b)
    edge=new['tiles'][0]['graph_edges'][0]
    assert edge['risky'] and edge['s51_third_line_overlap_advisory']
    assert edge['s49_prior_warning'] is False
    assert edge['physical_auto_join_authorized'] is False
    assert edge['native_relation_truth']=='UNKNOWN'
    assert a['tiles'][0]['graph_edges'][0]['risky'] is False
    assert new['summary']['S51_additional_overlap_advisories']==1


def test_rejects_bad_provenance_and_autojoin():
    a,b=sources();a['native_GT_count']=1
    with pytest.raises(ValueError):join_advisories(a,b)
    a,b=sources();b['details_PRIVATE'][0]['edges_PRIVATE'][0]['physics_auto_join_authorized']=True
    with pytest.raises(ValueError):join_advisories(a,b)
    a,b=sources();b['details_PRIVATE'][0]['edges_PRIVATE'][0]['i']=1
    with pytest.raises(ValueError):join_advisories(a,b)


def test_offline_viewer_has_correct_controls():
    template=build_template()
    assert 'showCollisionOnly' in template
    assert '__INLINE_PAYLOAD__' in template
    assert 'https://' not in template and 'http://' not in template
    assert 'fetch(' not in template
    assert 'физически подтверждённых' not in template or 'НЕ ПОДТВЕРЖДЁННАЯ' in template
