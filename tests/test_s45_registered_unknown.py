"""S45 guarantees transductive repeat evidence cannot silently become truth."""
import numpy as np
import cv2
import pytest
from bubble_chamber.s45_registered_unknown import (
    repeat_advisory,registered_uncertainty,downgrade_for_diagnostic_only,
    graph_repeat_warning,
)


def ev(a=1.,b=1.,ctrl=0.,fid=False,ident="a"):
    return {"draft_id":ident,"source_multiscale_ridge_coverage":a,
            "counterpart_same_registered_path_coverage":b,
            "displaced_same_shape_counterpart_control_coverage":ctrl,
            "known_fiducial_uncertainty_NOT_NATIVE_TRUTH":fid}


def test_registered_repeat_requires_both_support_and_shift_contrast():
    assert repeat_advisory(ev())["registered_repeat_risk"]
    assert not repeat_advisory(ev(b=.55))["registered_repeat_risk"]
    assert not repeat_advisory(ev(ctrl=.95))["registered_repeat_risk"]
    assert not repeat_advisory(ev(fid=True))["registered_repeat_risk"]
    assert repeat_advisory(ev())["native_particle_identity"]=="UNKNOWN"
    assert not repeat_advisory(ev())['heldout_training_eligibility']=="ALLOWED"


def test_missing_control_and_bad_values_fail_closed():
    assert not repeat_advisory(ev(ctrl=None))["registered_repeat_risk"]
    for v in (float('nan'),-1.,1.2):
        with pytest.raises(ValueError):repeat_advisory(ev(b=v))
    with pytest.raises(ValueError):repeat_advisory({})
    bad=ev();del bad['known_fiducial_uncertainty_NOT_NATIVE_TRUTH']
    with pytest.raises(ValueError):repeat_advisory(bad)


def make_image():
    im=np.full((160,160),220,dtype=np.uint8)
    cv2.line(im,(30,75),(125,75),20,2)
    pts=np.column_stack([np.linspace(30,125,40),np.full(40,75)]).tolist()
    return im,[{'draft_id':'a','points_xy':pts}]


def test_mask_downgrades_not_promotes_and_does_not_erase_candidates():
    im,c=make_image()
    risk,decisions=registered_uncertainty(im,im.copy(),c,[ev()])
    assert risk.sum()>0 and risk.dtype==bool
    assert decisions[0]['registered_repeat_risk']
    original=np.full(im.shape,255,dtype='uint8')
    original[75,30:126]=1
    original[76,30:126]=0
    updated,meta=downgrade_for_diagnostic_only(original,risk)
    assert np.all(updated[risk]==255)
    assert not np.any((original==255)&(updated!=255))
    assert not meta['leave_one_photo_out_model_training_allowed']
    assert meta['native_physics_ground_truth_created']==0
    assert len(c)==1


def test_does_not_mask_without_valid_repeat_or_when_fiducial():
    im,c=make_image()
    for e in (ev(ctrl=.98),ev(fid=True),ev(b=.2)):
        m,_=registered_uncertainty(im,im,c,[e]);assert not m.any()


def test_identity_mismatch_and_invalid_path_rejected():
    im,c=make_image()
    with pytest.raises(ValueError):registered_uncertainty(im,im,c,[ev(ident='x')])
    with pytest.raises(ValueError):registered_uncertainty(im,im,c,[ev(),ev()])
    with pytest.raises(ValueError):registered_uncertainty(im.astype('float32'),im,c,[ev()])
    bad=[{'draft_id':'a','points_xy':[[0,0],[1,0],[190,0],[190,1]]}]
    with pytest.raises(ValueError):registered_uncertainty(im,im,bad,[ev()])


def test_graph_keeps_all_edges_and_abstains_on_repeat():
    g={'nodes':[{'source_candidate_ids':['a']},{'source_candidate_ids':['b']}],
       'candidate_edges':[{'i':0,'j':1,'physics_join_authorized':False}]}
    r=graph_repeat_warning(g,{'a':True,'b':False})
    assert r['registered_repeat_edge_count']==1
    assert r['original_edge_count']==1 and r['physical_auto_joins_performed']==0
    assert r['edges_PRIVATE'][0]['status']=='REGISTERED_REPEAT_ABSTAIN'
    with pytest.raises(ValueError):graph_repeat_warning(g,{'a':True})
    with pytest.raises(ValueError):graph_repeat_warning({'nodes':g['nodes'],
                     'candidate_edges':[{'i':0,'j':5}]},{'a':False,'b':False})


def test_fiducial_repeat_union_counts_exact_and_no_autojoin():
    from bubble_chamber.s45_registered_unknown import combine_with_known_fiducial_guard
    a={'candidate_edges':[{'i':0,'j':1,'fiducial_overlap':True,'previously_ranked_unambiguous_UNVERIFIED':True},
                          {'i':0,'j':2,'fiducial_overlap':False,'previously_ranked_unambiguous_UNVERIFIED':True},
                          {'i':1,'j':2,'fiducial_overlap':False,'previously_ranked_unambiguous_UNVERIFIED':False}]}
    b={'edges_PRIVATE':[{'i':0,'j':1,'registered_repeat_endpoint_risk':True,'physical_auto_join_allowed':False},
                        {'i':0,'j':2,'registered_repeat_endpoint_risk':True,'physical_auto_join_allowed':False},
                        {'i':1,'j':2,'registered_repeat_endpoint_risk':False,'physical_auto_join_allowed':False}]}
    r=combine_with_known_fiducial_guard(a,b)
    assert (r['fiducial_risk'],r['repeat_risk'],r['overlap_both'])==(1,2,1)
    assert r['new_repeat_not_fiducial']==1
    assert r['union_any']==2 and r['unflagged_but_still_unverified']==1
    assert r['previously_ranked_unambiguous_union_risk']==2
    with pytest.raises(ValueError):combine_with_known_fiducial_guard(a,{'edges_PRIVATE':b['edges_PRIVATE'][:2]})
