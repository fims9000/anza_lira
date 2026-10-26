"""S41 invariants: no pseudo-truth promotion, strict lineage, reproducible masks."""
import cv2
import numpy as np
import pytest
from bubble_chamber.s41_machine_quality import (
    exposure_views, ridge_support_maps, candidate_stability,
    uncertainty_region, refine_pseudo_mask, annotate_s41)


def image():
    a=np.full((192,192),210,dtype=np.uint8)
    cv2.line(a,(12,65),(180,65),32,2)
    return a


def cand(x=20,y=65,ident='a'):
    return {'draft_id':ident,'points_xy':np.column_stack((np.linspace(x,x+80,30),
                                                       np.full(30,y))).tolist()}


def zone():
    return {'photo':'005','label':'f1','center_xy_PRIVATE':[60.,65.],
            'machine_ambiguity_radius_px':40.,'arms_angles_rad_PRIVATE':[]}


def test_exposure_is_deterministic_and_not_creating_floats():
    a=image()
    v=exposure_views(a)
    assert set(v)=={'identity','gamma_0p78','gamma_1p30','blur_sigma_1p1'}
    assert all(x.dtype==np.uint8 and x.shape==a.shape for x in v.values())
    assert np.array_equal(v['identity'],a)
    assert np.array_equal(exposure_views(a)['blur_sigma_1p1'],v['blur_sigma_1p1'])
    with pytest.raises(ValueError):exposure_views(a.astype(float))


def test_stability_never_claims_physics_truth():
    maps=ridge_support_maps(image())
    r=candidate_stability(cand()['points_xy'],maps)
    assert len(r['pixel_ridge_support_fraction_by_view'])==4
    assert r['native_particle_identity']=='UNKNOWN'
    assert isinstance(r['photometric_machine_stable_NOT_GT'],bool)
    with pytest.raises(ValueError):candidate_stability([[1,1],[400,1]],maps)
    with pytest.raises(ValueError):candidate_stability(cand()['points_xy'],{'identity':maps['identity']})


def test_fiducial_marks_unknown_both_positive_and_negative_not_background():
    old=np.full((192,192),255,dtype=np.uint8)
    old[60:90,10:120]=1
    old[90:120,10:120]=0
    scores=[{'photometric_machine_stable_NOT_GT':True}]
    region,flags=uncertainty_region(old.shape,[cand()],scores,[zone()],(0,0),'005')
    new,changes=refine_pseudo_mask(old,region)
    assert flags[0]['fiducial_risk']
    assert new[65,60]==255 and new[105,60]==255
    assert new[65,115]==1 and new[105,115]==0
    assert changes['native_expert_truth_created']==0
    assert (new==1).sum()< (old==1).sum()
    assert (new==0).sum()< (old==0).sum()
    assert not np.shares_memory(new,old)


def test_unstable_photo_score_must_not_censor_pseudo_labels_after_negative_control():
    old=np.ones((192,192),dtype=np.uint8)
    r,flags=uncertainty_region(old.shape,[cand()],
        [{'photometric_machine_stable_NOT_GT':False}],[],(0,0),'005')
    new,_=refine_pseudo_mask(old,r)
    assert flags[0]['unstable_photo_response'] and flags[0]['photo_stability_is_advisory_only']
    assert not r.any() and np.array_equal(new, old)
    assert not np.any(new==0)


def test_outside_photo_fiducial_provenance_not_applied():
    s=[{'photometric_machine_stable_NOT_GT':True}]
    region,flags=uncertainty_region((192,192),[cand()],s,[zone()],(0,0),'006')
    assert not region.any() and not flags[0]['fiducial_risk']


def test_bad_mask_cannot_silently_become_truth():
    arr=np.zeros((192,192),dtype=np.uint8)
    arr[1,1]=4
    with pytest.raises(ValueError):refine_pseudo_mask(arr,np.zeros(arr.shape,dtype=bool))
    arr[1,1]=1
    with pytest.raises(ValueError):refine_pseudo_mask(arr,arr)


def test_end_to_end_preserves_ignored_pixels_and_all_sources():
    a=image();mask=np.full(a.shape,255,dtype=np.uint8)
    mask[60:70,:]=1;mask[100:110,:]=0
    original=mask.copy()
    x=annotate_s41(a,mask,[cand()],[],(0,0),'005')
    y=annotate_s41(a,mask,[cand()],[],(0,0),'005')
    assert np.array_equal(x['refined_mask_PRIVATE'],y['refined_mask_PRIVATE'])
    assert np.array_equal(mask,original)
    assert x['native_physics_gt_count']==0
    assert x['physical_auto_join_allowed'] is False
    assert int((x['refined_mask_PRIVATE']==255).sum())>=int((mask==255).sum())


def test_graph_stability_is_advisory_and_fiducial_must_abstain():
    from scripts.research.bubble_chamber.s41_run_machine_quality import combined_graph_guard
    graph={'nodes':[
        {'draft_id':'a','source_candidate_ids':['a']},
        {'draft_id':'b','source_candidate_ids':['b']},
        {'draft_id':'c','source_candidate_ids':['c']}],
        'candidate_edges':[{'i':0,'j':1},{'i':1,'j':2}],
        'unambiguous_ranked_edges_UNVERIFIED':[{'i':0,'j':1}]}
    prior={'nodes':[{}, {}, {}],'candidate_edges':[
        {'i':0,'j':1,'fiducial_overlap':False},
        {'i':1,'j':2,'fiducial_overlap':True}]}
    flags={'a':{'fiducial_risk':False,'unstable_photo_response':True},
           'b':{'fiducial_risk':False,'unstable_photo_response':False},
           'c':{'fiducial_risk':True,'unstable_photo_response':False}}
    res=combined_graph_guard(graph,prior,flags)
    assert res['native_physics_auto_links']==0
    assert res['edges'][0]['status']=='UNVERIFIED_CANDIDATE_NO_AUTOJOIN'
    assert res['edges'][0]['unstable_endpoint_ADVISORY_ONLY'] is True
    assert res['edges'][1]['status']=='KNOWN_FIDUCIAL_RISK_ABSTAIN'
    assert res['summary']['fiducial_risk']==1
    assert res['summary']['stability_warning_advisory_only']==1


def test_graph_wrong_edge_order_fails_closed():
    from scripts.research.bubble_chamber.s41_run_machine_quality import combined_graph_guard
    graph={'nodes':[{'source_candidate_ids':['a']},{'source_candidate_ids':['b']}],
           'candidate_edges':[{'i':0,'j':1}]}
    prior={'nodes':[{},{}],'candidate_edges':[{'i':1,'j':0,'fiducial_overlap':False}]}
    flags={'a':{'fiducial_risk':False,'unstable_photo_response':False},
           'b':{'fiducial_risk':False,'unstable_photo_response':False}}
    with pytest.raises(ValueError):combined_graph_guard(graph,prior,flags)


def test_photo_teacher_source_leakage_is_not_independent_validation():
    # The S39 opposite-photo AZ model was trained on test photo. A student
    # trained on its pseudo labels cannot honestly claim photo-disjoint testing.
    src=[{'photo_id':'005','source_AZ_model_trained_on_other_photo':'006'},
         {'photo_id':'006','source_AZ_model_trained_on_other_photo':'005'}]
    for photo in ('005','006'):
        row=next(r for r in src if r['photo_id']==photo)
        test_photo='006' if photo=='005' else '005'
        assert row['source_AZ_model_trained_on_other_photo']==test_photo
