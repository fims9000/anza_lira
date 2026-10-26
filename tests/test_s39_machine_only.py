"""S39 machine-only correctness/security, not validation against physical particle GT."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import cv2

from bubble_chamber.s39_machine_auto import (
    ridge_evidence, fixed_budget_mask, extract_ridges, annotate_tile,
    contrast_baseline,canny_score,SCHEMA)
from bubble_chamber.s39_graph_proposals import collapse_near_duplicates,link_graph
from scripts.research.bubble_chamber.s39_run_machine_only import predict_cross_photo,run as run_machine
from scripts.research.bubble_chamber.s39_full_frame_audit import fixed_sample,_image
from scripts.research.bubble_chamber.s39_benchmark_machine_segmentation import tolerant_fraction,detection_scores


def image_with_line():
    a=np.full((192,192),210,dtype='uint8')
    cv2.line(a,(50,28),(50,162),70,2)
    cv2.line(a,(92,30),(150,145),130,2)
    return a


def fake_candidate(start,end,name='cand',method='other'):
    x=np.linspace(start[0],end[0],28);y=np.linspace(start[1],end[1],28)
    return {'draft_id':name,'source_method':method,'source':'S39_MULTISCALE_RIDGE_PROPOSAL',
            'points_xy':np.column_stack((x,y)).tolist(),
            'review_priority_NOT_PHYSICS_PROBABILITY':.7}


def test_ridge_evidence_takes_real_uint8_and_not_constant_as_lines():
    image=image_with_line()
    e=ridge_evidence(image)
    assert e['scales'].shape==(3,192,192)
    assert np.isfinite(e['score']).all()
    assert e['score'][100,50]>e['score'][100,55]
    constant=ridge_evidence(np.full((192,192),200,dtype='uint8'))
    assert np.count_nonzero(fixed_budget_mask(constant['score']))==0
    with pytest.raises(ValueError):ridge_evidence(np.ones((192,192),dtype=float))


def test_fixed_budget_must_be_bounded_and_deterministic():
    a=np.arange(10000,dtype='float32').reshape(100,100)
    b=fixed_budget_mask(a,fraction=.035)
    assert b.sum()==350
    assert np.array_equal(b,fixed_budget_mask(a,fraction=.035))
    with pytest.raises(ValueError):fixed_budget_mask(a,fraction=.9)
    with pytest.raises(ValueError):fixed_budget_mask(a+np.float32(np.inf))


def test_new_line_discovery_recovers_controlled_dark_curves():
    e=ridge_evidence(image_with_line())
    paths=extract_ridges(e)
    assert paths
    assert any(np.min(np.linalg.norm(p-np.array([50,99]),axis=1))<5 for p in paths)


def test_all_machine_labels_preserve_unknown_not_fabricated_real_gt():
    image=image_with_line()
    proposals,out=annotate_tile(image,[],az_map=np.zeros_like(image,dtype='float32'))
    mask=out['pseudo_mask']
    assert len(proposals)>0
    assert set(np.unique(mask)).issubset({0,1,255})
    assert int((mask==255).sum())>image.size//2
    assert out['stats']['physics_verified_count']==0
    assert out['stats']['original_empty_tile_has_new_proposals'] is True
    assert all('review_priority_NOT_PHYSICS_PROBABILITY' in p for p in proposals)


def test_fake_AZ_bright_probabilities_do_not_become_ground_truth():
    image=image_with_line()
    with pytest.raises(ValueError):annotate_tile(image,[],az_map=np.ones((10,10)))
    with pytest.raises(ValueError):annotate_tile(image,[],az_map=np.full_like(image,2.,dtype='float32'))


def test_no_end_to_end_graph_auto_merge_and_duplicate_collapsing():
    a=fake_candidate((25,100),(95,100),'a')
    b=fake_candidate((25,100),(95,100),'b')
    c=fake_candidate((112,100),(175,100),'c')
    nodes=collapse_near_duplicates([a,b,c])
    assert len(nodes)==2
    assert set(nodes[0]['source_candidate_ids'])=={'a','b'}
    out=link_graph([a,b,c],max_nodes=10)
    assert len(out['candidate_edges'])>=1
    assert out['physics_auto_links_performed']==0
    assert out['all_edges_require_future_independent_physics_validation']
    assert all(not row['physics_join_authorized'] for row in out['candidate_edges'])
    assert all(row['native_relation_truth']=='UNKNOWN' for row in out['candidate_edges'])


def test_graph_abstains_without_plausible_continuation():
    a=fake_candidate((25,100),(95,100),'a')
    b=fake_candidate((150,180),(175,180),'b')
    out=link_graph([a,b],max_nodes=10)
    assert out['candidate_edges']==[] and out['unambiguous_ranked_edges_UNVERIFIED']==[]


def test_graph_ambiguity_and_one_to_one():
    a=fake_candidate((25,100),(95,100),'a')
    b=fake_candidate((110,99),(145,99),'b')
    c=fake_candidate((112,102),(145,105),'c')
    out=link_graph([a,b,c],max_nodes=10,margin=.5)
    selected=out['unambiguous_ranked_edges_UNVERIFIED']
    endpoints=[z for p in selected for z in (p['i'],p['j'])]
    assert len(set(endpoints))==len(endpoints)
    assert out['physics_auto_links_performed']==0


def test_untrusted_private_source_is_rejected_before_reading(tmp_path):
    p=tmp_path/'fake.zip';p.write_bytes(b'unsafe not source')
    with pytest.raises(ValueError,match='immutable verified source'):
        run_machine(p,tmp_path,tmp_path/'out')


def test_unbiased_grid_sampling_is_repeatable_balanced_and_in_bounds():
    p=fixed_sample(9020,26151,'005',n=64)
    assert p==fixed_sample(9020,26151,'005',n=64)
    assert len(set(p))==64
    assert all(x>=0 and y>=0 and x+512<=9020 and y+512<=26151 for x,y in p)
    groups=np.bincount([min(7,int((y+256)/26151*8)) for x,y in p],minlength=8)
    assert np.all(groups==8)
    assert p!=fixed_sample(9268,25324,'006',n=64)


def test_bad_bmp_header_does_not_disable_pillow_safety(tmp_path):
    p=tmp_path/'H50Z_030_01_005.bmp';p.write_bytes(b'bad BMP')
    with pytest.raises(Exception):_image(p,'005')


class DummyModel(torch.nn.Module):
    def forward(self,x):return torch.zeros_like(x)


def test_az_stitched_patch_map_exact_shape_and_no_uncovered_seams():
    x=image_with_line()
    y=predict_cross_photo(x,DummyModel(),step=56,batch=8)
    assert y.shape==x.shape
    assert np.allclose(y,.5,atol=1e-6)


def test_synthetic_measure_not_native_precision():
    found=np.zeros((20,20),dtype=bool);found[10,10]=True
    stroke=np.zeros_like(found);stroke[10,12]=True
    assert tolerant_fraction(found,stroke,radius=2)==1.
    assert tolerant_fraction(found,stroke,radius=1)==0.
    with pytest.raises(ValueError):tolerant_fraction(found.astype('uint8'),stroke)


def test_score_comparator_arms_have_equal_image_shape():
    image=image_with_line()
    x=detection_scores(image,np.full(image.shape,.5,dtype='float32'))
    assert len(x)==6
    assert all(m.shape==image.shape and np.isfinite(m).all() for m in x.values())
    assert np.max(x['local_darkness'])>0
