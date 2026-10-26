import numpy as np
import pytest
from bubble_chamber.s62_image_candidate_diagnostics import (
    image_only_predictions,proxy_mask_fragment_support,
    proxy_extracted_piece_support,scene_upper_bound_from_proxy_support)


def test_image_only_detector_accepts_only_image_pixels():
    a=np.full((192,192),188,dtype=np.uint8)
    a[95:98,27:164]=60
    out=image_only_predictions(a)
    assert len(out['masks'])==3
    for mask in out['masks'].values():
        assert mask.dtype==bool and mask.shape==a.shape and mask.mean()<=.036
    with pytest.raises(ValueError):image_only_predictions(np.ones((192,192)))


def test_proxy_metric_orientations_and_endpoint_requirement():
    x=np.linspace(30,150,31)
    curve=np.column_stack((x,np.ones(31)*97))
    mask=np.zeros((192,192),bool);mask[97,30:151]=True
    a=proxy_mask_fragment_support(mask,curve)
    b=proxy_mask_fragment_support(mask,curve[::-1])
    assert a['usable_proxy_fragment'] and a==b
    mask[97,146:153]=False
    assert not proxy_mask_fragment_support(mask,curve)['usable_proxy_fragment']


def test_empty_image_only_pieces_cannot_claim_detected_proxy():
    t=np.column_stack((np.arange(10,41),np.full(31,50)))
    assert proxy_extracted_piece_support([],t)['usable_proxy_fragment'] is False


def test_truth_labels_are_confined_to_evaluator_never_source_detector():
    x=[{'usable_proxy_fragment':True} for _ in range(7)]
    r=scene_upper_bound_from_proxy_support(x)
    assert r['proxy_true_joins_upper_bound']==4
    assert r['proxy_complete_three_piece_paths_upper_bound']==2
    assert r['synthetic_decoy_fragment_usable']==1
    x[1]['usable_proxy_fragment']=False
    assert scene_upper_bound_from_proxy_support(x)['proxy_true_joins_upper_bound']==2
