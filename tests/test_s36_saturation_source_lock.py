from __future__ import annotations
import numpy as np
import pytest
import torch
from bubble_chamber.s33_photometric_domain import image_input
from bubble_chamber.s36_saturation_normalization import (
    saturation_blend_input,matched_s36_photo_views,holdout_exposure,
    S36_TEST_FORWARD_SEEDS,S36_TEST_REVERSE_SEEDS)
from scripts.research.bubble_chamber.s36_eval_saturation import _load_saturation


def test_normalizer_unclipped_input_equals_predeclared_quantile():
    rng=np.random.default_rng(111)
    gray=rng.integers(25,230,size=(64,64),dtype=np.uint8)
    out,diag=saturation_blend_input(gray,return_diagnostics=True)
    assert diag['local_branch_weight_uncalibrated']==0
    np.testing.assert_array_equal(out,image_input(gray,'quantile'))
    assert diag['physical_track_probability'] is False


def test_complete_saturation_switches_to_local_but_never_recovers_pixels():
    gray=np.full((64,64),255,np.uint8)
    gray[10:20,10:20]=0
    out,diag=saturation_blend_input(gray,return_diagnostics=True)
    assert diag['local_branch_weight_uncalibrated']==1.
    np.testing.assert_allclose(out,image_input(gray,'local'),atol=1e-6)
    assert np.isfinite(out).all()
    assert np.min(out)>=0 and np.max(out)<=1


def test_smooth_intermediate_weight_not_target_leakage():
    arr=np.arange(64*64,dtype=np.uint16).reshape(64,64)
    gray=(30+(arr%170)).astype(np.uint8)
    gray.flat[:int(gray.size*.10)]=255
    x,diag=saturation_blend_input(gray,return_diagnostics=True)
    assert 0<diag['local_branch_weight_uncalibrated']<1
    assert x.shape==gray.shape
    assert x.dtype==np.float32


def test_two_photometric_views_reproducible_and_unchanged_geometry():
    gray=np.random.default_rng(55).integers(100,250,size=(64,64),dtype=np.uint8)
    x1,x2=matched_s36_photo_views(gray,seed=44)
    y1,y2=matched_s36_photo_views(gray,seed=44)
    np.testing.assert_array_equal(x1,y1);np.testing.assert_array_equal(x2,y2)
    assert x1.shape==(64,64) and x2.shape==(64,64)
    assert len(set(S36_TEST_FORWARD_SEEDS)&set(S36_TEST_REVERSE_SEEDS))==0
    assert S36_TEST_FORWARD_SEEDS==(20261049,20261050,20261051,20261052)


def test_clipped_exposure_is_deliberately_harder_than_source_image():
    gray=np.full((64,64),210,dtype=np.uint8)
    bright=holdout_exposure(gray,'highlight_clip')
    assert np.mean(bright>=254)>0.95
    assert np.mean(gray>=254)==0
    with pytest.raises(ValueError,match='Unregistered'):
        holdout_exposure(gray,'future_film')


def test_invalid_input_or_untrusted_checkpoint_fails_closed(tmp_path):
    with pytest.raises(ValueError,match='Expected source'):
        saturation_blend_input(np.ones((64,64),dtype=np.float64))
    with pytest.raises(ValueError,match='Expected source'):
        saturation_blend_input(np.ones((16,16),dtype=np.uint8))
    p=tmp_path/'azunet_saturation_blend_train005_seed20261025.pt'
    torch.save({'schema':'EXPERT_GROUND_TRUTH','architecture':'azunet','mode':'saturation_blend'},p)
    with pytest.raises(ValueError,match='other/photo labels'):
        _load_saturation(tmp_path,'005',20261025)
