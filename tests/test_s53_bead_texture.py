"""S53 machine-only texture study: invariants and simulator label safety."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from bubble_chamber.s53_bead_texture import FEATURE_NAMES, texture_features, merge_features
from scripts.research.bubble_chamber.s53_benchmark_bead_texture import (
    PROTOCOL, FITTING_NEGATIVES, TEST_ONLY_NEGATIVES, RENDERERS,
    renderer_mask, render_variants, _paired_bootstrap_delta, evaluate,
)


def curve():
    t=np.linspace(0,1,121)
    return np.column_stack((25.+125.*t,97.+4.*np.sin(3*t)))


def test_textured_feature_descriptor_finite_deterministic_and_no_side_effects():
    img=np.full((192,192),178,dtype='uint8')
    cv2.polylines(img,[np.rint(curve()).astype('int32')],False,60,1)
    before=img.copy()
    a=texture_features(img,curve())
    b=texture_features(img,curve())
    assert len(a)==len(FEATURE_NAMES)
    assert np.array_equal(a,b) and np.isfinite(a).all()
    assert np.array_equal(img,before)


def test_invalid_and_unregistered_input_fails():
    img=np.full((192,192),200,dtype='uint8')
    for wrong in [img.astype('float32'), img[:,:,None], img[:64]]:
        with pytest.raises(ValueError): texture_features(wrong,curve())
    with pytest.raises(ValueError): texture_features(img,curve()+400)
    with pytest.raises(ValueError): merge_features(np.ones((5,14)),np.ones((4,len(FEATURE_NAMES))))


def test_origin_indistinguishable_synthetic_clone_is_truly_same_renderer():
    p=curve(); shape=(192,192)
    a=renderer_mask('simulated_bead_track',p,np.random.default_rng(14),shape)
    b=renderer_mask('same_bead_renderer_scratch',p,np.random.default_rng(14),shape)
    assert np.array_equal(a,b)
    assert not any(k in FITTING_NEGATIVES for k in TEST_ONLY_NEGATIVES)
    assert len(RENDERERS)==6


def test_all_renderers_retain_paired_geometric_paths_and_unmodified_background():
    raw=np.full((192,192),201,dtype='uint8')
    before=raw.copy()
    a=render_variants(raw,curve(),seed=41,include_open_set=True)
    b=render_variants(raw,curve(),seed=41,include_open_set=True)
    assert list(a)==list(RENDERERS)
    assert all(np.array_equal(a[k],b[k]) for k in RENDERERS)
    assert all(x.shape==raw.shape and x.dtype==np.uint8 for x in a.values())
    assert all(np.count_nonzero(x < 201)>30 for x in a.values())
    assert np.array_equal(raw,before)
    assert list(render_variants(raw,curve(),seed=41,include_open_set=False))==list(RENDERERS[:4])


def test_per_scene_bootstrap_has_zero_interval_for_identical_models():
    # Identical scores provide a strict no-effect control.
    scenes=np.repeat(np.arange(10),6)
    y=np.tile([1,0,0,0,0,0],10)
    data={'labels':y,'scenes':scenes}
    scores=np.tile([.8,.1,.3,.5,.4,.6],10)
    d=_paired_bootstrap_delta(data,scores,scores,seed=17,iterations=25)
    assert d['delta_AUROC_S53_minus_S43_SYNTHETIC']==0
    assert d['simulator_scene_bootstrap_CI_95pct']==[0,0]
    assert d['physical_run_disjoint_CI'] is False


def test_frozen_protocol_and_calibration_exclude_open_set():
    assert PROTOCOL['train_seeds']=={'005':20265301,'006':20265311}
    assert PROTOCOL['test_seeds']['005']==[20265303,20265304]
    assert PROTOCOL['source_test_frame_count']==36
    assert not any(name in FITTING_NEGATIVES for name in TEST_ONLY_NEGATIVES)
    assert PROTOCOL['same_background_path_and_nominal_ink_for_all_scene_variants']
