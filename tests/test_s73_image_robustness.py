import numpy as np
import pytest
from bubble_chamber.s73_image_robustness import CONDITIONS,perturb


def test_fixed_family_deterministic_and_no_truth_api():
    gray=np.random.default_rng(99).integers(30,231,(192,192),dtype='uint8')
    assert CONDITIONS==('clean','contrast_072','blur_sigma_11','grain_sigma_9','two_film_streaks')
    for kind in CONDITIONS:
        a=perturb(gray,kind,42);b=perturb(gray,kind,42)
        assert a.shape==gray.shape and a.dtype==gray.dtype
        np.testing.assert_array_equal(a,b)
        assert not np.shares_memory(a,gray)
        if kind!='clean':assert np.count_nonzero(a!=gray)>0
    np.testing.assert_array_equal(gray,perturb(gray,'clean',21))


def test_strict_input_contract_rejects_private_identity_and_bad_rasters():
    a=np.zeros((192,192),np.uint8)
    for bad in (np.zeros((191,192),np.uint8),a.astype(float),np.zeros((192,192,3),np.uint8)):
        with pytest.raises(ValueError):perturb(bad,'clean',7)
    with pytest.raises(ValueError):perturb(a,'unknown',7)
    with pytest.raises(ValueError):perturb(a,'grain_sigma_9',-1)
    with pytest.raises(TypeError):perturb(a,'clean',7,gt_points=[])


def test_seed_affects_only_stochastic_arms():
    a=np.full((192,192),180,np.uint8)
    assert np.array_equal(perturb(a,'blur_sigma_11',1),perturb(a,'blur_sigma_11',3))
    assert not np.array_equal(perturb(a,'grain_sigma_9',1),perturb(a,'grain_sigma_9',3))
    assert not np.array_equal(perturb(a,'two_film_streaks',1),perturb(a,'two_film_streaks',3))
