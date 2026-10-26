"""Source-blind S77 tests; artificial graphics only, no H50Z native labels."""
import cv2
import numpy as np
import pytest
from bubble_chamber.s77_short_gap_instances import (
 ARMS,propose_pixel_gap_pairs,repair_pixel_only,MIN_GAP,RIVAL_MARGIN)


def image_line():
    im=np.full((192,192),210,np.uint8)
    cv2.line(im,(20,80),(110,80),25,2)
    return im


def curves():
    return [np.column_stack((np.linspace(20,54,22),np.full(22,80.))),
            np.column_stack((np.linspace(59,110,28),np.full(28,80.)))]


def test_short_supported_gap_merges_only_pixel_observed_components():
    im=image_line();orig=curves()
    result=repair_pixel_only(im,orig,'short_gap_relaxed')
    assert result['physical_identity_verified'] is False
    assert result['input_curves']==2
    assert result['merged_pairs']==1
    assert len(result['polylines'])==1
    assert len(result['polylines'][0])>len(orig[0])+len(orig[1])
    assert np.linalg.norm(result['polylines'][0][0]-result['polylines'][0][-1])>70
    assert np.array_equal(orig[0],curves()[0]) # source not mutated


def test_no_hallucinations_on_uniform_image_or_long_gap():
    im=np.full((192,192),220,np.uint8)
    assert repair_pixel_only(im,curves(),'short_gap_relaxed')['merged_pairs']==0
    far=curves();far[1][:,0]+=30
    assert repair_pixel_only(image_line(),far,'short_gap_relaxed')['merged_pairs']==0


def test_guard_does_not_erase_third_line_or_force_crossing_identity():
    im=image_line();cv2.line(im,(57,55),(57,108),35,2)
    other=np.column_stack((np.full(22,57.),np.linspace(55,108,22)))
    result=repair_pixel_only(im,curves()+[other],'short_gap_relaxed')
    assert result['merged_pairs']==0
    assert len(result['polylines'])==3


def test_direction_invariance_of_repair_and_determinism():
    im=image_line();orig=curves()
    base=repair_pixel_only(im,orig,'short_gap_relaxed')
    rev=repair_pixel_only(im,[orig[1][::-1],orig[0][::-1]],'short_gap_relaxed')
    assert base['merged_pairs']==rev['merged_pairs']==1
    assert len(base['polylines'][0])==len(rev['polylines'][0])
    z=np.asarray(base['polylines'][0])
    q=np.asarray(rev['polylines'][0])
    assert np.allclose(z,q) or np.allclose(z,q[::-1])
    duplicate=repair_pixel_only(im,orig,'short_gap_relaxed')
    assert np.array_equal(base['polylines'][0],duplicate['polylines'][0])


def test_baseline_is_exact_and_inference_has_no_GT_argument():
    im=image_line();orig=curves()
    assert repair_pixel_only(im,orig,'baseline')['merged_pairs']==0
    assert all(np.array_equal(a,b) for a,b in zip(orig,repair_pixel_only(im,orig,'baseline')['polylines']))
    assert MIN_GAP==1.6 and RIVAL_MARGIN==.13
    with pytest.raises(ValueError):repair_pixel_only(im.astype(float),orig,'short_gap_relaxed')
    with pytest.raises(ValueError):repair_pixel_only(im,orig,'GT_based_hypothesis')
    with pytest.raises(ValueError):repair_pixel_only(im,orig*17,'short_gap_relaxed')


def test_tiny_image_near_boundary_does_not_crash():
    im=image_line(); p=[np.column_stack((np.linspace(2,18,10),np.full(10,3.))),
                        np.column_stack((np.linspace(22,40,10),np.full(10,3.)))]
    r=repair_pixel_only(im,p,'short_gap_relaxed')
    assert r['merged_pairs']==0 and len(r['polylines'])==2
