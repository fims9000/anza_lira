"""S68 safety and topology tests; all pixel figures manufactured, not native GT."""
import cv2
import numpy as np
import pytest

from bubble_chamber.s68_branch_preserving_instances import (
    moore_arm_count, split_skeleton_at_junctions,
    extract_fragment_hypotheses, _refine_endpoints, ARMS,
)


def _cross_image():
    im = np.full((192, 192), 205, dtype='uint8')
    cv2.line(im, (20, 95), (170, 95), 10, 2)
    cv2.line(im, (95, 20), (95, 170), 10, 2)
    return im


def test_moore_crossing_detects_four_arms_not_two_on_diagonal():
    sk = np.zeros((35, 35), bool)
    sk[17, 5:30] = True
    sk[5:30, 17] = True
    k = moore_arm_count(sk)
    assert k[17, 17] == 4
    assert (k[17, 9] == 2) and (k[9, 17] == 2)
    cut, centers = split_skeleton_at_junctions(sk)
    assert len(centers) == 1
    assert cut[17, 17] == 0
    assert cut[17, 7] == 1
    diagonal = np.zeros((35, 35), bool)
    for t in range(5, 27):
        diagonal[t, t] = True
    dcut, dcent = split_skeleton_at_junctions(diagonal)
    assert len(dcent) == 0
    assert np.array_equal(dcut, diagonal.astype('uint8'))


def test_topology_pixel_only_outputs_and_no_physics_assertions():
    for option in (False, True):
        a = extract_fragment_hypotheses(_cross_image(), refine=option)
        assert a['pixel_junctions'] >= 1
        assert len(a['polylines']) >= 4
        assert len(a['polylines']) <= 32
        assert a['physical_identity_verified'] is False
        assert a['native_expert_labels_used'] is False
        assert a['synthetic_labels_used_by_predictor'] is False
        assert a['automatic_physical_join_authorized'] is False
        assert all(x.ndim == 2 and x.shape[1] == 2 and len(x) >= 8
                   and np.isfinite(x).all() for x in a['polylines'])
        # Do NOT merge the four arms into one invented physical particle.
        assert max(np.linalg.norm(x[0]-x[-1]) for x in a['polylines']) < 125


def test_source_limits_fail_closed():
    for bad in (None, np.zeros((192,192),dtype='float32'),
                np.zeros((192,193),dtype='uint8'),
                np.zeros((192,192,3),dtype='uint8')):
        with pytest.raises(ValueError):
            extract_fragment_hypotheses(bad)
    for n in (0,33,True,1.0):
        with pytest.raises(ValueError):
            extract_fragment_hypotheses(_cross_image(),max_curves=n)
    with pytest.raises(ValueError):
        extract_fragment_hypotheses(_cross_image(),refine=1)
    with pytest.raises(ValueError):
        split_skeleton_at_junctions(np.zeros((50,50),bool),radius=3)


def test_refinement_never_extrapolates_large_or_unsupported_gap():
    path=np.column_stack((np.linspace(30,88,22),np.full(22,90.)))
    score=np.zeros((192,192),np.float32)
    np.testing.assert_array_equal(_refine_endpoints(path,[np.array([90.,90.])],score),path)
    score[87:94,87:95]=1
    extended=_refine_endpoints(path,[np.array([90.,90.])],score)
    assert extended[-1,0]>path[-1,0]
    assert 0<extended[-1,0]-path[-1,0]<=2.5
    np.testing.assert_array_equal(_refine_endpoints(path,[np.array([104.,90.])],score),path)


def test_identical_pixels_deterministic_and_bounded():
    x=_cross_image()
    for refine in (False,True):
        a=extract_fragment_hypotheses(x,refine=refine,max_curves=3)
        b=extract_fragment_hypotheses(x,refine=refine,max_curves=3)
        assert len(a['polylines'])==len(b['polylines'])<=3
        for p,q in zip(a['polylines'],b['polylines']):
            np.testing.assert_array_equal(p,q)
    assert set(ARMS) == {'s39_original','s67_crossing_orientation_v1',
                         's68_branch_split','s68_branch_endpoint_refine'}


def test_equidistant_junctions_are_deterministic_not_array_sort():
    path=np.column_stack((np.linspace(30,88,22),np.full(22,90.)))
    score=np.ones((192,192),np.float32)
    centers=[np.array([90.,92.]),np.array([90.,88.])]
    for _ in range(3):
        x=_refine_endpoints(path,centers,score)
        assert np.isfinite(x).all()
        assert len(x)==len(path)
        np.testing.assert_array_equal(x,_refine_endpoints(path,centers,score))
