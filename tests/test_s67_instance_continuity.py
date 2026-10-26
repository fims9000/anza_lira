"""S67 tests deliberately use manufactured graphics, never label native H50Z."""
import cv2
import numpy as np
import pytest
from bubble_chamber.s67_instance_continuity import (
    extract_instance_hypotheses,_candidate_pairings,_assemble,METHOD)


def test_reject_nonpixel_data_and_bounded_candidates():
    for image in (np.zeros((192,192),dtype=np.float32),
                  np.zeros((192,193),dtype=np.uint8),np.zeros((192,192,3),dtype=np.uint8)):
        with pytest.raises(ValueError):extract_instance_hypotheses(image)
    with pytest.raises(ValueError):extract_instance_hypotheses(np.zeros((192,192),dtype=np.uint8),max_curves=33)


def test_pure_line_has_no_fake_gt_or_physical_claim():
    image=np.full((192,192),205,dtype=np.uint8)
    cv2.line(image,(20,80),(172,80),15,2)
    x=extract_instance_hypotheses(image)
    assert x['method']==METHOD
    assert x['synthetic_or_native_GT_used'] is False
    assert x['physical_identity_verified'] is False
    assert all(p.ndim==2 and p.shape[1]==2 and np.isfinite(p).all() for p in x['polylines'])


def test_unambiguous_crossing_pairing_from_local_tangents():
    # Parts terminate towards one center. Must not confuse orthogonal arms.
    top=np.array([[90,y] for y in range(20,89,3)],float)
    bottom=np.array([[90,y] for y in range(98,164,3)],float)
    left=np.array([[x,92] for x in range(20,89,3)],float)
    right=np.array([[x,92] for x in range(98,164,3)],float)
    parts=[top,bottom,left,right]
    links=_candidate_pairings(parts,[np.array([91,92.])])
    got={frozenset((x[0][0],x[1][0])) for x in links}
    assert got=={frozenset((0,1)),frozenset((2,3))}
    merged=_assemble(parts,links)
    assert len(merged)==2
    assert len(set().union(*[set(map(tuple,x)) for x in merged]))>4


def test_ambiguous_crossing_must_abstain():
    # One opposite branch and two nearly equal alternatives for same port.
    a=np.array([[80,y] for y in range(20,89,3)],float)
    b=np.array([[84,y] for y in range(96,165,3)],float)
    c=np.array([[76,y] for y in range(96,165,3)],float)
    links=_candidate_pairings([a,b,c],[np.array([80,92.])])
    assert links==[]


def test_deterministic_on_same_pixels():
    image=np.full((192,192),190,dtype=np.uint8)
    cv2.line(image,(20,95),(172,95),12,2)
    cv2.line(image,(95,20),(95,172),12,2)
    x=extract_instance_hypotheses(image)
    y=extract_instance_hypotheses(image)
    assert {k:x[k] for k in ('prepaired_components','provisional_continuity_pairs','pixel_junctions')} == {k:y[k] for k in ('prepaired_components','provisional_continuity_pairs','pixel_junctions')}
    assert len(x['polylines'])==len(y['polylines'])
    for a,b in zip(x['polylines'],y['polylines']):np.testing.assert_array_equal(a,b)
