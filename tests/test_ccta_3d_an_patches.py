"""3D CCTA physical sampling geometry with no labels or clinical examples."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_an_patches import PatchConfig,extract_candidate_patch,local_frame


def synthetic_ct():
    xyz=np.indices((50,55,60),dtype=np.float32)
    ct=150+1.8*xyz[0]+2*xyz[1]-xyz[2]
    affine=np.diag([.8,1.1,1.4,1.])
    affine[:3,3]=[8,11,-6]
    ras=np.array([[25,24,25],[31,25,28],[26,30,29]])
    lps=(affine[:3,:3]@ras.T).T+affine[:3,3]
    lps*=np.array([-1,-1,1])
    return ct,affine,lps


def test_physical_patch_and_mask_input_have_same_shape_for_pair_and_junction():
    ct,affine,pts=synthetic_ct()
    cfg=PatchConfig(side=16,voxel_mm=1.0)
    for selected in (pts[:2],pts):
        patch,audit=extract_candidate_patch(ct,affine,selected,cfg)
        assert patch.shape==(2,16,16,16)
        assert patch.dtype==np.float32
        assert np.isfinite(patch).all()
        assert 0<=patch[1].min()<=patch[1].max()<=1.
        assert -1<=patch[0].min()<=patch[0].max()<=1.
        assert audit["sampling_inside_volume_fraction"]>.95
        assert audit["candidate_degree"]==len(selected)
        assert audit["orientation_det"]==pytest.approx(1.0,abs=1e-8)


def test_same_endpoint_heatmap_but_different_ct_changes_only_hu_channel():
    ct,affine,pts=synthetic_ct()
    one,_=extract_candidate_patch(ct,affine,pts[:2],PatchConfig(side=16))
    two,_=extract_candidate_patch(ct+400,affine,pts[:2],PatchConfig(side=16))
    assert np.array_equal(one[1],two[1])
    assert not np.array_equal(one[0],two[0])


def test_frame_is_right_handed_and_rejects_degeneracy():
    _,_,pts=synthetic_ct()
    center,frame=local_frame(pts[:2])
    assert frame.T@frame==pytest.approx(np.eye(3))
    assert np.linalg.det(frame)==pytest.approx(1.)
    with pytest.raises(ValueError,match="Coincident"):
        local_frame(np.stack([pts[0],pts[0]]))
    with pytest.raises(ValueError,match="even"):
        PatchConfig(side=15)


def test_3d_patch_invariant_to_pair_and_junction_endpoint_permutations():
    from itertools import permutations
    ct,affine,pts=synthetic_ct()
    cfg=PatchConfig(side=16,voxel_mm=1.)
    for selected in (pts[:2],pts):
        reference,ref_info=extract_candidate_patch(ct,affine,selected,cfg)
        for order in permutations(range(len(selected))):
            rearranged=selected[list(order)]
            value,info=extract_candidate_patch(ct,affine,rearranged,cfg)
            assert np.array_equal(value,reference)
            assert info["patch_center_lps_mm"]==pytest.approx(
                ref_info["patch_center_lps_mm"]
            )


def test_3d_endpoint_heatmap_coverage_is_explicit_in_physical_mm():
    ct,affine,_=synthetic_ct()
    ras=np.array([[25.,25.,25.],[40.,25.,25.]])
    pts=(affine[:3,:3]@ras.T).T+affine[:3,3]
    pts*=np.array([-1.,-1.,1.])
    small=PatchConfig(side=8,voxel_mm=1.)
    _,meta=extract_candidate_patch(ct,affine,pts,small)
    assert not meta["all_endpoints_inside_patch"]
    assert meta["endpoint_patch_margin_mm"]<0
    large=PatchConfig(side=24,voxel_mm=1.)
    _,meta2=extract_candidate_patch(ct,affine,pts,large)
    assert meta2["all_endpoints_inside_patch"]
    assert meta2["endpoint_patch_margin_mm"]>0


def test_fixed_lps_frame_is_stable_under_order_flip_and_small_jitter():
    # The historical canonical frame can rotate almost 180 degrees when the
    # x-coordinate order of two endpoints changes after a tiny perturbation.
    first=np.array([[-.1,0.,0.],[.1,5.,0.]],dtype=np.float64)
    jittered=np.array([[.1,0.,0.],[-.1,5.,0.]],dtype=np.float64)
    _,original=local_frame(first)
    _,perturbed=local_frame(jittered)
    canonical_angle=np.degrees(np.arccos(np.clip(
        (np.trace(original.T@perturbed)-1.)/2.,-1.,1.
    )))
    assert canonical_angle>170.
    _,fixed=local_frame(first,"fixed_lps")
    _,fixed_jittered=local_frame(jittered,"fixed_lps")
    assert np.array_equal(fixed,np.eye(3))
    assert np.array_equal(fixed_jittered,fixed)


def test_fixed_lps_patch_is_permutation_invariant_and_has_full_provenance():
    ct,affine,pts=synthetic_ct()
    cfg=PatchConfig(side=16,frame_mode="fixed_lps")
    original,audit=extract_candidate_patch(ct,affine,pts,cfg)
    shuffled,other=extract_candidate_patch(ct,affine,pts[[2,0,1]],cfg)
    assert np.array_equal(original,shuffled)
    assert audit["frame_mode"]=="fixed_lps"
    assert audit["orientation_det"]==pytest.approx(1.)
    assert other["frame_mode"]=="fixed_lps"
    with pytest.raises(ValueError,match="frame_mode"):
        PatchConfig(frame_mode="unexpected")
