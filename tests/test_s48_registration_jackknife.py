"""Jackknife regression and safety tests. No native physics GT claimed."""
import cv2
import numpy as np
import pytest

from bubble_chamber.s48_registration_jackknife import (
    leave_one_fiducial_out, photo_transforms, shifted_path_for_registration,
    ridge_binary, ridge_path_support, assess_candidate, annotate_graph)
from bubble_chamber.s44_crossframe_persistence import ridge_near_path
from bubble_chamber.s39_machine_auto import ridge_evidence


def _labels():
    positions = np.asarray([(1400, 1500), (1900, 1500), (1400, 2000),
                            (1900, 2000), (1600, 2450), (2100, 2500)], dtype=float)
    out = {}
    for tag, dx, dy in (("005", 0, 0), ("006", 17, -9)):
        out[f"H50Z_030_01_{tag}.bmp"] = {
            "keypoints": [{"label": f"f{i+1}", "x_px":float(x+dx),"y_px":float(y+dy)}
                          for i, (x, y) in enumerate(positions)]}
    return out


def _ev(a=1., b=1., control=0., fid=False):
    return {"draft_id": "a", "source_multiscale_ridge_coverage": a,
            "counterpart_same_registered_path_coverage": b,
            "displaced_same_shape_counterpart_control_coverage": control,
            "known_fiducial_uncertainty_NOT_NATIVE_TRUTH": fid}


def _path(y=80):
    return np.column_stack([np.linspace(30, 160, 70), np.full(70, y)])


def test_loo_uses_six_copies_of_same_family_and_no_pixel_training():
    r = leave_one_fiducial_out(_labels())
    assert len(r["loo_forward_PRIVATE"]) == 6
    assert r["heldout_max_px"] < .01
    assert not r["physical_3d_calibration_performed"]
    for src in ("005", "006"):
        full, alt = photo_transforms(r, src)
        assert len(alt) == 6
        p = shifted_path_for_registration(_path(), [200, 300], full, alt[0])
        assert np.allclose(p, _path(), atol=.02)
    with pytest.raises(ValueError):photo_transforms(r,"007")
    broken = _labels()
    broken["H50Z_030_01_006.bmp"]["keypoints"].pop()
    with pytest.raises(ValueError):leave_one_fiducial_out(broken)


def test_spatial_shift_uses_absolute_global_coords_and_inverse():
    full = np.array([[1,0,10],[0,1,-15]], dtype=float)
    alt = np.array([[1,0,14],[0,1,-9]], dtype=float)
    p = _path()
    shifted = shifted_path_for_registration(p,[1990,2020],full,alt)
    assert np.allclose(shifted, p + [4,6])
    bad = full.copy();bad[0,0]=0;bad[1,1]=0
    with pytest.raises(ValueError):shifted_path_for_registration(p,[0,0],bad,alt)
    with pytest.raises(ValueError):shifted_path_for_registration(np.ones((2,2)),[0,0],full,alt)


def test_cached_ridge_support_agrees_with_S44_and_border_is_not_clipped():
    im = np.full((192,192), 210, dtype=np.uint8)
    cv2.line(im,(30,80),(160,80),15,2)
    cached = ridge_binary(im)
    p = _path()
    reference = ridge_near_path(ridge_evidence(im)["score"],p)
    assert abs(ridge_path_support(cached,p)-reference) < 1e-7
    assert ridge_path_support(cached,p+200) is None
    with pytest.raises(ValueError):ridge_binary(im.astype("float32"))


def test_uncertainty_only_no_truth_or_auto_join_and_fiducial_guard():
    a = np.full((192,192), 210, dtype=np.uint8)
    cv2.line(a,(30,80),(160,80),15,2)
    p = _path()
    original = {"draft_id":"a","points_xy":p.tolist()}
    m = np.array([[1.,0.,0.],[0.,1.,0.]])
    variants = [m.copy() for _ in range(6)]
    d = assess_candidate(a,a,original,_ev(),[100,100],m,variants)
    assert d["S45_original_repeat_risk"]
    assert d["jackknife_variant_repeat_hits"]==6
    assert d["registration_status"]=="BASELINE_REPEAT_STABLE_ACROSS_POSES"
    assert not d["physical_auto_join_authorized"]
    assert not d["allowed_for_heldout_training"]
    e = assess_candidate(a,a,original,_ev(fid=True),[100,100],m,variants)
    assert e["registration_status"]=="FIDUCIAL_GUARD_IS_SEPARATE"
    assert not e["new_pose_only_warning"]
    with pytest.raises(ValueError):assess_candidate(a,a,original,_ev(),[0,0],m,variants[:2])


def test_shifted_repeat_can_be_an_ambiguous_pose_only_warning():
    im = np.full((192,192), 210, dtype=np.uint8)
    target = im.copy()
    cv2.line(im,(30,80),(160,80),15,2)
    cv2.line(target,(30,86),(160,86),15,2)
    m = np.array([[1.,0.,0.],[0.,1.,0.]])
    alt = np.array([[1.,0.,0.],[0.,1.,6.]])
    evidence = _ev(a=1.,b=.0,control=0.)
    d = assess_candidate(im,target,{"draft_id":"a", "points_xy":_path().tolist()},evidence,
                         [0,0],m,[alt for _ in range(6)])
    assert d["registration_status"]=="POSE_ONLY_POSSIBLE_REPEAT_AMBIGUOUS"
    assert d["new_pose_only_warning"]
    assert d["jackknife_variant_repeat_hits"]==6
    assert d["native_track_truth"]=="UNKNOWN"


def test_graph_does_not_drop_edges_or_relabel_unflagged_as_safe():
    source = {"nodes":[{"source_candidate_ids":["a"]}, {"source_candidate_ids":["b"]}],
              "candidate_edges":[{"i":0,"j":1,"physics_join_authorized":False}]}
    s40 = {"candidate_edges":[{"i":0,"j":1,"fiducial_overlap":False}]}
    d = {"a":{"S45_original_repeat_risk":False,"new_pose_only_warning":True},
         "b":{"S45_original_repeat_risk":False,"new_pose_only_warning":False}}
    r=annotate_graph(source,s40,d)
    assert r["counters"]["union_any"]==1
    assert r["counters"]["S48_additional_pose_only_risk"]==1
    assert r["counters"]["physics_auto_joins"]==0
    assert not r["per_edge_PRIVATE"][0]["physical_auto_join_authorized"]
    with pytest.raises(ValueError):annotate_graph(source,s40,{"a":d["a"]})
    with pytest.raises(ValueError):annotate_graph(source,{"candidate_edges":[]},d)
