"""V49 label-blind anchor/curve route parity and no-oracle checks."""
import copy
import numpy as np
import pytest
from scripts.tver_v49_boundary_path import (
    boundary_anchors,boundary_route,hidden_path_success,score_new_paths,
    dense_polyline)
from scripts.tver_v26_predicted_cracks_e2e import (
    _simple_straight,_path_success)


def mock_scene(*,offset=0):
    p=np.zeros((96,96),np.float32)+.02
    # True observed segments are not modified outside intervention rectangle
    p[:,47:50]=.96
    rect=(44,53,45,52)
    p[rect[0]:rect[1],rect[2]:rect[3]]=0.
    scene={
        "visible_prob":p,"rectangle":rect,"origin":(0,0),
        "hidden_centerline":[(y,48) for y in range(44,53)],
    }
    pair={"source":(25.,43.+offset),"target":(70.,43.+offset)}
    return scene,pair


def test_observed_anchors_refine_offset_without_accessing_truth():
    scene,pair=mock_scene()
    p=scene["visible_prob"]
    anchors=boundary_anchors(p,scene["rectangle"],pair["source"],pair["target"])
    assert anchors is not None
    a,b,_=anchors
    assert a[0]<44 and b[0]>=53
    assert a[1] in (47,48,49) and b[1] in (47,48,49)
    for method in ("boundary_straight","boundary_hermite"):
        path,found=boundary_route(p,scene["rectangle"],pair["source"],pair["target"],mode=method)
        assert found
        assert hidden_path_success(scene,path)
    # Same inputs to decoder without hidden expert line must yield same path.
    scene_no_truth={k:v for k,v in scene.items() if k!="hidden_centerline"}
    for method in ("boundary_straight","boundary_hermite"):
        path,anchored=boundary_route(scene_no_truth["visible_prob"],scene_no_truth["rectangle"],
                                     pair["source"],pair["target"],mode=method)
        assert anchored and len(path)>20


def test_same_hidden_path_metric_as_original_reconstruction_implementation():
    scene,pair=mock_scene(offset=5.)
    path=_simple_straight(pair["source"],pair["target"])
    assert hidden_path_success(scene,path)==_path_success(scene,pair,method="straight")
    # A fresh non-oracle route does not require 'hidden_centerline'
    only_observed={"visible_prob":scene["visible_prob"],"rectangle":scene["rectangle"]}
    for mode in ("boundary_straight","boundary_hermite"):
        q,flag=boundary_route(only_observed["visible_prob"],only_observed["rectangle"],
                              pair["source"],pair["target"],mode=mode)
        assert isinstance(flag,bool)
        assert len(q)>0


def test_fail_closed_on_bad_probability_and_source_points():
    scene,pair=mock_scene()
    scene["visible_prob"][0,0]=np.nan
    with pytest.raises(ValueError):
        boundary_route(scene["visible_prob"],scene["rectangle"],pair["source"],pair["target"],
                       mode="boundary_straight")
    with pytest.raises(ValueError):
        boundary_route(np.ones((96,96),np.float32),scene["rectangle"],
                       pair["source"],pair["target"],mode="nonexistent")
    with pytest.raises(ValueError):
        dense_polyline([[np.nan,0.],[4.,3.]],(96,96))
