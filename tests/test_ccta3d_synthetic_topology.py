"""Synthetic tube labels must agree with actual 3D vessel connectivity.

Regression: historical JUNCTION negative generator emitted a 26-connected GT
vessel for many y=0 examples, making topology supervision contradictory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.ndimage import label as cc_label

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts/research/ccta_graph_lira_safe_repair"))
from benchmark_structural_tubes import make_phantoms
from models.structural_bridge3d import _peak_centers


@pytest.mark.parametrize("kind",["PAIR","JUNCTION"])
@pytest.mark.parametrize("seed",[20261111,20261911])
def test_synthetic_reference_components_agree_with_binary_labels(kind,seed):
    x,y,masks=make_phantoms(32,seed,kind,side=12)
    assert x.shape==(32,2,12,12,12)
    assert masks.shape==(32,1,12,12,12)
    assert y.sum()==16
    endpoints,valid=_peak_centers(torch.from_numpy(x[:,1]),2 if kind=="PAIR" else 3)
    assert valid.all()
    for i,expected in enumerate(y.astype(bool)):
        components,_=cc_label(masks[i,0]>.5,np.ones((3,3,3),dtype=np.uint8))
        positions=endpoints[i].numpy().round().astype(int)
        labels=[int(components[tuple(p)]) for p in positions]
        connected=all(v>0 and v==labels[0] for v in labels)
        assert connected==expected,(kind,seed,i,labels,expected)


def test_positive_and_negative_have_same_input_heatmap_protocol():
    x,y,_=make_phantoms(24,20261111,"JUNCTION")
    assert np.isfinite(x).all()
    assert x[:,1].min()>=0 and x[:,1].max()<=1
    # Heatmap encodes candidate endpoint positions, not the class label.
    positives=x[y==1,1]
    negatives=x[y==0,1]
    assert positives.shape==negatives.shape
    assert (positives.sum(axis=(1,2,3))>0).all()
    assert (negatives.sum(axis=(1,2,3))>0).all()


def test_permuted_hu_control_preserves_intensity_histogram_and_endpoint_heatmap():
    from benchmark_structural_tubes import permute_hu_voxels
    x,_,_=make_phantoms(10,20262001,"PAIR")
    modified=permute_hu_voxels(x,seed=12345)
    assert np.array_equal(modified[:,1],x[:,1])
    assert np.array_equal(np.sort(modified[:,0].reshape(10,-1),axis=1),
                          np.sort(x[:,0].reshape(10,-1),axis=1))
    assert not np.array_equal(modified[:,0],x[:,0])
    assert np.array_equal(modified,permute_hu_voxels(x,seed=12345))


def test_single_scalar_shortcut_orientation_frozen_using_train_only():
    from benchmark_structural_tubes import simple_shortcut_audit
    train=make_phantoms(32,20261111,"JUNCTION")
    valid=make_phantoms(24,20261911,"JUNCTION")
    controls=simple_shortcut_audit(train,valid)
    assert "hu_mean" in controls and "heatmap_mean" in controls
    for values in controls.values():
        assert values["orientation_frozen_from_train"] in (-1,1)
        assert 0<=values["holdout_AUROC"]<=1
        assert 0<=values["holdout_AUPRC"]<=1


def test_rank_matched_hu_histograms_eliminate_scalar_shortcuts_without_changing_geometry():
    from benchmark_structural_tubes import normalize_synthetic_hu_histograms
    x,y,mask=make_phantoms(12,20261111,"JUNCTION")
    remapped=normalize_synthetic_hu_histograms(x)
    assert np.array_equal(remapped[:,1],x[:,1])
    assert np.array_equal(np.sort(remapped[0,0].flatten()),
                          np.sort(remapped[1,0].flatten()))
    assert np.array_equal(np.sort(remapped[1,0].flatten()),
                          np.sort(remapped[-1,0].flatten()))
    assert np.unique(remapped[0,0]).size==remapped[0,0].size
    assert not np.array_equal(x[:,0],remapped[:,0])
    # Rank ordering within each patch preserves local bright-tube geometry.
    for i in range(len(y)):
        assert np.array_equal(np.argsort(x[i,0].ravel(),kind="stable"),
                              np.argsort(remapped[i,0].ravel(),kind="stable"))
    assert np.all((remapped[:,0]>=-1)&(remapped[:,0]<=1))
    with pytest.raises(ValueError,match="candidate patch"):
        normalize_synthetic_hu_histograms(x[:,0])
