"""Paired Y-tree study: no aggregate CT or mask shortcut between labels."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import label as cc_label

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from benchmark_same_tree_identity import make_paired_trees


@pytest.mark.parametrize("seed",[20261111,20261911])
@pytest.mark.parametrize("rank",[False,True])
def test_paired_y_trees_share_ct_gt_endpoint_distance_and_component(seed,rank):
    x,y,targets,metadata=make_paired_trees(
        8,seed,side=24,same_histogram=rank
    )
    assert x.shape==(16,4,24,24,24)
    assert targets.shape==(16,1,24,24,24)
    assert y.tolist()==[1.,0.]*8
    assert metadata["n_independent_trees"]==8
    assert metadata["maximum_positive_negative_endpoint_distance_difference_vox"]<1e-4
    assert np.array_equal(x[::2,0],x[1::2,0])
    assert np.array_equal(targets[::2],targets[1::2])
    assert np.count_nonzero(x[:,3])==0
    for i in range(0,len(x),2):
        components,_=cc_label(targets[i,0]>.5,np.ones((3,3,3),dtype=np.uint8))
        end=np.stack([x[i,1],x[i,2],x[i+1,2]])
        pts=np.array([np.unravel_index(x.argmax(),x.shape) for x in end])
        ids=[int(components[tuple(p)]) for p in pts]
        assert ids[0]>0 and len(set(ids))==1


def test_paired_y_tree_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        make_paired_trees(2,20261111)
    with pytest.raises(ValueError):
        make_paired_trees(8,20261111,angle_low_deg=80,angle_high_deg=30)
