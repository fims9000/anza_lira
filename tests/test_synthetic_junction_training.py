"""JUNCTION synthetic train/test provenance and checkpoint replay contracts."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.ndimage import label as cc_label

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from benchmark_synthetic_junction_training import synthetic_data
from benchmark_structural_tubes import fit


def test_synthetic_junction_three_arms_have_verified_26_connectivity():
    data,y,gt=synthetic_data(16,20261112,16,
                             same_histogram=False,endpoints_only=False)
    assert data.shape==(16,4,16,16,16)
    assert y.shape==(16,) and set(y.tolist())=={0.,1.}
    assert gt.shape==(16,1,16,16,16)
    assert np.isfinite(data).all()
    for i in range(len(y)):
        components,_=cc_label(gt[i,0]>.5,np.ones((3,3,3),dtype=np.uint8))
        centers=[np.unravel_index(data[i,j].argmax(),data.shape[2:]) for j in (1,2,3)]
        parts=[int(components[c]) for c in centers]
        assert (len(set(parts))==1 and parts[0]>0)==bool(y[i])


def test_endpoint_only_uses_identical_frozen_labels_and_erases_ct():
    full,y,gt=synthetic_data(16,20261112,16,
                             same_histogram=False,endpoints_only=False)
    tip,yy,gg=synthetic_data(16,20261112,16,
                             same_histogram=False,endpoints_only=True)
    assert np.array_equal(y,yy) and np.array_equal(gt,gg)
    assert np.array_equal(full[:,1:],tip[:,1:])
    assert not np.all(full[:,0]==0)
    assert np.all(tip[:,0]==0)


def test_junction_reloaded_checkpoint_independent_nonpaired_test(tmp_path):
    torch.set_num_threads(2)
    train=synthetic_data(16,20261112,16,
                         same_histogram=False,endpoints_only=False)
    valid=synthetic_data(16,20261912,16,
                         same_histogram=False,endpoints_only=False)
    test=synthetic_data(16,20261114,16,
                        same_histogram=False,endpoints_only=False)
    saved=tmp_path/"junction_model.pt"
    score=fit(
        "conv3d_sep_structure","JUNCTION",train,valid,
        epochs=1,batch_size=8,seed=17,device="cpu",
        independent_test=test,independent_paired_test=False,
        checkpoint_path=saved,
    )
    assert saved.exists()
    check=score["independent_test"]
    assert check["n_examples"]==16 and check["n_trees"] is None
    assert check["paired_correct"] is None
    assert check["status"]=="independent_synthetic_test_not_clinical"
    assert 0<=check["AUROC"]<=1
    assert score["checkpoint"]["verified_reload"] is True


def test_junction_does_not_misreport_nonpaired_truth_as_paired():
    test=synthetic_data(16,20261114,16,
                        same_histogram=False,endpoints_only=False)
    assert not np.array_equal(test[1],np.tile([1.,0.],8))
