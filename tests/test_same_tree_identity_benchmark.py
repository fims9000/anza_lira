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


def test_paired_ranking_loss_keeps_same_tree_pairs_in_minibatches():
    from benchmark_structural_tubes import fit
    import torch
    torch.set_num_threads(2)
    train=make_paired_trees(8,20261111)
    val=make_paired_trees(4,20261911)
    report=fit(
        "conv3d_sep_structure","PAIR",train[:3],val[:3],
        epochs=1,batch_size=4,seed=17,device="cpu",
        paired_validation=True,paired_ranking_weight=.5,
    )
    assert report["paired_ranking_weight"]==.5
    assert report["paired_validation"]["n_trees"]==4
    assert sum(report["paired_validation"][k]
               for k in ("positive_wins","negative_wins","ties"))==4
    assert np.isfinite(report["paired_validation"]["mean_correct_minus_wrong_score"])
    bad=(train[0],train[1].copy(),train[2])
    bad[1][1]=1
    with pytest.raises(ValueError,match="adjacent positive-negative"):
        fit(
            "conv3d_sep_structure","PAIR",bad,val[:3],
            epochs=1,batch_size=4,seed=17,device="cpu",
            paired_ranking_weight=.5,
        )


def test_synthetic_model_checkpoint_is_replayable_on_independent_test(tmp_path):
    import hashlib
    import torch
    from benchmark_structural_tubes import fit
    from models.ccta3d_candidate_encoder import Candidate3dEncoder
    torch.set_num_threads(2)
    train=make_paired_trees(4,20261111)
    valid=make_paired_trees(4,20261911)
    test=make_paired_trees(4,20261113)
    path=tmp_path/"pretrained_cnn_seed17.pt"
    m=fit(
        "conv3d_sep_structure","PAIR",train[:3],valid[:3],
        epochs=1,batch_size=4,seed=17,device="cpu",
        paired_validation=True,independent_test=test[:3],
        checkpoint_path=path,
    )
    assert path.is_file()
    assert m["checkpoint"]["verified_reload"] is True
    assert m["checkpoint"]["sha256"]==hashlib.sha256(path.read_bytes()).hexdigest()
    assert m["independent_test"]["n_trees"]==4
    assert m["independent_test"]["status"]=="independent_synthetic_test_not_clinical"
    scores=m["independent_test"]["score_true"]+m["independent_test"]["score_false"]
    assert all(0.<=p<=1. for p in scores)
    candidate=Candidate3dEncoder(
        "conv3d_sep_structure",channels=6,candidate_type="PAIR"
    )
    candidate.load_state_dict(torch.load(path,weights_only=True),strict=True)
    candidate.eval()
    with torch.no_grad():
        actual=torch.sigmoid(candidate(torch.from_numpy(test[0]))).numpy()
    assert np.allclose(actual[::2],m["independent_test"]["score_true"],atol=1e-6)
    assert np.allclose(actual[1::2],m["independent_test"]["score_false"],atol=1e-6)
    with pytest.raises(FileExistsError,match="never be overwritten"):
        fit(
            "conv3d_sep_structure","PAIR",train[:3],valid[:3],
            epochs=1,batch_size=4,seed=17,device="cpu",
            checkpoint_path=path,
        )
