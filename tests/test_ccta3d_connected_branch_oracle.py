"""Oracle negative-control dataset: same connected GT with wrong continuation."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from audit_connected_branch_oracle import generate
from models.structural_bridge3d import VesselBridgeReadout


def test_same_connected_gt_contains_directionally_wrong_pair_candidates():
    image,tips,y,connected=generate(n=4,seed=20261111)
    assert image.shape==(8,24,24,24)
    assert tips.shape==(8,2,24,24,24)
    assert y.tolist()==[1,0,1,0,1,0,1,0]
    assert connected.all()
    for j in range(0,len(image),2):
        assert np.array_equal(image[j],image[j+1])
        # The two alternative relations have different endpoint identity;
        # a topology-only component-label baseline would predict both true.
        assert not np.array_equal(tips[j],tips[j+1])


def test_branch_specific_oracle_does_not_confuse_undirected_connectedness():
    image,tips,y,connected=generate(n=4,seed=20261111)
    with torch.no_grad():
        scores=VesselBridgeReadout(2)(
            torch.from_numpy(image),torch.from_numpy(tips)
        ).numpy()
    assert np.isfinite(scores).all()
    assert connected.all()
    assert scores.shape==y.shape
    # Do NOT assert that 5 hardcoded straight paths always classify 3D
    # branch identity: that is exactly the scientific question under test.


def test_shallow_in_tree_false_branch_can_pass_path_085_oracle_despite_same_tree():
    """Regression of structural failure: 25° wrong branch is nearly collinear."""
    from audit_connected_branch_oracle import BRANCH_ANGLES_DEGREES
    vessel,tips,y,same=generate(
        n=len(BRANCH_ANGLES_DEGREES),seed=20261111
    )
    assert same.all()
    # i=2 -> 25 degrees, candidate indices 4 true, 5 false.
    i=BRANCH_ANGLES_DEGREES.index(25)
    with torch.no_grad():
        support=VesselBridgeReadout(2)(
            torch.from_numpy(vessel[2*i:2*i+2]),
            torch.from_numpy(tips[2*i:2*i+2]),
        ).numpy()
    assert y[2*i:2*i+2].tolist()==[1,0]
    # Ground truth topology is identical for the two candidate endpoints.
    assert np.array_equal(vessel[2*i],vessel[2*i+1])
    assert support[0]>.9
    assert support[1]>.85,(
        "If this changes, report improved oracle branch specificity instead "
        "of silently deleting the adversarial case"
    )
