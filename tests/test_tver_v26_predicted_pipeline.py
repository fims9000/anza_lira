"""V26 real-source-only endpoint and selective LIRA contracts (synthetic tests).

The test scene is fabricated solely for unit tests; it is never reported
as real F3 ANZA-LIRA model performance.
"""
from __future__ import annotations

import inspect

import numpy as np

from scripts.cracks_real_oracle_gap_audit import COLORS
from scripts.tver_v26_predicted_cracks_e2e import (
    ROI,_expert_gaps,_world,_roi,sbpp_proposals,
    is_true_relation,_features,_calibrate,_case_prediction,
)


def _fake_long_expert():
    mask=np.zeros((140,224,3),dtype=np.uint8)
    mask[:]=COLORS["white"]
    mask[61,20:180]=COLORS["blue"]
    return mask


def test_ground_truth_expert_mask_only_creates_intervention_never_source_ports():
    mask=_fake_long_expert()
    gaps=_expert_gaps(mask,34)
    assert gaps
    scene=gaps[0]
    assert scene["section"]==34
    assert len(scene["hidden_centerline"])==12
    assert len(scene["true_ends"])==2
    assert "true_ends" not in inspect.signature(sbpp_proposals).parameters
    assert "hidden_centerline" not in inspect.signature(sbpp_proposals).parameters
    assert "labels" not in inspect.signature(_features).parameters
    image=np.zeros((140,224,3),np.float32)
    roi,origin,local=_roi(image,scene["rectangle"])
    assert roi.shape==(ROI,ROI,3)
    assert all(0<=p<ROI for p in local)


def test_model_only_mask_and_candidate_absence_are_distinct_controlled_worlds():
    gap=_expert_gaps(_fake_long_expert(),34)[0]
    image=np.zeros((140,224,3),dtype=np.float32)
    crop,origin,rect=_roi(image,gap["rectangle"])
    assert crop.shape==(ROI,ROI,3)
    prob=np.full((ROI,ROI),.1,np.float32)
    prob[ROI//2,10:ROI-10]=.95
    yes=_world(prob,rect,gap["true_ends"],origin,"LINK")
    no=_world(prob,rect,gap["true_ends"],origin,"NO_LINK")
    assert np.all(yes[rect[0]:rect[1],rect[2]:rect[3]]==0)
    assert np.all(no[rect[0]:rect[1],rect[2]:rect[3]]==0)
    assert np.all(no<=yes)
    assert int(np.sum(yes!=no))>0


def test_sbpp_predicted_not_expert_hard_soft_input_contract():
    x=np.zeros((ROI,ROI),np.float32)
    # Two thin predicted fragments with a missing midline:
    # only model probabilities and visible image are supplied.
    x[48,12:45]=.93
    x[48,53:86]=.91
    image=np.zeros_like(x)
    image[48,12:86]=.8
    result=sbpp_proposals(
        x,image,(45,52,44,54),hard_threshold=.5)
    assert result["status"] in {
        "OK","NO_SOURCES_OR_CANDIDATES",
        "NO_PREDICTED_FAULT_PIXELS",
        "DEGENERATE_DENSE_PREDICTED_MASK",
        "TOO_MANY_PREDICTED_BRANCHES"}
    assert result["source_count"]>=0
    for item in result["proposals"]:
        assert set(item)=={"source","target","features","geometry"}
        assert len(item["features"])==12
        assert np.isfinite(item["features"]).all()
        assert "expert" not in item
        assert "label" not in item
        assert "y" not in item


def test_expert_link_label_is_used_only_after_predicted_candidate_generation():
    proposal={"source":(48.,42.),"target":(48.,57.)}
    ends=((48.,42.),(48.,57.))
    assert is_true_relation(proposal,ends,(0,0))
    assert is_true_relation(
        {"source":(48.,57.),"target":(48.,42.)},ends,(0,0))
    assert not is_true_relation(
        {"source":(14.,10.),"target":(48.,57.)},ends,(0,0))


def test_CAL_only_gate_can_reject_all_when_false_links_unavoidable():
    proposal={"features":[0.]*12,"geometry":-10.,"y":0}
    cal=[
        {"kind":"LINK","proposals":[proposal],"status":"OK"},
        {"kind":"LINK","proposals":[proposal],"status":"OK"},
        {"kind":"NO_LINK","proposals":[proposal],"status":"OK"},
        {"kind":"NO_LINK","proposals":[proposal],"status":"OK"},
    ]
    # Identical known-false candidates cannot meet an empirical 10% cap
    # and so should be rejected at the frozen CAL threshold.
    r=_calibrate(cal,"geometry",None)
    assert r["status"]=="CAL_FROZEN"
    assert r["n_false_accept_CAL"]==0
    assert all(not _case_prediction(
        s,"geometry",None,r["accept_threshold"])["accepted"]
        for s in cal)
