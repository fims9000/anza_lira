"""V33 external Thebe TRAIN-only scaling, line intervention, and graph port contracts.

No real TEST data, no large Thebe arrays in fixture.
"""
from __future__ import annotations

import numpy as np

from scripts.tver_v33_thebe_train_segmenters import (
    TRAIN_STEPS,PARAM_WIDTHS,TOPOLOGY_THRESHOLDS,
    _gaps,_rgb,_scaled_gray,_threshold_result,
)


def test_equal_cost_registered_thebe_segmenters():
    assert TRAIN_STEPS==600
    assert PARAM_WIDTHS==(4,8,12,16)
    assert TOPOLOGY_THRESHOLDS==(0.35,0.4,0.45,0.5,0.55,0.6,
                                0.65,0.7,0.75,0.8,0.85,0.9)


def test_observed_amplitude_is_normalized_from_frozen_train_stats():
    scale={
        "TRAIN_only_1pct":-2.0,"TRAIN_only_99pct":2.0,
        "normalized_channel_mean":.5,"normalized_channel_std":.2,
    }
    a=np.array([[-4.,-2.,0.],[1.,2.,9.]],np.float32)
    g=_scaled_gray(a,scale)
    np.testing.assert_allclose(g,[[0,0,.5],[.75,1,1]])
    rgb=_rgb(a,scale)
    assert rgb.shape==(2,3,3)
    np.testing.assert_array_equal(rgb[...,0],g)
    np.testing.assert_array_equal(rgb[...,1],g)
    np.testing.assert_array_equal(rgb[...,2],g)


def test_expert_trace_is_only_an_intervention_schedule():
    mask=np.zeros((255,701),bool)
    mask[100,100:280]=True
    gaps=_gaps(mask,"TRAIN_FIXTURE")
    assert gaps
    gap=gaps[0]
    assert gap["id"].startswith("thebe:TRAIN_FIXTURE:")
    assert len(gap["hidden_centerline"])==12
    assert len(gap["true_ends"])==2
    assert gap["rectangle"][0]<=100<gap["rectangle"][1]
    assert gap["rectangle"][2]>100


def test_no_gap_when_image_contains_no_confident_line():
    mask=np.zeros((255,701),bool)
    mask[100,100:130]=True
    assert _gaps(mask,"VALID_FIXTURE")==[]


def test_true_candidate_only_from_inference_and_expert_posthoc_scoring(monkeypatch):
    import scripts.tver_v33_thebe_train_segmenters as mod
    seen=[]
    def fake_sbpp(visible,gray,rect,thresh):
        seen.append((float(thresh),float(visible.mean()),tuple(rect)))
        return {
            "status":"OK","source_count":1,"hard_branches":2,
            "proposals":[{"source":(9.,9.),"target":(28.,28.)}]}
    def fake_relation(p,ends,origin):
        assert ends==((1.,1.),(3.,3.))
        return True
    monkeypatch.setattr(mod,"sbpp_proposals",fake_sbpp)
    monkeypatch.setattr(mod,"is_true_relation",fake_relation)
    p=np.ones((96,96),np.float32)*.8
    cases=[{"id":"A","prob":p,"gray":p,"rect":(12,22,12,22),
            "ends":((1.,1.),(3.,3.)),"origin":(0,0)}]
    receipt=_threshold_result(cases,.65)
    assert receipt["n_true_pair_gaps"]==1
    assert receipt["n_true_pairs"]==1
    assert receipt["n_false_pairs"]==0
    assert seen and seen[0][0]==.65
    assert seen[0][1]<.8  # intervention affects predicted mask
