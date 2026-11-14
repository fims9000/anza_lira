"""Tver V30: P0 genuine context input and calibrated refuse contracts.

All fixtures are synthetic and do not expose previously inspected F3 TEST.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from scripts.tver_v30_p0_context import (
    observed_pair_crop,paired_training_examples,fit_native_p0,
    context_scores,SEEDS,EPOCHS)
from scripts.tver_v30_real_p0_e2e import (
    _calibration,_decision,CAL_THRESHOLDS,MODEL_NAMES)


def _case(section,kind,features):
    rows=[]
    for i,(y,point) in enumerate(features):
        rows.append({
            "y":y,
            "source":(23.,24.),
            "target":(float(point),float(36.+i)),
            "features":(np.arange(12,dtype=float)*.1+i).tolist(),
            "geometry":-float(i+1),
        })
    rgb=np.zeros((96,96,3),np.float32)
    for k in range(3):
        rgb[...,k]=np.linspace(.1,.9,96,dtype=np.float32)[None,:]
    visible=np.zeros((96,96),np.float32)
    visible[20:30,20:31]=.95
    return {
        "section":section,"id":f"test:{section}",
        "kind":kind,"proposals":rows,
        "original_rgb":rgb,"visible_prob":visible,
        "source_count":1,"branch_count":2,"status":"OK",
        "has_true_proposal":any(p["y"] for p in rows),
        "true_ends":((9.,9.),(89.,89.)),  # must NEVER be read by crop
        "hidden_centerline":[(i,i) for i in range(18,40)],
    }


def test_observed_pair_context_is_real_original_P0_shape():
    s=_case(10,"LINK",[(1,50.),(0,60.)])
    q=observed_pair_crop(s,s["proposals"][0],.9)
    assert q.shape==(6,33,49)
    assert np.isfinite(q).all()
    # Visible P0 context has genuine predicted mask, and nonzero marker.
    assert float(q[4].max())>.01
    assert (q[3]>=0).all()


def test_no_leakage_of_expert_ends_or_target_label_into_P0_input():
    s=_case(10,"LINK",[(1,50.),(0,60.)])
    a=observed_pair_crop(s,s["proposals"][0],.9)
    t=copy.deepcopy(s)
    t["true_ends"]=((93.,93.),(1.,1.))
    t["hidden_centerline"]=[(1,1)]
    t["proposals"][0]["y"]=0
    t["section"]=999
    b=observed_pair_crop(t,t["proposals"][0],.9)
    np.testing.assert_array_equal(a,b)


def test_true_P0_architecture_five_convolutions():
    from path_completion.pair_classifier import EndpointPairClassifier
    net=EndpointPairClassifier()
    conv=[m for m in net.modules() if isinstance(m,torch.nn.Conv2d)]
    assert len(conv)==5
    with torch.no_grad():
        a=net(torch.randn(2,6,33,49))
    assert a.shape==(2,)
    assert torch.isfinite(a).all()


def test_no_pos_train_examples_must_hard_stop():
    source=[_case(10,"LINK",[(0,40.),(0,50.)])]
    try:
        paired_training_examples(source,.9)
        assert False,"No GT positives should STOP"
    except RuntimeError as e:
        assert "insufficient" in str(e).lower()


def test_balanced_unique_negative_training_source():
    source=[]
    for section in range(10,19):
        source.append(_case(section,"LINK",[(1,40.),(0,50.),(0,60.)]))
    im,y,features,passport=paired_training_examples(source,.9)
    assert im.shape==(18,6,33,49)
    assert y.sum()==9 and (1-y).sum()==9
    assert features.shape==(18,12)
    assert passport["positive_unique_predicted_train_pairs"]==9
    assert passport["matched_negative_unique_predicted_train_pairs"]==9


def test_strict_CAL_false_link_refusal_even_with_high_rank_probability():
    positive=_case(43,"LINK",[(1,44.)])
    negative=_case(43,"NO_LINK",[(0,45.)])
    cal=[positive,negative,positive,negative]
    strong=[np.asarray([.9]),np.asarray([.85]),
            np.asarray([.9]),np.asarray([.85])]
    receipt=_calibration(cal,strong)
    assert receipt["status"]=="CAL_FROZEN"
    # CAL negative false acceptance budget 10% of 2 => zero false allowed.
    assert receipt["n_false_accept_CAL"]==0
    assert receipt["selected_threshold"]>.85


def test_no_candidates_does_not_become_fake_success():
    empty=_case(10,"LINK",[])
    r=_decision(empty,np.empty((0,)),.4)
    assert not r["accepted"] and not r["correct"]
    assert r["top1_score"] is None


def test_registered_budget_and_policies():
    assert SEEDS==(11,23)
    assert EPOCHS==30
    assert len(MODEL_NAMES)==5
    assert CAL_THRESHOLDS[0]==0.0
    assert CAL_THRESHOLDS[-1]==1.01
