"""V46 deterministic label-blind candidate augmentation contract tests.

Synthetic fixtures only; original Thebe images, expert TEST7 and actual
candidate endpoints are never loaded by unit tests.
"""
import numpy as np
import pytest
from scripts.tver_v46_integrated import (
    apply_frozen_dense_fallback,evidence_counts,MAX_PAIRS)


def fake_source():
    p=np.linspace(.2,1.,96*96,dtype=np.float32).reshape(96,96)
    return {
        "id":"section:0","section":17,"kind":"LINK",
        "proposals":[],"status":"DEGENERATE_DENSE_PREDICTED_MASK",
        "visible_prob":p,"gray":np.zeros((96,96),np.float32),
        "rectangle":(12,20,12,20),"true_ends":((10.,20.),(25.,20.)),
        "origin":(0,0),"source_count":0,"branch_count":0,
        "has_true_proposal":False,
    }


def fake_propose(p,gray,rect,tau):
    # Enforce a genuine rescue only with density below original 0.38 cap.
    if float(np.mean(p>=tau))>.38:
        raise AssertionError("Adaptive density threshold never applied")
    return {
        "proposals":[{"source":(10.,20.),"target":(25.,20.),
                     "features":[0.]*12,"geometry":-.7}],
        "source_count":1,"hard_branches":3,"status":"OK",
    }


def correct_pair(p,ends,origin):
    return p["source"]==ends[0] and p["target"]==ends[1]


def test_dense_correct_recovered_only_after_candidates_made():
    s=fake_source()
    results,meta=apply_frozen_dense_fallback([s],fake_propose,correct_pair)
    assert s["proposals"]==[]
    assert results[0]["has_true_proposal"]
    assert results[0]["proposals"][0]["y"]==1
    assert results[0]["source_count"]==1
    assert meta["new_true_candidate_LINK_rescues"]==1
    assert 0.65<=results[0]["adaptive_tau"]<=1.001


def test_nolink_remains_negative_and_non_dense_is_identity_preserved():
    link=fake_source()
    other=fake_source()
    other["kind"]="NO_LINK";other["id"]="section:0"
    fixed=fake_source()
    fixed["id"]="section:1";fixed["status"]="OK"
    fixed["proposals"]=[{"source":(5.,5.),"target":(4.,3.),
                          "features":[0.]*12,"geometry":-.5,"y":0}]
    fixed["has_true_proposal"]=False
    result,meta=apply_frozen_dense_fallback([link,other,fixed],fake_propose,correct_pair)
    assert result[1]["proposals"][0]["y"]==0
    assert result[2] is fixed
    assert meta["fallback_worlds"]==2
    assert meta["new_true_candidate_LINK_rescues"]==1


def test_fail_closed_on_leaked_labels_and_budget():
    s=fake_source()
    def leaky(*args):
        res=fake_propose(*args);res["proposals"][0]["y"]=1
        return res
    with pytest.raises(ValueError,match="leaked"):
        apply_frozen_dense_fallback([s],leaky,correct_pair)
    def over(*args):
        res=fake_propose(*args);res["proposals"]*=MAX_PAIRS+1
        return res
    with pytest.raises(RuntimeError,match="budget"):
        apply_frozen_dense_fallback([s],over,correct_pair)


def test_matched_world_ids_need_exact_same_cases():
    a=fake_source();a["kind"]="LINK"
    b=fake_source();b["kind"]="NO_LINK"
    summary=evidence_counts([a,b])
    assert summary["LINK_worlds"]==summary["synthetic_NOLINK_worlds"]==1
    assert summary["candidate_available_LINK"]==0
    b["id"]="different_gap"
    with pytest.raises(RuntimeError,match="identities"):
        evidence_counts([a,b])
