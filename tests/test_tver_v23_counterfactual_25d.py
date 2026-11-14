"""V23 synthetic counterfactual three-way link tests, no real geology claims."""
from __future__ import annotations

import copy
import numpy as np

from scripts.tver_v23_25d_counterfactual_synthetic import (
    N_BASE,SEED_BASE,POLICIES,NO_LINK,
    render_triplet,visible_features,_semantic_pair_audit,
    _score_table,_calibrate,_predict,_metric,_paired_base_bootstrap,
)


def small_rows(split="train",n=8):
    rows=[]
    for base in range(n):
        for c in render_triplet(split,base):
            f=visible_features(c)
            rows.append({
                "base":c["base"],"label":c["label"],
                "central_sha":__import__("hashlib").sha256(
                    c["central"].tobytes()).hexdigest(),
                "center_features":f["center_features"],
                "all_features":f["all_features"],
                "scores":{k:f[k] for k in (
                    "geometry","central","adjacent_straight",
                    "far_sham","continuity_25d")},
            })
    return rows


def test_v23_three_worlds_have_bitwise_identical_central_pixels_and_geometry():
    triplet=render_triplet("train",3)
    assert {x["label"] for x in triplet}=={0,1,2}
    assert len({x["flip"] for x in triplet})==1
    assert all(x["source"]==triplet[0]["source"] for x in triplet)
    assert all(x["candidates"]==triplet[0]["candidates"] for x in triplet)
    assert all(np.array_equal(x["central"],triplet[0]["central"])
               for x in triplet)
    assert all(np.array_equal(x["far"],triplet[0]["far"]) for x in triplet)
    # Real adjacent layers differ by latent world; 2D alone cannot see it.
    assert not np.array_equal(
        triplet[0]["adjacent"],triplet[1]["adjacent"])
    assert not np.array_equal(
        triplet[0]["adjacent"],triplet[2]["adjacent"])


def test_v23_2d_predictions_cannot_beat_one_third_on_matched_worlds():
    rows=small_rows("cal",9)
    audit=_semantic_pair_audit(rows)
    assert audit["independent_base_groups"]==9
    for policy in ("geometry","central"):
        score=np.stack([r["scores"][policy] for r in rows])
        for threshold in (
            {"accept_threshold":-1e9,"margin_threshold":0.},
            {"accept_threshold":1e9,"margin_threshold":0.},
        ):
            predicted=_predict(score,threshold)
            assert _metric(rows,predicted)["all_3way_accuracy"]==1/3


def test_v23_visible_feature_extractor_does_not_observe_hidden_label():
    c=render_triplet("train",1)[0]
    f=visible_features(c)
    altered=copy.deepcopy(c)
    altered["label"]=2
    altered["world"]=999
    altered["base"]=-1
    g=visible_features(altered)
    for k in f:
        np.testing.assert_array_equal(f[k],g[k])


def test_no_link_world_is_truly_different_from_link_adjacent():
    for split in ("train","ood"):
        for ix in (0,1,3,9):
            a,b,n=render_triplet(split,ix)
            assert n["label"]==2
            assert not np.array_equal(a["adjacent"],n["adjacent"])
            assert not np.array_equal(b["adjacent"],n["adjacent"])
            assert np.array_equal(a["central"],n["central"])
            assert np.array_equal(b["central"],n["central"])


def test_per_base_group_bootstrap_not_world_or_pixel_pseudoreplication():
    rows=small_rows("train",6)
    # Source values are group-correlated, not 18 independent test scenes.
    truth=np.asarray([x["label"] for x in rows],int)
    good=truth.copy()
    bad=np.full(len(rows),NO_LINK)
    summary=_paired_base_bootstrap(rows,good,bad)
    assert summary["N_independent_base_groups"]==6
    assert summary["bootstrap_reps"]==3000
    assert np.isclose(summary["delta_3way_accuracy"],2/3,atol=1e-12)


def test_calibration_uses_only_calibration_scores_and_respects_risk_budget():
    rows=small_rows("cal",14)
    y=np.asarray([r["label"] for r in rows],int)
    # Class-conditional artificially easy finite scores in a controlled
    # test, not real V23 model performance. The positive acceptance gate
    # must not count a wrong candidate or NO_LINK as correct.
    scores=np.full((len(rows),2),-5.)
    for i,label in enumerate(y):
        if label in (0,1):scores[i,label]=5.
    threshold=_calibrate(y,scores)
    pred=_predict(scores,threshold)
    result=_metric(rows,pred)
    assert result["correct_accepted"]==28
    assert result["wrong_branch_accepted"]==0
    assert result["false_accepted"]==0


def test_v23_frozen_bases_disjoint_across_train_cal_iid_ood():
    sets={
        split:set(range(seed,seed+N_BASE[split]))
        for split,seed in SEED_BASE.items()}
    for a in sets:
        for b in sets:
            if a!=b:assert sets[a].isdisjoint(sets[b])
