"""Patient-safe scene-level PAIR/JUNCTION identity metric for 3D ANZA/CNN."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_an_train import candidate_scene_ranking_metrics


def fixture():
    truth=np.array([0.,1.,0.,0.,0.,1.,0.])
    scores=np.array([.1,.9,.5,.8,.6,.1,.8])
    examples=[
        {"scene_id":"A","patient_id":953,"candidate_id":f"A_{i}"}
        for i in range(3)
    ]+[
        {"scene_id":"B","patient_id":953,"candidate_id":f"B_{i}"}
        for i in range(2)
    ]+[
        {"scene_id":"C","patient_id":957,"candidate_id":f"C_{i}"}
        for i in range(2)
    ]
    return truth,scores,examples


def test_ranking_counts_negative_scenes_and_wrong_identities():
    y,score,examples=fixture()
    metric=candidate_scene_ranking_metrics(y,score,examples)
    assert metric["total_scenes"]==3
    assert metric["positive_scenes"]==2
    assert metric["none_scenes"]==1
    assert metric["top1_correct"]==1
    assert metric["top1_accuracy"]==pytest.approx(.5)
    assert metric["per_patient"]["953"]["positive_scenes"]==1
    assert metric["per_patient"]["953"]["none_scenes"]==1
    assert metric["per_patient"]["957"]["top1_accuracy"]==0.


def test_candidate_id_tie_break_is_stable_and_not_label_based():
    y=np.array([1.,0.])
    s=np.array([.5,.5])
    examples=[
        {"scene_id":"S","patient_id":953,"candidate_id":"Z"},
        {"scene_id":"S","patient_id":953,"candidate_id":"A"},
    ]
    assert candidate_scene_ranking_metrics(y,s,examples)["top1_correct"]==0
    assert candidate_scene_ranking_metrics(
        y[::-1],s[::-1],examples[::-1]
    )["top1_correct"]==0


def test_two_true_candidates_duplicate_ids_and_nonfinite_scores_fail():
    y,s,examples=fixture()
    twice=y.copy();twice[0]=1.
    with pytest.raises(ValueError,match="more than one true"):
        candidate_scene_ranking_metrics(twice,s,examples)
    duplicate=[dict(x) for x in examples]
    duplicate[1]["candidate_id"]=duplicate[0]["candidate_id"]
    with pytest.raises(ValueError,match="Duplicate"):
        candidate_scene_ranking_metrics(y,s,duplicate)
    nonfinite=s.copy();nonfinite[0]=np.nan
    with pytest.raises(ValueError,match="Invalid rank"):
        candidate_scene_ranking_metrics(y,nonfinite,examples)
