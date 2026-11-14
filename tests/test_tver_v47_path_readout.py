"""V47 frozen straight/maximin path scoring and source-risk matching tests.

Fixtures are synthetic. The 120 real Thebe source cases are evaluated only
by the read-only official GitHub Action from original TRAIN/VAL source files.
"""
import numpy as np
import pytest
from scripts.tver_v47_full_path_readout import audited_path_readout,CAL_THRESHOLDS
from scripts.tver_v26_predicted_cracks_e2e import _path_success
import scripts.tver_v47_full_path_readout as study
import scripts.tver_v26_predicted_cracks_e2e as path_module


def path_fixture():
    rect=(44,52,42,48)
    return {
        "visible_prob":np.full((96,96),.55,np.float32),
        "gray":np.zeros((96,96),np.float32),
        "rectangle":rect,"origin":(0,0),
        "hidden_centerline":[(48,x) for x in range(42,48)],
        "section":40,"id":"40:0","kind":"LINK",
    }


def test_actual_repos_straight_path_scoring_with_true_hidden_centerline():
    x=path_fixture()
    straight={"source":(48.,30.),"target":(48.,60.)}
    assert _path_success(x,straight,method="straight") is True
    wrong={"source":(10.,10.),"target":(10.,20.)}
    assert _path_success(x,wrong,method="straight") is False


class DummyGate:
    def predict_proba(self,x):
        return np.tile([.1,.9],(len(x),1))


def test_v47_path_and_gate_decisions_must_match_original_risk_table(monkeypatch):
    monkeypatch.setattr(study,"pair_rank_scores",lambda scaler,ranker,scene:np.asarray([.9]))
    monkeypatch.setattr(study,"feature_row",lambda scene,score:np.zeros(13))
    monkeypatch.setattr(path_module,"_path_success",lambda scene,pair,method="straight":True)
    scenes=[];rows=[]
    for section in range(60):
        for gap in range(2):
            s=path_fixture()
            s["section"]=section;s["id"]=f"g{section}:{gap}"
            s["proposals"]=[{"source":(48.,30.),"target":(48.,60.),"y":1}]
            scenes.append(s)
            rows.append({"method":"oof_scene_gate","section":section,"gap_id":s["id"],
                         "world":"LINK","top1_correct":1,"accept_score":.9})
            rows.append({"method":"oof_scene_gate","section":section,"gap_id":s["id"],
                         "world":"NO_LINK","top1_correct":0,"accept_score":.9})
    sweep=[{"threshold":t,"correctly_accepted_link_gaps":120 if t<=.9 else 0,
            "n_wrong_link_source_sections":0,
            "n_false_synthetic_nolink_source_sections":60 if t<=.9 else 0,
            "conditional_joint_95pct_10pct_gate":t>.9}
           for t in CAL_THRESHOLDS]
    evidence=audited_path_readout(
        scenes,None,None,DummyGate(),rows,{"oof_scene_gate":{"predeclared_sweep":sweep}})
    assert evidence["correct_top1_predicted_pairs"]==120
    assert evidence["successful_straight_path_given_top1_correct"]==120
    assert evidence["successful_maximin_path_given_top1_correct"]==120
    assert evidence["best_correct_straight_paths_under_conditional_simultaneous_CP"]==0
    assert evidence["best_correct_maximin_paths_empirical_10pct_posthoc"]==0
    assert len(evidence["case_rows"])==120
    broken=[dict(x) for x in rows]
    broken[0]["top1_correct"]=0
    with pytest.raises(RuntimeError,match="different top1"):
        audited_path_readout(
            scenes,None,None,DummyGate(),broken,{"oof_scene_gate":{"predeclared_sweep":sweep}})
