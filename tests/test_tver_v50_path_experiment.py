"""V50 coherent path-head readout: fixed OOF gate and correct metric audit."""
import numpy as np
import pytest
import scripts.tver_v50_path_net_experiment as study
import scripts.tver_v26_predicted_cracks_e2e as original_path


class FakeGate:
    def predict_proba(self,x):
        return np.tile([.10,.90],(len(x),1))


def build():
    scenes=[];rows=[]
    for section in range(60):
        for gap in range(2):
            sid=f"g{section}:{gap}"
            scene={
                "section":section,"id":sid,"kind":"LINK",
                "visible_prob":np.full((96,96),.55,np.float32),
                "gray":np.zeros((96,96),np.float32),
                "rectangle":(44,52,42,48),"origin":(0,0),
                "hidden_centerline":[(48,x) for x in range(42,48)],
                "proposals":[{"source":(48.,30.),"target":(48.,60.),"y":1}]
            }
            scenes.append(scene)
            rows.append({"method":"oof_scene_gate","section":section,"gap_id":sid,
                         "world":"LINK","top1_correct":1,"accept_score":.9})
            rows.append({"method":"oof_scene_gate","section":section,"gap_id":sid,
                         "world":"NO_LINK","top1_correct":0,"accept_score":.9})
    risk=[
        {"threshold":t,"correctly_accepted_link_gaps":120 if t<=.9 else 0,
         "n_wrong_link_source_sections":0,
         "n_false_synthetic_nolink_source_sections":60 if t<=.9 else 0,
         "conditional_joint_95pct_10pct_gate":t>.9}
         for t in study.CAL_THRESHOLDS
    ]
    return scenes,rows,{"oof_scene_gate":{"predeclared_sweep":risk}}


def test_learned_path_metric_never_bypasses_separate_gate(monkeypatch):
    monkeypatch.setattr(study,"pair_rank_scores",lambda s,r,scene:np.asarray([.9]))
    monkeypatch.setattr(study,"feature_row",lambda scene,score:np.zeros(13))
    monkeypatch.setattr(study,"score_new_paths",lambda scene,pair:{
        "boundary_straight":{"success":False,"found_observed_anchors":False},
        "boundary_hermite":{"success":False,"found_observed_anchors":False}
    })
    monkeypatch.setattr(study,"decode_candidate",
        lambda net,scene,pair,mode:(
            [(48,x) for x in range(42,48)] if mode=="learned_ribbon" else [],
            {"anchor_status":"observed_boundary_anchors"}))
    monkeypatch.setattr(original_path,"_path_success",
                        lambda scene,pair,method="straight":False)
    scenes,rows,risk=build()
    z=study.audited_path_readout(scenes,None,None,FakeGate(),rows,risk,None)
    assert z["correct_top1_predicted_pairs"]==120
    assert z["successful_learned_ribbon_given_top1_correct"]==120
    assert z["successful_untrained_ribbon_given_top1_correct"]==0
    assert z["best_correct_learned_ribbon_paths_under_conditional_simultaneous_CP"]==0
    assert z["best_correct_learned_ribbon_paths_empirical_10pct_posthoc"]==0
    assert len(z["case_rows"])==120
    bad=[dict(x) for x in rows]
    bad[0]["top1_correct"]=0
    with pytest.raises(RuntimeError,match="different top1"):
        study.audited_path_readout(scenes,None,None,FakeGate(),bad,risk,None)
