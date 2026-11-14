"""One-shot within-field holdout evaluation has no model selection or TEST7."""
import numpy as np
import pytest
import scripts.tver_v48_locked_holdout as study
import scripts.tver_v26_predicted_cracks_e2e as path_module
from scripts.tver_v48_locked_holdout import never_tuned_heldout_metrics,CAL_THRESHOLDS


class Gate:
    def predict_proba(self,x):
        return np.tile([.10,.90],(len(x),1))


def fixture():
    scenes=[];rows=[]
    for section in range(30,40):
        for k in range(2):
            tag=f"{section}:{k}"
            scenes.append({
                "kind":"LINK","section":section,"id":tag,
                "has_true_proposal":True,
                "proposals":[{"source":(48.,30.),"target":(48.,60.),"y":1}],
            })
            rows.extend([
                {"method":"oof_scene_gate","section":section,"gap_id":tag,
                 "world":"LINK","top1_correct":1,"accept_score":.9},
                {"method":"oof_scene_gate","section":section,"gap_id":tag,
                 "world":"NO_LINK","top1_correct":0,"accept_score":.9},
            ])
    return scenes,rows


def test_only_locked_holdout_and_no_claim_of_10pct_risk_certification(monkeypatch):
    monkeypatch.setattr(study,"pair_rank_scores",lambda scaler,ranker,scene:np.asarray([.9]))
    monkeypatch.setattr(study,"feature_row",lambda scene,scores:np.zeros(13))
    monkeypatch.setattr(path_module,"_path_success",lambda scene,pair,method="straight":True)
    s,r=fixture()
    z=never_tuned_heldout_metrics(s,None,None,Gate(),r)
    assert z["frozen_heldout_source_section_ids"]==list(range(30,40))
    assert z["heldout_true_candidate_available"]==20
    assert z["heldout_rank_top1_correct"]==20
    assert z["heldout_straight_path_success"]==20
    assert z["heldout_maximin_path_success"]==20
    assert len(z["predeclared_11_threshold_results_NO_SELECTION"])==len(CAL_THRESHOLDS)
    assert z["one_fixed_threshold_zero_error_95pct_upper_if_independent"]==pytest.approx(.2588655508930523)
    at09=[x for x in z["predeclared_11_threshold_results_NO_SELECTION"] if x["threshold"]==.9][0]
    assert at09["accepted_correct_pairs"]==20
    assert at09["accepted_synthetic_NOLINK_source_sections"]==10
    assert z["observed_risk_NOT_95pct_certified_at_10pct"]


def test_stop_wrong_partitions_and_score_drift(monkeypatch):
    monkeypatch.setattr(study,"pair_rank_scores",lambda scaler,ranker,scene:np.asarray([.9]))
    monkeypatch.setattr(study,"feature_row",lambda scene,scores:np.zeros(13))
    monkeypatch.setattr(path_module,"_path_success",lambda scene,pair,method="straight":True)
    s,r=fixture()
    s[0]["section"]=40
    with pytest.raises(RuntimeError,match="Holdout"):
        never_tuned_heldout_metrics(s,None,None,Gate(),r)
    s,r=fixture()
    r[0]["top1_correct"]=0
    with pytest.raises(RuntimeError,match="drift"):
        never_tuned_heldout_metrics(s,None,None,Gate(),r)
