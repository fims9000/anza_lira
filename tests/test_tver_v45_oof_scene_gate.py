"""V45 pure scene-level selective gate tests, no Thebe data downloads."""
import numpy as np
import pytest
from scripts.tver_v45_oof_scene_gate import (
    FEATURE_NAMES,CAL_THRESHOLDS,feature_row,source_fold_id,cp_upper,evaluate)


def sample_scene():
    return {
        "proposals":[
            {"features":[.4,0.,0.,0.,0.,.5,.3,0.,0.,0.,0.,0.]},
            {"features":[.5,0.,0.,0.,0.,.4,.2,0.,0.,0.,0.,0.]}],
        "visible_prob":np.full((96,96),.35,dtype=np.float32),
        "source_count":2,"branch_count":10,
        "kind":"LINK","section":40,"id":"40:0",
    }


def test_scene_features_are_prediction_only_and_empty_fail_closed():
    x=sample_scene()
    a=feature_row(x,[.8,.3])
    assert a.shape==(len(FEATURE_NAMES),)
    assert a[0]==pytest.approx(.8)
    assert a[1]==pytest.approx(.5)
    assert a[5]==pytest.approx(np.log1p(2))
    assert a[6]==.5
    x["proposals"]=[]
    assert np.all(feature_row(x,[] )==0)
    with pytest.raises(ValueError):
        feature_row(x,[.9])


def test_source_fold_is_deterministic_and_disjoint():
    for fold in range(5):
        train={s for s in range(100) if source_fold_id(s)!=fold}
        hold={s for s in range(100) if source_fold_id(s)==fold}
        assert len(train)==80 and len(hold)==20 and not (train&hold)
    with pytest.raises(ValueError):
        source_fold_id(100)


def sample_rows(perfect):
    out=[]
    for method in ("raw_pair_rank_score","oof_scene_gate"):
        for section in range(60):
            for j in range(2):
                for world in ("LINK","NO_LINK"):
                    i=section*2+j
                    correct=world=="LINK" and i<41
                    # Strong gate separates correct positive from all others
                    # on synthetic fixtures; raw model is intentionally poor.
                    score=(.95 if correct else .2) if method=="oof_scene_gate" else .8
                    out.append({
                        "section":section,"gap_id":f"{section}:{j}",
                        "world":world,"method":method,
                        "n_candidates":1,"top1_correct":int(correct),
                        "true_candidate_available":int(world=="LINK" and i<65),
                        "accept_score":score if perfect else .8,
                    })
    return out


def test_exact_multiple_comparison_risk_and_nonzero_coverage_possible():
    summary=evaluate(sample_rows(perfect=True))
    x=summary["oof_scene_gate"]
    assert x["rank_top1_correct_before_acceptance"]==41
    assert x["best_correct_with_conditional_simultaneous_CP_gate"]==41
    assert x["simultaneous_CP_zero_error_section_risk_bound"]<.1
    assert summary["raw_pair_rank_score"]["best_correct_with_conditional_simultaneous_CP_gate"]==0
    assert len(x["predeclared_sweep"])==len(CAL_THRESHOLDS)
    assert cp_upper(0,60,.05/22)<.1


def test_no_fake_coverage_if_scores_do_not_separate_and_input_mismatch():
    out=evaluate(sample_rows(perfect=False))
    assert out["oof_scene_gate"]["best_correct_with_conditional_simultaneous_CP_gate"]==0
    with pytest.raises(RuntimeError,match="Original"):
        evaluate(sample_rows(perfect=True)[:-1])
