"""Reproducible *real committed* CT28-HGB saved-feature sham, no private CT."""
import pytest
import numpy as np
import pandas as pd

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                       "scripts/research/ccta_graph_lira_safe_repair"))

from ct28_saved_ct_signal import (
    run_signal_diagnostic,_permuted_ct_scores,DEFAULT_MODEL,DEFAULT_SCORES
)


def test_committed_historical_HGB_is_reproduced_before_ct_permutations():
    assert DEFAULT_MODEL.exists()
    assert DEFAULT_SCORES.exists()
    report,runs,base=run_signal_diagnostic(repetitions=5,seed=11)
    assert report["n_validation_scenes"]==368
    assert report["n_validation_candidates"]==4071
    assert report["baseline"]["raw_accuracy"]==pytest.approx(341/368,abs=1e-10)
    assert report["baseline"]["thresholded_accuracy"]==pytest.approx(337/368,abs=1e-10)
    assert report["baseline"]["log_loss"]==pytest.approx(.22403215965681825,rel=1e-5)
    assert report["model_scope"]=="committed_historical_local_HGB_NOT_selected_K10_15"
    assert report["CT_sham"]["replications"]==5
    assert len(runs)==5 and len(base)==368
    assert report["held_out_test_accessed"] is False


def test_ct_sham_keeps_geometry_fusion_identity_and_patient_marginals():
    orig=pd.DataFrame({
        "scene_id":["s1","s2","s3","s4","s5","s6"],
        "candidate_id":["a","b","c","d","e","f"],
        "scan_id":[957,957,957,957,961,961],
        "candidate_type":["PAIR","PAIR","JUNCTION","JUNCTION","PAIR","PAIR"],
        "geometry_score":[.1,.2,.3,.4,.5,.6],
        "ct_score":[.01,.12,.23,.34,.45,.56],
        "geometry_plus_ct_score":[.9,.8,.7,.6,.5,.4],
    })
    transformed=_permuted_ct_scores(orig,seed=7)
    for field in ("candidate_id","scene_id","scan_id","candidate_type",
                  "geometry_score","geometry_plus_ct_score"):
        assert transformed[field].equals(orig[field])
    for group,sub in orig.groupby(["scan_id","candidate_type"]):
        patched=transformed.loc[sub.index]
        assert sorted(patched.ct_score)==sorted(sub.ct_score)
    assert set(transformed.ct_score)==set(orig.ct_score)
    with pytest.raises(ValueError,match="Unknown CT"):
        _permuted_ct_scores(orig,seed=7,mode="other")
