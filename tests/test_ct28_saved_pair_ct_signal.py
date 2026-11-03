"""Actual committed local HGB PAIR CT-feature dependence, no medical volumes."""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                       "scripts/research/ccta_graph_lira_safe_repair"))
from ct28_saved_pair_ct_signal import (
    run_pair_signal,_shuffle_ct_vectors,DEFAULT_FEATURES
)


def test_saved_historical_HGB_geometry_fusion_probability_reproduction():
    assert DEFAULT_FEATURES.exists()
    report,runs=run_pair_signal(repetitions=5,seed=100)
    pair=report["frozen_Pair_geometry"]
    fusion=report["frozen_Pair_geometry_plus_CT"]
    assert (pair["pair_candidate_count"],pair["pair_positive_candidates"])==(2298,170)
    assert (fusion["pair_candidate_count"],fusion["pair_positive_candidates"])==(2298,170)
    assert pair["roc_auc"]==pytest.approx(.9769764484741266,rel=1e-7)
    assert pair["average_precision"]==pytest.approx(.7174459397616875,rel=1e-7)
    assert fusion["roc_auc"]==pytest.approx(.9910161432994251,rel=1e-7)
    assert fusion["average_precision"]==pytest.approx(.901173169225066,rel=1e-7)
    assert fusion["average_precision"]>pair["average_precision"]
    assert len(runs)==5
    assert report["held_out_test_accessed"] is False


def test_patient_sham_preserves_original_CT_feature_histograms_and_geometry():
    frame=pd.DataFrame({
        "scan_id":[957,957,957,961,961],
        "scene_id":["s0","s1","s2","s3","s4"],
        "candidate_id":["a","b","c","d","e"],
        "y":[1,0,0,1,0],
        "pair_g0":[.0,.1,.2,.3,.4],
        **{f"pair_ct{i}":[1+i,2+i,3+i,4+i,5+i]
           for i in range(38)}
    })
    original=frame.copy(deep=True)
    changed=_shuffle_ct_vectors(frame,seed=17)
    assert changed[["scene_id","candidate_id","y","scan_id","pair_g0"]].equals(
        original[["scene_id","candidate_id","y","scan_id","pair_g0"]]
    )
    for patient in (957,961):
        orig=original.loc[original.scan_id.eq(patient)]
        fake=changed.loc[changed.scan_id.eq(patient)]
        assert sorted(orig["pair_ct0"])==sorted(fake["pair_ct0"])
        assert sorted(orig["pair_ct37"])==sorted(fake["pair_ct37"])
    assert frame.equals(original)
