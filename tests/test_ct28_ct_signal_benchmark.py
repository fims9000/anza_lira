"""Real committed CT28 CT-only/geometry-only/sham-CT benchmark contract."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ct28_ct_signal_benchmark import (
    read_frozen,expected_columns,make_matrix,metrics,
    TRAIN,VAL,TEST,DEFAULT_SOURCE,
)


def test_actual_15k_frozen_CCTA_candidate_features_and_patient_firewall():
    frame=read_frozen(DEFAULT_SOURCE)
    assert len(frame)==15092
    assert frame.loc[frame.split.eq("train")].shape[0]==11021
    assert frame.loc[frame.split.eq("val")].shape[0]==4071
    assert set(frame.loc[frame.split.eq("train"),"scan_id"])==TRAIN
    assert set(frame.loc[frame.split.eq("val"),"scan_id"])==VAL
    assert not set(frame.scan_id)&TEST
    for kind,ng,nc in (("PAIR",7,38),("JUNCTION",9,152)):
        subset=frame[frame.candidate_type.eq(kind)]
        geometry,ct=expected_columns(subset,kind)
        assert len(geometry)==ng
        assert len(ct)==nc
        assert subset.loc[subset.split.eq("val"),"y"].sum()>0


def test_patientwise_permutation_keeps_all_CT_rows_and_labels_but_breaks_alignment():
    rng=np.random.default_rng(15)
    frame=pd.DataFrame({
        "scan_id":[953]*8+[955]*8,
        "g":rng.normal(size=16),
        "ct_a":np.arange(16,dtype=float),
        "ct_b":np.arange(16,dtype=float)*100+1,
        "y":[0,1]*8,
    })
    original=make_matrix(frame,geometry=["g"],ct=["ct_a","ct_b"],
                         variant="geometry_plus_ct",seed=14)
    transformed=make_matrix(
        frame,geometry=["g"],ct=["ct_a","ct_b"],
        variant="geometry_plus_shuffled_ct",seed=14
    )
    assert np.array_equal(original[:,0],transformed[:,0])
    assert np.array_equal(original[:8,1:][np.argsort(original[:8,1])],
                          transformed[:8,1:][np.argsort(transformed[:8,1])])
    assert np.array_equal(original[8:,1:][np.argsort(original[8:,1])],
                          transformed[8:,1:][np.argsort(transformed[8:,1])])
    assert not np.array_equal(original[:,1:],transformed[:,1:])
    assert np.array_equal(transformed,make_matrix(
        frame,geometry=["g"],ct=["ct_a","ct_b"],
        variant="geometry_plus_shuffled_ct",seed=14
    ))


def test_not_a_test_set_and_real_source_provenance_immutable(tmp_path):
    frame=pd.read_csv(DEFAULT_SOURCE)
    frame.loc[frame.split.eq("val"),"scan_id"]=954
    source=tmp_path/"tampered.csv"
    frame.to_csv(source,index=False)
    with pytest.raises(ValueError,match="Val patients changed"):
        read_frozen(source)


def test_probabilities_and_constant_target_metrics():
    one=metrics(np.array([0,1,0,1]),np.array([.1,.8,.3,.9]))
    assert one["auroc"]==1.
    assert one["average_precision"]==1.
    constant=metrics(np.array([1,1]),np.array([.5,.6]))
    assert constant["average_precision"] is None
