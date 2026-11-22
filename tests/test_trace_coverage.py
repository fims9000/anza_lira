"""Synthetic, label-blind coverage-layer invariants."""
import numpy as np
import pytest

from trace_extraction.coverage import (
    annotated_trace_coverage, classify_trace_evidence, mode_layers,
)


def test_keeps_connected_and_isolated_weak_evidence_separate():
    p=np.zeros((32,64),np.float32)
    p[10,2:8]=.92
    p[10,8:20]=.44
    p[23,10:22]=.42
    p[23,10:12]=.62
    before=p.copy()
    layers=classify_trace_evidence(p,low=.35,high=.80)
    assert layers.high_confidence[10,4]
    assert layers.supported_weak[10,12]
    assert layers.review_weak[23,18]
    assert np.array_equal(layers.possible_trace,p>=.35)
    assert not np.any(layers.high_confidence & layers.review_weak)
    assert not np.any(layers.supported_weak & layers.review_weak)
    assert np.array_equal(p,before)


def test_coverage_tradeoff_is_monotone_and_precision_not_fabricated():
    p=np.zeros((48,80),np.float32)
    p[20,10:25]=.92
    p[20,25:57]=.38
    p[35,10:50]=.4    # unannotated possible structure
    expert=np.zeros_like(p,bool)
    expert[20,10:57]=True
    rows=annotated_trace_coverage(p,expert,thresholds=(.8,.6,.3))
    assert rows[0]["annotated_centerline_coverage"] < rows[-1]["annotated_centerline_coverage"]
    assert rows[0]["foreground_fraction"] < rows[-1]["foreground_fraction"]
    assert rows[-1]["annotated_centerline_coverage"]==1.
    assert rows[-1]["predicted_pixels_near_annotation_fraction"]<1.


def test_no_expert_line_is_not_reported_as_zero_model_recall():
    p=np.zeros((16,16),np.float32)
    rows=annotated_trace_coverage(p,np.zeros_like(p,bool))
    assert rows[0]["annotated_centerline_coverage"] is None
    assert rows[0]["reference_centerline_pixels"]==0


def test_modes_are_operator_configurations_not_implicit_gt_threshold_search():
    p=np.full((16,16),.32,np.float32)
    assert mode_layers(p,"coverage").possible_trace.all()
    assert not mode_layers(p,"strict").possible_trace.any()
    with pytest.raises(ValueError,match="Unknown"):
        mode_layers(p,"optimize_test")


@pytest.mark.parametrize("bad",[-.1,1.1,float("nan")])
def test_invalid_probability_fails_closed(bad):
    p=np.full((16,16),.2,np.float32)
    p[0,0]=bad
    with pytest.raises(ValueError,match="probability"):
        mode_layers(p)


def test_fragment_level_recall_reports_separate_evidence_not_fault_identity():
    p=np.zeros((64,128),np.float32)
    p[20,10:85]=.9
    p[20,41:61]=.30
    p[44,10:95]=.9
    ref=np.zeros_like(p,bool)
    ref[20,10:85]=True
    ref[44,10:95]=True
    high,low=annotated_trace_coverage(p,ref,thresholds=(.8,.25))
    assert high["reference_trace_fragment_count"]>=2
    assert high["annotated_fragment_recall_95pct"]<1.
    assert low["annotated_fragment_recall_95pct"]==1.
    assert high["annotated_centerline_coverage"]<1.
    assert low["annotated_centerline_coverage"]==1.
