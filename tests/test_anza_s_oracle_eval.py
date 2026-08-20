import numpy as np
import pytest
import torch

from anza_s.cases import candidate_stream, candidates_for_sample
from anza_s.oracle_eval import METHODS, _candidate_scores, calibrate_thresholds, evaluate, oracle_rows
from synthetic.crossing_trace_bench_v4 import generate_sample_v4


def test_oracle_candidate_stream_is_complete_and_balanced_by_frozen_cases():
    stream = candidate_stream("train", image_size=64)
    assert len(stream) == 536
    labels = [candidate.label for _sample, candidate in stream]
    assert labels.count(1) == 288
    assert labels.count(0) == 248
    assert {candidate.task for _sample, candidate in stream} == {
        "straight_gap", "negative_gap", "curved_gap", "parallel_correct", "parallel_wrong",
        "x_correct", "x_wrong_turn", "ty_continuation",
    }


def test_oracle_refuses_confirm_and_test():
    with pytest.raises(PermissionError):
        oracle_rows("confirm")
    with pytest.raises(PermissionError):
        oracle_rows("test")


def test_matched_gap_geometry_gives_same_trajectory_and_shadowing_scores():
    positive = generate_sample_v4("train", 0, image_size=64)
    negative = generate_sample_v4("train", 128, image_size=64)
    positive_candidate = candidates_for_sample(positive)[0]
    negative_candidate = candidates_for_sample(negative)[0]
    positive_scores, _ = _candidate_scores(positive, positive_candidate, device=torch.device("cpu"))
    negative_scores, _ = _candidate_scores(negative, negative_candidate, device=torch.device("cpu"))
    for method in ("O2_tangent_streamline", "O3_cocycle_rollout", "O4_cocycle_shadowing"):
        assert np.isclose(positive_scores[method], negative_scores[method], atol=1e-12)


def test_taskwise_calibration_and_gate_schema():
    rows = []
    for method in METHODS:
        for task in ("straight_gap", "curved_gap", "x_correct", "parallel_correct", "ty_continuation"):
            rows.extend({"method": method, "task": task, "label": 1, "score": 0.9} for _ in range(20))
        for task in ("x_wrong_turn", "parallel_wrong", "negative_gap"):
            rows.extend({"method": method, "task": task, "label": 0, "score": 0.4} for _ in range(40))
    thresholds = calibrate_thresholds(rows)
    assert all(np.isfinite(value["threshold"]) for value in thresholds.values())
    result = evaluate(rows, thresholds)
    assert set(result["methods"]) == set(METHODS)
    assert set(result["baseline_gate_checks"]) == {"O0_scalar_anza", "O2_tangent_streamline"}
    assert result["training_performed"] is False
    assert result["confirm_opened"] is False
    assert result["gate_pass"] is False  # no relative X reduction can be claimed from baseline FPR=0
