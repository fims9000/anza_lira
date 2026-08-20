import pytest

from anza2.phase3d.oracle_graph_eval import calibrate_thresholds, evaluate_oracle_rows, oracle_rows


def test_oracle_eval_refuses_confirm_stream():
    with pytest.raises(PermissionError):
        oracle_rows("confirm")


def test_train_calibration_and_validation_gate_schema_on_small_rows():
    rows = []
    for method, positive, negative in (("G0_scalar", 0.8, 0.7), ("G1_mode_state", 0.9, 0.1)):
        rows.extend([
            {"method": method, "label": 1, "task": "x_correct", "score": positive},
            {"method": method, "label": 1, "task": "curved_continuation", "score": positive},
            {"method": method, "label": 1, "task": "parallel_correct", "score": positive},
            {"method": method, "label": 1, "task": "positive_gap", "score": positive},
            {"method": method, "label": 1, "task": "ty_continuation", "score": positive},
            {"method": method, "label": 0, "task": "x_wrong_turn", "score": negative},
            {"method": method, "label": 0, "task": "parallel_wrong", "score": negative},
            {"method": method, "label": 0, "task": "negative_gap", "score": negative},
        ])
    # Calibration negatives need at least one accepted operating point below 5%.
    train = rows + [dict(row, score=0.0) for row in rows if row["label"] == 0 for _ in range(40)]
    thresholds = calibrate_thresholds(train)
    result = evaluate_oracle_rows(rows, thresholds)
    assert set(result["gate_checks"]) == {
        "positive_noninferiority", "x_wrong_turn_relative_reduction_at_least_50pct",
        "parallel_false_bridge_noninferiority", "negative_gap_false_bridge_noninferiority",
        "curved_continuation_noninferiority",
    }
    assert result["training_performed"] is False
    assert result["confirm_evaluation_opened"] is False
