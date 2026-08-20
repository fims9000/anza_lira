from scripts.validate_original_anza_phase0 import validate_phase0


def test_original_anza_phase0_is_fail_closed_not_failed_validation() -> None:
    result = validate_phase0()
    assert result["status"] == "PASS"
    assert result["research_status"] == "STOP_OPERATOR_DEFINITION_MISMATCH"
    assert result["secondary_split_status"] == "STOP_NO_INDEPENDENT_CONFIRM_SPLIT"
    assert result["instrumentation_performed"] is False
    assert result["confirm_performed"] is False
    assert result["training_performed"] is False
    assert result["expert_data_accessed"] is False
    assert result["next_phase_allowed"] is False
