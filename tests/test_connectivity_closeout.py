from pathlib import Path

from connectivity_repair.closeout import TERMINAL_STATUS, validate_pretraining_gates


def test_current_pretraining_gate_state_is_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    result = validate_pretraining_gates(root)
    assert result["status"] == TERMINAL_STATUS
    assert result["d0_d3_training"] == "NOT_AUTHORIZED_NOT_RUN"
    assert result["cracks"] == "NOT_AUTHORIZED_NOT_RUN"
    assert result["v5_test"] == "LOCKED_UNOPENED"

