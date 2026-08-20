from pathlib import Path

from path_completion.closeout import FINAL_STATUS, TERMINAL_REASON, validate_frozen_inputs


def test_closeout_validates_all_frozen_locks_without_expert_access():
    root = Path(__file__).resolve().parents[1]
    result = validate_frozen_inputs(root)
    assert result["status"] == "PASS"
    assert all(result["checks"].values())


def test_closeout_status_is_claim_safe_and_does_not_hide_failed_gates():
    assert FINAL_STATUS == "MAXMIN_PATH_ORACLE_PASS"
    assert "LEARNED_CONFIRM" in TERMINAL_REASON
    assert "CLEAN_ANZA" in TERMINAL_REASON
