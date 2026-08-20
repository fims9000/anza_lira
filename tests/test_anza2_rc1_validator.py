import json

from scripts.validate_anza2_rc1_membership_repair import validate


def test_rc1_validator_accepts_honest_bounded_stop():
    result = validate()
    assert result["status"] == "PASS"
    assert result["research_status"] == "STOP_RC1_MEMBERSHIP_REPAIR_FAILED"
    assert result["three_seed_runs_performed"] is False
    assert result["beta_fit_performed"] is False
    assert result["confirm_allowed"] is False


def test_rc1_stop_keeps_confirm_and_real_data_locked():
    result = validate()
    assert result["confirm_opened"] is False
    assert result["cracks_data_accessed"] is False
    assert result["expert_data_accessed"] is False
    metrics = json.loads(open("results/anza2/phase3c_b_rc1/selected_config.json").read())
    assert metrics["selected_config"] is None
    assert all(not row["membership_safety_pass"] for row in metrics["config_metrics"])
