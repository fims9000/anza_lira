import json

from scripts.validate_anza2_phase3 import validate


def test_phase3_validator_preserves_positive_oracle_and_negative_learned_gate():
    result = validate()
    assert result["status"] == "PASS"
    assert result["research_status"] == "STOP_PHASE3B_LEARNED_AFFINITY_NO_GAIN"
    assert result["phase2_positive_preserved"] is True
    assert result["phase4_allowed"] is False
    assert result["confirm_opened"] is False
    metrics = json.loads(open("results/anza2/phase3b/metrics_reaudited.json").read())
    assert metrics["three_seed_tpr_delta"] < metrics["minimum_tpr_delta"]
    assert all(row["generic_off"]["fpr"] <= 0.05 for row in metrics["seed_metrics"])
    assert all(row["anza_on"]["fpr"] <= 0.05 for row in metrics["seed_metrics"])
