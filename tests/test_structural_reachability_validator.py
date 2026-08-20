from pathlib import Path

from scripts.validate_structural_reachability_phase_a import validate_phase_a


def test_frozen_phase_a_artifacts_validate_fail_closed() -> None:
    root = Path("results/structural_reachability/phase_a")
    result = validate_phase_a(root)
    assert result["status"] == "PASS"
    assert result["research_status"] in {
        "PHASE_A_PASS", "STOP_ARCHITECTURAL_ANZA_NO_CAUSAL_GEOMETRY_GAIN"
    }
    assert result["expert_data_accessed"] is False
    assert result["training_performed"] is False
