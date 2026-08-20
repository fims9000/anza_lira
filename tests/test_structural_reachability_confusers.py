from pathlib import Path

from scripts.analyze_structural_reachability_confusers import build_confuser_audit


def test_predeclared_confuser_audit_cannot_change_primary_gate() -> None:
    result = build_confuser_audit(Path("results/structural_reachability/phase_a"))
    assert result["pair_count"] == 20
    assert result["primary_gate_unchanged"] is True
    assert result["expert_data_accessed"] is False
    assert result["training_performed"] is False
