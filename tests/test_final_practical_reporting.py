import json

from path_completion.final_practical_reporting import FINAL_ROOT, build_closeout
from scripts.validate_final_practical_cycle import validate


def test_final_practical_closeout_is_machine_linked_and_claim_safe() -> None:
    numbers = build_closeout()
    assert numbers["statuses"]["cracks_partial_labels"] == "CRACKS_PARTIAL_LABEL_SUCCESS"
    assert numbers["statuses"]["cracks_real_pair_classifier"] == "CRACKS_REAL_PAIR_CLASSIFIER_GATE_FAIL"
    assert numbers["statuses"]["anza_guided_completion"] == "NOT_RUN_GATE_LOCKED"
    assert numbers["statuses"]["final"] == "FINAL_PRACTICAL_NEGATIVE_WITH_ROOT_CAUSE"
    assert numbers["expert_data_accessed"] is False
    stored = json.loads((FINAL_ROOT / "THESIS_NUMBERS.json").read_text())
    assert stored == numbers
    report = (FINAL_ROOT / "FINAL_PRACTICAL_REPORT.md").read_text()
    assert f"{numbers['real_pair_classifier']['tpr']:.6f}" in report
    assert "ANZA_GUIDED_COMPLETION_SUCCESS" not in report


def test_final_practical_validator_passes_frozen_artifacts() -> None:
    build_closeout()
    result = validate()
    assert result["status"] == "PASS", result["failures"]
    assert result["expert_data_accessed"] is False
