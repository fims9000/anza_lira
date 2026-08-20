from __future__ import annotations

from method_repair.root_cause import build_root_cause_analysis


def test_real_root_cause_analysis_is_fail_closed_and_uses_no_test_or_expert(tmp_path) -> None:
    result = build_root_cause_analysis(
        "results/method_repair/synthetic_v2",
        tmp_path / "root_cause.json",
        device="cpu",
        diagnostic_samples=1,
    )
    assert result["status"] == "METHOD_REPAIR_NEGATIVE_WITH_ROOT_CAUSE"
    assert result["cracks_training"].startswith("NOT_RUN")
    assert result["expert_evaluation"] == "NOT_RUN"
    assert result["old_test_samples_opened"] == 0
    assert result["new_test_samples_opened"] == 0
    assert result["expert_data_accessed"] is False
    assert [item["id"] for item in result["root_causes"]] == [
        "RC1_POINTWISE_AMBIGUITY_OBSERVABILITY",
        "RC2_NO_NEGATIVE_GAP_OBJECTIVE",
        "RC3_CONTEXT_ONLY_IN_TRANSPORT_NOT_GATE",
        "RC4_RESIDUAL_SAFETY_WORKED",
    ]
    assert all(item["status"] != "HYPOTHESIS_UNTESTED" for item in result["root_causes"])
