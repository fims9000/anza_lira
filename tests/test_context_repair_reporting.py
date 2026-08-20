from __future__ import annotations

import method_repair.context_reporting as reporting


def test_root_cause_is_derived_from_failed_frozen_checks(monkeypatch) -> None:
    monkeypatch.setattr(
        reporting,
        "_gate_target_audit",
        lambda: {"effective_weighted_positive_mass_fraction": 1.0},
    )
    gate = {
        "decisions": {
            "B1": {"checks": {"gate_auroc": False, "false_bridge": False}},
            "B2": {"checks": {"neff_mean": False}},
            "B3": {"checks": {"gap_recovery": False, "visible_dice_safe": False}},
        }
    }
    summaries = {
        "B0": {"metrics": {"false_bridge_rate": 1.0}},
        "B1": {"metrics": {"route_entropy_normalized": 0.4}},
        "B2": {"metrics": {"route_entropy_normalized": 0.3}},
        "B3": {"metrics": {
            "gate_auroc": 0.9,
            "mode_count_accuracy": 1.0,
            "membership_set_kl": 0.1,
            "correction_to_base_abs_mean_ratio": 0.1,
            "false_bridge_rate": 1.0,
        }},
    }
    result = reporting._root_cause(gate, summaries)
    assert result["causes"] == [
        "POSITIVE_COMPLETION_RECOVERY_INSUFFICIENT",
        "RESIDUAL_BRANCH_VIOLATED_SEGMENTATION_SAFETY",
    ]
