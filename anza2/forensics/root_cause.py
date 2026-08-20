"""Predeclared single-label RC1--RC7 classification for Phase 3C-A."""

from __future__ import annotations

from typing import Any


def _mechanism_pass(row: dict[str, Any]) -> bool:
    return bool(
        row.get("overall_branch_recall") is not None
        and row["overall_branch_recall"] >= 0.98
        and row.get("x_branch_recall") is not None
        and row["x_branch_recall"] >= 0.95
        and row.get("parallel_fault_false_bridge") is not None
        and row["parallel_fault_false_bridge"] <= 0.02
    )


def classify_root_cause(
    matrix: dict[str, dict[str, Any]],
    fusion: dict[str, Any],
    *,
    phase2b_reproduced: bool,
) -> dict[str, Any]:
    """Choose exactly one root cause without opening a new data stream."""

    if not phase2b_reproduced or not _mechanism_pass(matrix["F0_full_oracle"]):
        return {
            "root_cause": "STOP_ORACLE_EFFECT_NOT_REPRODUCIBLE", "rc_code": "RC7",
            "repair_authorized": False,
            "reason": "The oracle relation no longer passes its frozen mechanism checks.",
        }
    f1 = _mechanism_pass(matrix["F1_full_learned"])
    f8 = _mechanism_pass(matrix["F8_learned_geometry_oracle_membership"])
    f9 = _mechanism_pass(matrix["F9_learned_membership_oracle_geometry"])
    if not f1:
        if f8 and not f9:
            return {
                "root_cause": "ROOT_CAUSE_MEMBERSHIP_LEARNING", "rc_code": "RC1",
                "repair_authorized": True,
                "reason": "Oracle membership restores the mechanism while oracle geometry alone does not.",
            }
        learned_orientation_only = _mechanism_pass(matrix["F3_learned_orientation_only"])
        learned_base_only = _mechanism_pass(matrix["F4_learned_base_scale_only"])
        learned_h_only = _mechanism_pass(matrix["F5_learned_hyperbolicity_only"])
        if not learned_orientation_only and learned_base_only and learned_h_only:
            return {
                "root_cause": "ROOT_CAUSE_ORIENTATION_LEARNING", "rc_code": "RC2",
                "repair_authorized": True,
                "reason": "Learned orientation alone breaks an otherwise passing oracle field.",
            }
        if learned_orientation_only and (not learned_base_only or not learned_h_only or not f8):
            return {
                "root_cause": "ROOT_CAUSE_HYPERBOLIC_SCALE_LEARNING", "rc_code": "RC3",
                "repair_authorized": True,
                "reason": "Replacing learned scale/hyperbolicity with the frozen oracle reference restores the mechanism.",
            }
        return {
            "root_cause": "ROOT_CAUSE_MULTI_COMPONENT_FIELD_LEARNING", "rc_code": "RC4",
            "repair_authorized": True,
            "reason": "No single component replacement explains the failure; interacting field errors do.",
        }
    raw = fusion["sources"]["raw_anza"]["low_fpr"]["tpr_at_fpr_0_05"]
    generic = fusion["sources"]["generic"]["low_fpr"]["tpr_at_fpr_0_05"]
    fused = fusion["sources"]["fused"]["low_fpr"]["tpr_at_fpr_0_05"]
    if raw >= generic + 0.05 and fused < generic + 0.01:
        return {
            "root_cause": "ROOT_CAUSE_GENERIC_HEAD_REDUNDANCY", "rc_code": "RC5",
            "repair_authorized": True,
            "reason": "Raw learned ANZA is structurally useful, but fusion adds no practical value over generic affinity.",
        }
    return {
        "root_cause": "ROOT_CAUSE_METRIC_OBJECTIVE_MISMATCH", "rc_code": "RC6",
        "repair_authorized": True,
        "reason": "The learned raw field retains the mechanism, but the local low-FPR edge objective does not express a practical incremental gain.",
    }
