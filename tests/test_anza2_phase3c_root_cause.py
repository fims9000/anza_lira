from anza2.forensics.root_cause import classify_root_cause


def _row(passed):
    return {
        "overall_branch_recall": 1.0 if passed else 0.5,
        "x_branch_recall": 1.0 if passed else 0.5,
        "parallel_fault_false_bridge": 0.0 if passed else 1.0,
    }


def test_oracle_membership_restoration_selects_rc1_only():
    matrix = {name: _row(True) for name in (
        "F0_full_oracle", "F1_full_learned", "F3_learned_orientation_only",
        "F4_learned_base_scale_only", "F5_learned_hyperbolicity_only",
        "F8_learned_geometry_oracle_membership", "F9_learned_membership_oracle_geometry",
    )}
    matrix["F1_full_learned"] = _row(False)
    matrix["F9_learned_membership_oracle_geometry"] = _row(False)
    fusion = {"sources": {name: {"low_fpr": {"tpr_at_fpr_0_05": 0.1}} for name in ("raw_anza", "generic", "fused")}}
    result = classify_root_cause(matrix, fusion, phase2b_reproduced=True)
    assert result["rc_code"] == "RC1"
    assert result["repair_authorized"] is True
