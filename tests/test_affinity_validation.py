from copy import deepcopy

from affinity_repair.validation import HARD_STRATA, decide_affinity_gate


def _metrics():
    return {
        "visible_dice": 0.90,
        "visible_cldice": 0.90,
        "latent_skeleton_f1_2px": 0.90,
        "endpoint_f1": 0.90,
        "false_bridge_rate": 0.40,
        "gap_recovery_rate": 0.90,
        "hard_affinity_macro_ap": 0.90,
        "matched_negative_gap_auroc": 0.90,
        "true_minus_false_affinity_ci95": [0.2, 0.1, 0.3],
        "beta_on_minus_off_latent_skeleton_f1_ci95": [0.02, 0.01, 0.03],
    }


def test_hard_strata_are_predeclared_exactly():
    assert set(HARD_STRATA) == {
        "acute_angle_crossing", "similar_tangent_crossing", "nontrivial_pairing",
        "crossing_near_junction", "near_parallel_close", "matched_negative_gap",
    }


def test_gate_pass_and_single_failure_are_fail_closed():
    baseline = _metrics()
    baseline["false_bridge_rate"] = 0.80
    summaries = {name: {"metrics": deepcopy(_metrics())} for name in ("C0", "C1", "C2", "C3")}
    summaries["C0"]["metrics"] = baseline
    passed = decide_affinity_gate(summaries)
    assert passed["status"] == "AFFINITY_MECHANISM_PASS"
    summaries["C2"]["metrics"]["hard_affinity_macro_ap"] = 0.84
    summaries["C3"]["metrics"]["hard_affinity_macro_ap"] = 0.84
    failed = decide_affinity_gate(summaries)
    assert failed["status"] == "AFFINITY_MECHANISM_FAIL"
    assert not failed["confirm_authorized"] and not failed["cracks_authorized"]
