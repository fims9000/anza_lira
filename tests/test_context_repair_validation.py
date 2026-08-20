from __future__ import annotations

from copy import deepcopy

from method_repair.context_validation import decide_context_gate


def _metrics() -> dict:
    return {
        "visible_dice": 0.90,
        "visible_cldice": 0.92,
        "route_average_precision": 0.95,
        "route_entropy_normalized": 0.30,
        "orientation_error_model_modes_median_deg": 8.0,
        "neff_junction_minus_straight": 0.20,
        "neff_junction_minus_straight_median": 0.15,
        "neff_junction_minus_straight_ci95": [0.10, 0.30],
        "membership_set_kl": 0.50,
        "gate_auroc": 0.90,
        "gate_junction_minus_straight_ci95": [0.10, 0.30],
        "negative_gap_count": 128,
        "false_bridge_rate": 0.30,
        "gap_recovery_rate": 0.90,
    }


def test_context_gate_selects_only_all_gate_candidate() -> None:
    baseline = _metrics()
    baseline["false_bridge_rate"] = 1.0
    summaries = {name: {"metrics": deepcopy(_metrics())} for name in ("B0", "B1", "B2", "B3")}
    summaries["B0"]["metrics"] = baseline
    summaries["B1"]["metrics"]["gate_auroc"] = 0.5
    summaries["B2"]["metrics"]["membership_set_kl"] = 0.9
    result = decide_context_gate(summaries)
    assert result["status"] == "CONTEXT_MECHANISM_PASS"
    assert result["selected_candidate"] == "B3"
    assert result["confirm_authorized"] is True
    assert result["cracks_authorized"] is False


def test_context_gate_fails_closed_if_false_bridge_improves_by_too_little() -> None:
    baseline = _metrics()
    baseline["false_bridge_rate"] = 0.55
    summaries = {name: {"metrics": deepcopy(_metrics())} for name in ("B0", "B1", "B2", "B3")}
    summaries["B0"]["metrics"] = baseline
    for name in ("B1", "B2", "B3"):
        summaries[name]["metrics"]["false_bridge_rate"] = 0.40
    result = decide_context_gate(summaries)
    assert result["status"] == "CONTEXT_MECHANISM_FAIL"
    assert result["selected_candidate"] is None
    assert not result["confirm_authorized"]
