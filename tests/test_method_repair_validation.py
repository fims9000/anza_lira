from __future__ import annotations

import copy

from method_repair.validation import bootstrap_mean_ci, decide_mechanism_gate


def _summary(*, baseline: bool = False) -> dict:
    metrics = {
        "visible_dice": 0.80,
        "visible_cldice": 0.78,
        "false_bridge_rate": 0.5,
        "route_average_precision": None if baseline else 0.80,
        "route_mrr": None if baseline else 0.82,
        "route_entropy_normalized": None if baseline else 0.70,
        "route_excess_over_chance": None if baseline else 0.20,
        "neff_junction_minus_straight": None if baseline else 0.25,
        "neff_junction_minus_straight_ci95": [None, None] if baseline else [0.10, 0.40],
        "ambiguity_junction_minus_straight": None if baseline else 0.20,
        "ambiguity_junction_minus_straight_ci95": [None, None] if baseline else [0.08, 0.32],
    }
    return {"metrics": metrics}


def test_bootstrap_is_deterministic_and_detects_positive_delta() -> None:
    first = bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4])
    second = bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4])
    assert first == second
    assert first[1] > 0


def test_gate_selects_only_all_pass_candidate() -> None:
    summaries = {"A0": _summary(baseline=True)}
    summaries.update({name: _summary() for name in ("A1", "A2", "A3", "A4")})
    summaries["A1"]["metrics"]["route_entropy_normalized"] = 0.99
    gate = decide_mechanism_gate(summaries)
    assert gate["cracks_authorized"] is True
    assert gate["selected_candidate"] in {"A2", "A3", "A4"}
    assert gate["decisions"]["A1"]["all_gates_pass"] is False
    assert gate["expert_data_accessed"] is False


def test_gate_fails_closed_on_visible_regression() -> None:
    summaries = {"A0": _summary(baseline=True)}
    summaries.update({name: _summary() for name in ("A1", "A2", "A3", "A4")})
    for name in ("A1", "A2", "A3", "A4"):
        summaries[name] = copy.deepcopy(summaries[name])
        summaries[name]["metrics"]["visible_dice"] = 0.78
    gate = decide_mechanism_gate(summaries)
    assert gate["status"] == "SYNTHETIC_MECHANISM_FAIL"
    assert gate["cracks_authorized"] is False
