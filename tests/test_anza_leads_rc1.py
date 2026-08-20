from __future__ import annotations

import numpy as np

from anza_leads.rc1_evaluation import _pixel_frontier, interpolate_frontier, select_threshold, threshold_candidates, unsupported_white_metrics
from anza_leads.rc1_protocol import BASE_PROTOCOL, build_split


def test_rc1_split_is_fresh_section_disjoint_and_expert_locked() -> None:
    split = build_split()
    groups = [set(split[key]) for key in ("training_pool", "calibration_buffer", "calibration", "development_buffer", "development")]
    assert not any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i + 1, len(groups)))
    forbidden = set(split["old_a1_active_sections_excluded_from_rc1_evaluation"]) | set(split["old_a1_selection_sections_excluded_from_rc1"])
    assert not ((set(split["calibration"]) | set(split["development"])) & forbidden)
    assert split["expert_data_accessed"] is False


def test_score_frontier_includes_extreme_tail_and_not_only_old_grid() -> None:
    candidates = threshold_candidates(np.asarray([0.1, 0.96, 0.9997]), count=17, explicit=BASE_PROTOCOL["calibration"]["explicit_thresholds"])
    assert 0.9999 in candidates
    assert 0.995 in candidates
    assert 0.9997 in candidates


def test_rc1_selection_never_falls_back_when_precision_infeasible() -> None:
    infeasible = select_threshold([{"threshold": .99, "precision": .89, "recall": .1, "dice": .2, "cldice": .3}])
    assert not infeasible["constraint_feasible"]
    assert infeasible["selected_threshold"] is None
    feasible = select_threshold([
        {"threshold": .97, "precision": .91, "recall": .4, "dice": .5, "cldice": .6},
        {"threshold": .99, "precision": .95, "recall": .2, "dice": .4, "cldice": .7},
    ])
    assert feasible["selected_threshold"] == .99


def test_vectorized_pixel_frontier_has_one_value_per_threshold() -> None:
    observation = {"section_id": 1, "probability": np.asarray([[.1, .7, .9]]),
                   "truth": np.asarray([[False, True, True]]), "valid": np.ones((1, 3), dtype=bool)}
    frontier = _pixel_frontier([observation], np.asarray([.5, .8, .95]))
    assert len(frontier) == 3
    assert all(np.isfinite(row["precision"]) for row in frontier)


def test_unsupported_white_separates_supported_continuation_from_island() -> None:
    target = np.zeros((15, 15), dtype=bool); weight = np.zeros((15, 15), dtype=float)
    target[7, 2:5] = True; weight[7, 2:5] = 1.0
    probability = np.zeros((15, 15), dtype=float)
    probability[7, 2:8] = 1.0  # connected continuation into white
    probability[1, 12] = 1.0   # isolated unsupported island
    before_target = target.copy(); before_weight = weight.copy()
    metrics = unsupported_white_metrics(probability, target, weight, .5)
    assert metrics["white_connected_foreground_fraction"] > 0
    assert metrics["unsupported_white_foreground_fraction"] > 0
    assert metrics["white_isolated_foreground_fraction"] > 0
    assert np.array_equal(target, before_target); assert np.array_equal(weight, before_weight)


def test_frontier_interpolation_is_diagnostic_and_deterministic() -> None:
    curve = [{"precision": .8, "cldice": .7}, {"precision": .9, "cldice": .8}]
    first = interpolate_frontier(curve, [.8, .85, .9]); second = interpolate_frontier(curve, [.8, .85, .9])
    assert first == second
    assert np.isclose(first[1]["cldice"], .75)


def test_rc1_locks_and_single_seed_budget() -> None:
    assert BASE_PROTOCOL["seed"] == 41
    assert BASE_PROTOCOL["training"]["epochs"] == 20
    assert not any(BASE_PROTOCOL["locks"].values())
