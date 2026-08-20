from __future__ import annotations

import json

import numpy as np
import pytest

from synthetic.crossing_trace_bench import generate_sample
from synthetic.evaluation_corrected import (
    ORIGINAL_TEST_RANGE,
    REPLACEMENT_TEST_RANGE,
    evaluate_sample_corrected,
)
from synthetic.evaluator_audit import (
    audit_gap_predictions,
    false_bridge_verdict,
    freeze_corrected_evaluator,
    run_legacy_test_reanalysis,
    run_replacement_confirmation,
    validate_negative_gap_contract,
)
from synthetic.structural_metrics_corrected import (
    compute_route_metrics,
    geometry_only_minimum_angle_heuristic,
    topology_constrained_assignment,
)


def _row_stochastic_truth(sample: dict) -> np.ndarray:
    truth = np.asarray(sample["continuation_relation_matrix"], dtype=bool)
    eligible = np.asarray(sample["continuation_eligible_matrix"], dtype=bool)
    probability = np.zeros_like(truth, dtype=np.float64)
    for row in range(len(probability)):
        selected = np.flatnonzero(truth[row])
        if len(selected):
            probability[row, selected] = 1.0 / len(selected)
        elif eligible[row].any():
            probability[row, eligible[row]] = 1.0 / int(eligible[row].sum())
    return probability


def test_baseline_route_metrics_are_na_not_geometry_copies() -> None:
    sample = generate_sample("validation", 31, case="x_junction")
    result = evaluate_sample_corrected(sample["visible_fault_mask"], sample)
    assert result["family_b"]["available"] is False
    assert result["family_b"]["route_top1_hit"] is None
    assert result["family_c"] is None


def test_geometry_heuristic_is_separate_and_gt_geometry_conditioned() -> None:
    sample = generate_sample("validation", 32, case="nontrivial_pairing")
    diagnostic = geometry_only_minimum_angle_heuristic(sample)
    assert diagnostic["diagnostic_id"] == "geometry_only_minimum_angle_heuristic"
    assert diagnostic["uses_generator_branch_geometry"] is True
    assert diagnostic["is_model_specific"] is False


def test_x_assignment_is_one_of_three_matchings_and_order_invariant() -> None:
    sample = generate_sample("validation", 33, case="x_junction")
    probabilities = _row_stochastic_truth(sample)
    expected = {tuple(pair) for pair in sample["junctions"][0]["continuation_relation"]}
    predicted = topology_constrained_assignment(probabilities, sample["branch_ids"], "x_crossing", [1, 2, 3, 4])
    assert predicted == expected
    permutation = [2, 0, 3, 1]
    permuted = probabilities[np.ix_(permutation, permutation)]
    branch_ids = [sample["branch_ids"][index] for index in permutation]
    assert topology_constrained_assignment(permuted, branch_ids, "x_crossing", [4, 2, 1, 3]) == expected


def test_t_selects_one_pair_and_y_selects_two_pairs_with_shared_hub() -> None:
    for case, expected_count in (("t_junction", 1), ("y_junction", 2)):
        sample = generate_sample("validation", 34, case=case)
        predicted = topology_constrained_assignment(
            _row_stochastic_truth(sample),
            sample["branch_ids"],
            sample["junctions"][0]["junction_type"],
            sample["junctions"][0]["incident_branch_ids"],
        )
        assert len(predicted) == expected_count
        if case == "y_junction":
            assert len(set.intersection(*(set(pair) for pair in predicted))) == 1


def test_top1_and_mrr_work_when_winning_probability_is_below_half() -> None:
    sample = generate_sample("validation", 35, case="x_junction")
    probability = np.full((4, 4), 0.0)
    eligible = np.asarray(sample["continuation_eligible_matrix"], dtype=bool)
    truth = np.asarray(sample["continuation_relation_matrix"], dtype=bool)
    for row in range(4):
        choices = np.flatnonzero(eligible[row])
        true = int(np.flatnonzero(truth[row])[0])
        probability[row, choices] = 0.30
        probability[row, true] = 0.40
    metrics = compute_route_metrics(probability, sample)
    assert metrics["route_top1_hit"] == 1.0
    assert metrics["route_mrr"] == 1.0
    assert metrics["legacy_threshold_0_5_pair_metric"] == 0.0


def test_multi_positive_row_ap_mass_and_chance_do_not_require_half_probability() -> None:
    sample = generate_sample("validation", 36, case="y_junction")
    target = dict(sample)
    target["branch_ids"] = [1, 2, 3, 4]
    eligible = np.ones((4, 4), dtype=bool)
    np.fill_diagonal(eligible, False)
    truth = np.zeros((4, 4), dtype=bool)
    truth[0, 1] = truth[1, 0] = True
    truth[0, 2] = truth[2, 0] = True
    target["continuation_eligible_matrix"] = eligible
    target["continuation_relation_matrix"] = truth
    probability = np.full((4, 4), 1 / 3, dtype=float)
    np.fill_diagonal(probability, 0.0)
    probability[0] = [0.0, 0.35, 0.35, 0.30]
    metrics = compute_route_metrics(probability, target)
    assert metrics["route_true_probability_mass"] > 0.0
    assert metrics["route_average_precision"] > 0.0
    assert metrics["chance_top1"] == pytest.approx(np.mean([2 / 3, 1 / 3, 1 / 3]))


def test_prediction_assignment_api_has_no_truth_argument() -> None:
    with pytest.raises(TypeError):
        topology_constrained_assignment(np.eye(3), [1, 2, 3], "t_intersection", [1, 2, 3], truth={(1, 2)})


def test_negative_gap_contract_and_fixed_threshold_sensitivity() -> None:
    sample = generate_sample("validation", 37, case="negative_gap")
    assert validate_negative_gap_contract(sample)["status"] == "PASS"
    probability = np.asarray(sample["latent_fault_mask"], dtype=float)
    audit = audit_gap_predictions(sample, probability)
    assert audit["primary_coverage_threshold"] == 0.5
    assert audit["threshold_selection_permitted"] is False
    assert audit["false_bridge_rate_at_fixed_0_5"] == 0.0
    broken = dict(sample)
    broken["negative_gap_masks"] = np.asarray(sample["negative_gap_masks"]).copy()
    broken["negative_gap_masks"][0, 0, 0] = True
    broken["latent_fault_mask"] = np.asarray(sample["latent_fault_mask"]).copy()
    broken["latent_fault_mask"][0, 0] = True
    with pytest.raises(ValueError, match="must not enter"):
        validate_negative_gap_contract(broken)


def test_false_bridge_saturation_is_named_nondiscriminative_not_optimized() -> None:
    verdict = false_bridge_verdict(
        {
            "B0": {"false_bridge_rate_at_fixed_0_5": 1.0},
            "C3": {"false_bridge_rate_at_fixed_0_5": 1.0},
        }
    )
    assert verdict["status"] == "FALSE_BRIDGE_ENDPOINT_SATURATED_NONDISCRIMINATIVE"
    assert verdict["sensitivity_used_for_threshold_selection"] is False
    assert verdict["eligible_for_positive_mechanism_claim"] is False


def test_legacy_outputs_immutable_and_replacement_requires_prior_freeze(tmp_path) -> None:
    study_root = tmp_path / "study"
    legacy = study_root / "synthetic" / "test"
    legacy.mkdir(parents=True)
    original = legacy / "summary.json"
    original.write_text('{"legacy": true}\n')
    result = run_legacy_test_reanalysis(study_root, reanalysis={"corrected": True})
    assert result["status"] == "POSTHOC_REANALYSIS_NOT_CONFIRMATORY"
    assert original.read_text() == '{"legacy": true}\n'
    with pytest.raises(RuntimeError, match="freeze"):
        run_replacement_confirmation(study_root)


def test_replacement_range_is_disjoint_and_confirmation_is_frozen_first(tmp_path) -> None:
    assert ORIGINAL_TEST_RANGE[1] <= REPLACEMENT_TEST_RANGE[0]
    study_root = tmp_path / "study"
    freeze_corrected_evaluator(
        study_root,
        model_checkpoint_hashes={"C3": "abc"},
        visible_thresholds={"C3": 0.4},
    )
    result = run_replacement_confirmation(study_root, confirmation={"sample_count": 2000})
    assert result["status"] == "REPLACEMENT_CONFIRMATION_AFTER_EVALUATOR_AUDIT"
    assert result["indices"] == [2000, 4000]
    assert json.loads((study_root / "synthetic" / "replacement_confirmation" / "freeze.json").read_text())["no_tuning_after_opening"] is True
