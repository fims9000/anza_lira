from __future__ import annotations

import numpy as np

from synthetic.crossing_trace_bench import generate_sample
from synthetic.structural_metrics import compute_structural_metrics


def _mode_resolved_orientation(sample: dict) -> np.ndarray:
    return 0.5 * np.arctan2(
        np.asarray(sample["branch_tangent_sin2"]),
        np.asarray(sample["branch_tangent_cos2"]),
    )


def test_perfect_observed_and_structural_predictions_have_exact_scores() -> None:
    sample = generate_sample("validation", 120, case="x_junction")
    metrics = compute_structural_metrics(
        sample["visible_fault_mask"],
        sample,
        predicted_completion_mask=sample["latent_fault_mask"],
        predicted_instance_masks=sample["instance_masks"],
        predicted_continuation_scores=sample["continuation_relation_matrix"],
        predicted_orientation=_mode_resolved_orientation(sample),
    )
    for name in (
        "visible_dice",
        "visible_iou",
        "visible_precision",
        "visible_recall",
        "visible_cldice",
        "latent_cldice",
        "latent_skeleton_f1_2px",
        "branch_continuation_f1",
        "branch_pairing_accuracy",
    ):
        assert metrics[name] == 1.0
    assert metrics["false_merge_rate"] == 0.0
    assert metrics["false_split_rate"] == 0.0
    assert metrics["identity_switch_rate"] == 0.0
    assert metrics["orientation_error_median_deg"] < 1e-4
    assert "dice" not in metrics
    assert "iou" not in metrics
    assert "cldice" not in metrics


def test_semantic_crossing_without_instance_routing_is_counted_as_merge() -> None:
    sample = generate_sample("validation", 121, case="x_junction")
    metrics = compute_structural_metrics(
        sample["visible_fault_mask"],
        sample,
        predicted_completion_mask=sample["latent_fault_mask"],
    )
    assert metrics["visible_dice"] == 1.0
    assert metrics["false_merge_rate"] == 1.0
    assert metrics["branch_continuation_f1"] == 0.0


def test_gap_recovery_is_separate_from_visible_segmentation() -> None:
    sample = generate_sample("validation", 122, case="fault_with_gap")
    complete = compute_structural_metrics(
        sample["visible_fault_mask"],
        sample,
        predicted_completion_mask=sample["latent_fault_mask"],
    )
    visible_only = compute_structural_metrics(
        sample["visible_fault_mask"],
        sample,
        predicted_completion_mask=sample["visible_fault_mask"],
    )
    assert complete["visible_dice"] == visible_only["visible_dice"] == 1.0
    assert complete["gap_recovery_rate"] == 1.0
    assert visible_only["gap_recovery_rate"] == 0.0
    assert complete["latent_cldice"] > visible_only["latent_cldice"]


def test_negative_gap_penalizes_unconditional_bridge_closing() -> None:
    sample = generate_sample("validation", 123, case="negative_gap")
    correct = compute_structural_metrics(
        sample["visible_fault_mask"],
        sample,
        predicted_completion_mask=sample["latent_fault_mask"],
    )
    bridged_mask = np.asarray(sample["latent_fault_mask"], dtype=bool) | np.asarray(
        sample["negative_gap_mask"], dtype=bool
    )
    bridged = compute_structural_metrics(
        sample["visible_fault_mask"],
        sample,
        predicted_completion_mask=bridged_mask,
    )
    assert correct["false_bridge_rate"] == 0.0
    assert correct["false_bridge_count"] == 0
    assert bridged["false_bridge_rate"] == 1.0
    assert bridged["false_bridge_count"] == 1
    assert bridged["visible_dice"] == correct["visible_dice"] == 1.0


def test_t_and_y_relations_are_scored_without_forcing_x_pairing() -> None:
    for case in ("t_junction", "y_junction"):
        sample = generate_sample("validation", 124, case=case)
        metrics = compute_structural_metrics(
            sample["visible_fault_mask"],
            sample,
            predicted_completion_mask=sample["latent_fault_mask"],
            predicted_instance_masks=sample["instance_masks"],
            predicted_continuation_scores=sample["continuation_relation_matrix"],
        )
        assert metrics["branch_continuation_f1"] == 1.0
        assert metrics["branch_pairing_count"] == 0
        assert metrics["branch_pairing_accuracy"] == 1.0


def test_abstention_does_not_fake_pairing_and_wrong_pair_counts_per_gt_continuation() -> None:
    sample = generate_sample("validation", 126, case="x_junction")
    absent = np.zeros_like(sample["continuation_relation_matrix"], dtype=np.float32)
    absent_metrics = compute_structural_metrics(
        sample["visible_fault_mask"],
        sample,
        predicted_completion_mask=sample["latent_fault_mask"],
        predicted_continuation_scores=absent,
    )
    wrong = absent.copy()
    wrong[0, 2] = wrong[2, 0] = 1.0
    wrong[1, 3] = wrong[3, 1] = 1.0
    wrong_metrics = compute_structural_metrics(
        sample["visible_fault_mask"],
        sample,
        predicted_completion_mask=sample["latent_fault_mask"],
        predicted_continuation_scores=wrong,
    )
    assert absent_metrics["branch_pairing_accuracy"] == 0.0
    assert absent_metrics["branch_continuation_f1"] == 0.0
    assert absent_metrics["identity_switch_rate"] == 0.0
    assert wrong_metrics["branch_pairing_accuracy"] == 0.0
    assert wrong_metrics["identity_switch_rate"] == 1.0


def test_unmatched_noise_components_do_not_dilute_false_merge_rate() -> None:
    sample = generate_sample("validation", 127, case="x_junction")
    merged = np.any(sample["instance_masks"], axis=0)
    noise = np.zeros_like(merged)
    noise[:2, :2] = True
    metrics = compute_structural_metrics(
        sample["visible_fault_mask"],
        sample,
        predicted_completion_mask=sample["latent_fault_mask"],
        predicted_instance_masks=np.stack([merged, noise]),
        predicted_continuation_scores=sample["continuation_relation_matrix"],
    )
    assert metrics["false_merge_rate"] == 1.0


def test_empty_prediction_metrics_are_finite_and_named_by_task() -> None:
    sample = generate_sample("validation", 125, case="single_straight")
    empty = np.zeros_like(sample["visible_fault_mask"])
    metrics = compute_structural_metrics(empty, sample, predicted_completion_mask=empty)
    assert all(np.isfinite(float(value)) for value in metrics.values())
    assert metrics["visible_dice"] == 0.0
    assert metrics["latent_skeleton_f1_2px"] == 0.0
