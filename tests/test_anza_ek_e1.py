from __future__ import annotations

import numpy as np
import pytest

from anza_ek.e1_bench import PAIRS_PER_TASK, TASKS, benchmark_config, generate_pair
from anza_ek.kernels import METHODS, deterministic_structure_score, generated_kernel_bank, kernel_feature_vector, local_correlations
from anza_ek.metrics import auroc, matched_ranking, summarize_scores, tpr_at_fpr
from anza_ek.protocol import protocol_payload


def test_e1_has_six_frozen_identifiable_tasks_and_256_pairs_each():
    config = benchmark_config()
    assert len(TASKS) == 6 and config["tasks"] == list(TASKS)
    assert PAIRS_PER_TASK == 256
    assert config["training"] is False and config["classifier"] is False


def test_pair_generation_is_deterministic_and_observably_distinct():
    for task in TASKS:
        first = generate_pair(task, 0)
        second = generate_pair(task, 0)
        assert np.array_equal(first["positive"], second["positive"])
        assert np.array_equal(first["negative"], second["negative"])
        assert first["pixel_equal"] is False
        assert first["l2_difference"] > 1e-6


def test_perturbations_are_present_but_shape_preserving():
    pair = generate_pair(TASKS[0], 0)
    assert pair["positive"].shape == pair["positive_perturbed"].shape == (65, 65)
    assert not np.array_equal(pair["positive"], pair["positive_perturbed"])


@pytest.mark.parametrize("method", METHODS)
def test_fixed_correlation_and_feature_shapes(method):
    pair = generate_pair(TASKS[0], 0)
    kernels = generated_kernel_bank(method, orientation=pair["orientation"])
    correlations = local_correlations(pair["positive"], kernels)
    features = kernel_feature_vector(correlations)
    score = deterministic_structure_score(correlations)
    assert correlations.shape == (7,) and features.shape == (12,)
    assert np.isfinite(score) and np.isfinite(features).all()


def test_matched_ranking_and_auroc_oracles():
    positive = np.asarray([0.8, 0.7, 0.6])
    negative = np.asarray([0.2, 0.3, 0.4])
    assert matched_ranking(positive, negative) == 1.0
    assert auroc(positive, negative) == 1.0


def test_tpr_at_fpr_respects_budget():
    positive = np.linspace(0.5, 1.0, 100)
    negative = np.linspace(0.0, 0.6, 100)
    tpr, fpr, _ = tpr_at_fpr(positive, negative, maximum_fpr=0.05)
    assert fpr <= 0.05 and 0 <= tpr <= 1


def test_score_summary_reports_all_frozen_metrics():
    rows = [
        {"positive_score": 0.8 + index * 0.01, "negative_score": 0.2 + index * 0.01, "positive_perturbed_score": 0.79 + index * 0.01, "negative_perturbed_score": 0.21 + index * 0.01}
        for index in range(20)
    ]
    summary = summarize_scores(rows)
    assert summary["matched_ranking"] == 1.0
    assert summary["tpr_at_fpr05"] >= 0.9
    assert "perturbation_score_correlation" in summary


def test_protocol_freezes_zero_training_gate_and_downstream_locks():
    protocol = protocol_payload()
    assert protocol["methods"] == list(METHODS)
    assert protocol["gate"]["task_gain_tpr_or_ranking"] == 0.08
    assert protocol["gate"]["minimum_passing_tasks"] == 2
    assert protocol["training_performed"] is False and protocol["learned_classifier"] is False
    assert not any(protocol[key] for key in ("E2_opened", "conjugacy_opened", "confirm_created", "cracks_accessed", "expert_accessed"))
