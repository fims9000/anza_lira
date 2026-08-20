import numpy as np

from anza_ks_k2.evaluation import paired_bootstrap_improvement, recall95_threshold, structural_summary


def test_recall95_threshold_and_counts() -> None:
    target = np.linspace(0.0, 1.0, 100)
    negative = np.linspace(0.0, 0.4, 100)
    threshold = recall95_threshold(target)
    summary = structural_summary(target, negative, threshold)
    assert summary["target_recall"] >= 0.95
    assert summary["true_positive_count"] >= 95


def test_paired_bootstrap_uses_scene_events() -> None:
    control = np.ones(100, dtype=np.uint8); candidate = np.zeros(100, dtype=np.uint8)
    result = paired_bootstrap_improvement(control, candidate, resamples=100)
    assert result["mean_improvement"] == 1.0
    assert result["ci95_lower"] == 1.0
