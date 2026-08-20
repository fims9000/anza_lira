import numpy as np

from anza_ks.benchmark.matched_generator import TASKS, generate_pair
from anza_ks.constants import FEATURE_WIDTH
from anza_ks.experiments.k0_audit import run_k0_math
from anza_ks.experiments.k1_feature_study import _fit_model, _score, extract_feature
from anza_ks.features import METHODS
from anza_ks.protocol import protocol_payload
from anza_ks.runner import SOURCE_FILES, source_manifest
from anza_ks.stats.low_fpr import threshold_at_fpr, tpr_at_fpr_curve
from anza_ks.stats.paired_bootstrap import bootstrap_macro_ranking_delta


def test_protocol_freezes_readout_gate_and_downstream_locks():
    protocol = protocol_payload()
    assert protocol["version"] == "ANZA_KS_K0_K1_V1"
    assert protocol["methods"] == list(METHODS)
    assert protocol["readout"]["hyperparameter_sweep"] is False
    assert protocol["gate"]["minimum_full_gain_tasks"] == 3
    assert protocol["gate"]["kolmogorov_macro_gain"] == 0.04
    assert not any(protocol[key] for key in ("K2_opened", "confirm_evaluated", "cracks_accessed", "expert_accessed"))


def test_all_k1_methods_have_same_input_width():
    pair = generate_pair(TASKS[0], "dev", 0)
    for method in METHODS:
        feature = extract_feature(pair["positive"], method, pair["orientation"])
        assert feature.shape == (FEATURE_WIDTH,)
        assert np.isfinite(feature).all()


def test_fixed_logistic_readout_and_low_fpr_threshold_are_deterministic():
    rng = np.random.default_rng(4)
    positive = rng.normal(0.5, 1.0, size=(64, FEATURE_WIDTH))
    negative = rng.normal(-0.5, 1.0, size=(64, FEATURE_WIDTH))
    first = _fit_model(positive, negative)
    second = _fit_model(positive, negative)
    assert np.allclose(_score(first, positive), _score(second, positive))
    threshold = threshold_at_fpr(_score(first, negative), 0.05)
    assert np.mean(_score(first, negative) >= threshold) <= 0.05
    tpr, fpr, _ = tpr_at_fpr_curve(_score(first, positive), _score(first, negative), 0.05)
    assert 0 <= tpr <= 1 and fpr <= 0.05


def test_paired_bootstrap_reports_positive_delta_for_better_candidate():
    candidate = {task: np.ones(32) for task in TASKS}
    control = {task: -np.ones(32) for task in TASKS}
    result = bootstrap_macro_ranking_delta(candidate, control, resamples=200, seed=2)
    assert result["observed_macro_ranking_delta"] == 1.0
    assert result["ci95_lower"] > 0


def test_k0_math_passes_without_training():
    result = run_k0_math()
    assert result["status"] == "ANZA_KS_K0_MATH_PASS"
    assert result["training_performed"] is False


def test_source_manifest_covers_isolated_branch_and_scripts():
    assert len(SOURCE_FILES) >= 20
    manifest = source_manifest()
    assert len(manifest["sha256"]) == 64
    assert all(len(row["sha256"]) == 64 for row in manifest["files"])
