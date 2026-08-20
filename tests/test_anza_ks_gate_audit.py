import numpy as np

from scripts.audit_anza_ks_k1_gate import _rowwise_tpr_at_fpr, bootstrap_macro_tpr_delta
from anza_ks.benchmark.matched_generator import TASKS


def test_rowwise_low_fpr_tpr_respects_false_positive_budget():
    positive = np.asarray([[0.9, 0.8, 0.7, 0.6] * 25])
    negative = np.asarray([[0.4, 0.3, 0.2, 0.1] * 25])
    assert _rowwise_tpr_at_fpr(positive, negative, 0.05)[0] == 1.0


def test_bootstrap_detects_macro_tpr_gain():
    candidate = {task: {"positive": np.full(64, 0.9), "negative": np.full(64, 0.1)} for task in TASKS}
    control = {task: {"positive": np.full(64, 0.2), "negative": np.full(64, 0.3)} for task in TASKS}
    result = bootstrap_macro_tpr_delta(candidate, control, resamples=100, seed=3, chunk_size=20)
    assert result["observed_macro_tpr_delta"] == 1.0
    assert result["ci95_lower"] > 0
