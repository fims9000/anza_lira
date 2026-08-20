import numpy as np

from anza2.eval.low_fpr import low_fpr_metrics, operating_curve, select_threshold


def test_exact_inclusive_fpr_budget_and_perfect_ranking():
    positive = np.array([0.8, 0.9, 1.0])
    negative = np.array([0.1] * 95 + [0.5] * 5)
    threshold = select_threshold(negative, max_fpr=0.05)
    metrics = low_fpr_metrics(positive, negative, max_fpr=0.05)
    assert threshold == 0.5
    assert metrics["fpr"] == 0.05
    assert metrics["tpr_at_fpr_0_05"] == 1.0
    assert metrics["ranking_probability"] == 1.0
    curve = operating_curve(positive, negative)
    assert curve[0]["fpr"] == 0.0 and curve[-1]["fpr"] == 1.0
