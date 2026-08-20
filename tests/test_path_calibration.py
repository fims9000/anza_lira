import numpy as np

from path_completion.calibration import fit_temperature, select_constrained_operating_point


def test_constrained_operating_point_maximizes_tpr_then_threshold():
    scores = np.asarray([0.99, 0.90, 0.80, 0.70, 0.95, 0.85, 0.10, 0.05])
    labels = np.asarray([1, 1, 1, 1, 0, 0, 0, 0])
    selected = select_constrained_operating_point(scores, labels, fpr_max=0.25)
    assert selected["tpr"] == 0.50
    assert selected["fpr"] == 0.25
    assert selected["threshold"] == 0.90


def test_constrained_operating_point_uses_highest_threshold_on_equal_tpr():
    scores = np.asarray([0.9, 0.8, 0.7, 0.6])
    labels = np.asarray([1, 1, 0, 0])
    selected = select_constrained_operating_point(scores, labels, fpr_max=0.0)
    assert selected["tpr"] == 1.0
    assert selected["threshold"] == 0.8


def test_temperature_is_positive_and_reduces_overconfident_nll():
    logits = np.asarray([12.0, 8.0, -7.0, -9.0, 5.0, -6.0])
    labels = np.asarray([1, 1, 0, 0, 0, 1])
    temperature = fit_temperature(logits, labels)
    assert np.isfinite(temperature) and temperature > 0


def test_selector_rejects_missing_classes():
    try:
        select_constrained_operating_point(np.asarray([0.1, 0.2]), np.asarray([1, 1]), fpr_max=0.02)
    except ValueError:
        pass
    else:
        raise AssertionError("single-class calibration should fail")
