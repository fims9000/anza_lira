import numpy as np
import pytest

from cracks_experiment.evaluation import (
    evaluate_binary_section,
    fragmentation,
    hard_cldice,
    verify_threshold_freeze,
)


def test_expert_evaluation_is_locked_without_freeze_receipt(tmp_path) -> None:
    with pytest.raises(PermissionError, match="receipt missing"):
        verify_threshold_freeze(tmp_path)


def test_exact_binary_section_has_perfect_finite_metrics() -> None:
    target = np.zeros((32, 32), dtype=bool)
    target[16, 4:28] = True
    result = evaluate_binary_section(
        target.astype(np.float32),
        target,
        np.ones_like(target),
        0.5,
        orientation_sensitivity_radii=(3, 7),
    )
    assert result["dice"] == pytest.approx(1.0)
    assert result["iou"] == pytest.approx(1.0)
    assert result["cldice"] == pytest.approx(1.0)
    assert result["skeleton_f1_at_2px"] == pytest.approx(1.0)
    assert result["fragmentation"] == pytest.approx(0.0)
    assert result["trace_orientation_error_median_deg"] == pytest.approx(0.0)
    assert result["orientation_error_median_deg_r3"] == pytest.approx(0.0)
    assert result["orientation_error_median_deg_r5"] == pytest.approx(0.0)
    assert result["orientation_error_median_deg_r7"] == pytest.approx(0.0)
    assert np.isfinite(list(result.values())).all()


def test_empty_masks_are_defined_and_fragmentation_counts_splits() -> None:
    empty = np.zeros((24, 24), dtype=bool)
    assert hard_cldice(empty, empty) == pytest.approx(1.0)
    assert fragmentation(empty, empty) == pytest.approx(0.0)
    target = empty.copy()
    target[12, 2:22] = True
    predicted = target.copy()
    predicted[12, 10:14] = False
    assert fragmentation(predicted, target, tolerance=2) == pytest.approx(1.0)
