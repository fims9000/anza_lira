import numpy as np
import pytest

from connectivity_repair.balanced_metrics import balanced_matched_pair_metrics


def test_balanced_metrics_ignore_natural_edge_prevalence() -> None:
    result = balanced_matched_pair_metrics(
        np.asarray([0.9, 0.8, 0.7]), np.asarray([0.1, 0.2, 0.3])
    )
    assert result["prevalence"] == 0.5
    assert result["auroc"] == 1.0
    assert result["balanced_auprc"] == 1.0
    assert result["matched_pair_ranking_probability"] == 1.0


def test_balanced_metrics_ties_receive_half_ranking_credit() -> None:
    result = balanced_matched_pair_metrics(
        np.asarray([0.6, 0.5]), np.asarray([0.2, 0.5])
    )
    assert result["matched_pair_ranking_probability"] == 0.75
    assert result["matched_pair_accuracy"] == 0.5


@pytest.mark.parametrize(
    "positive,negative",
    [([], []), ([0.5], [0.5, 0.4]), ([float("nan")], [0.2]), ([1.1], [0.2])],
)
def test_balanced_metrics_fail_closed(positive, negative) -> None:
    with pytest.raises(ValueError):
        balanced_matched_pair_metrics(np.asarray(positive), np.asarray(negative))

