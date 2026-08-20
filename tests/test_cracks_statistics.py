import numpy as np
import pytest

from cracks_experiment.statistics import bootstrap_mean, paired_section_delta


def test_bootstrap_resamples_section_values_and_is_reproducible() -> None:
    first = bootstrap_mean([1.0, 2.0, 9.0], resamples=100, seed=7)
    second = bootstrap_mean([1.0, 2.0, 9.0], resamples=100, seed=7)
    assert first == second
    assert first["mean"] == pytest.approx(4.0)
    assert first["n_sections"] == 3


def test_paired_delta_requires_same_sections() -> None:
    result = paired_section_delta({1: 0.8, 2: 0.6}, {1: 0.3, 2: 0.4}, resamples=50)
    assert result["mean"] == pytest.approx(0.35)
    assert np.isfinite([result["ci95_low"], result["ci95_high"]]).all()
    with pytest.raises(ValueError, match="identical"):
        paired_section_delta({1: 1.0}, {2: 1.0})
