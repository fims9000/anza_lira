import pytest

import numpy as np

from cracks_experiment.human import annotator_role, section_bootstrap_spearman


def test_annotator_role_is_explicit() -> None:
    assert annotator_role("novice03") == "novice"
    assert annotator_role("practitioner4") == "practitioner"
    assert annotator_role("expert") == "expert"
    with pytest.raises(ValueError, match="Unknown"):
        annotator_role("reviewer")


def test_section_bootstrap_uses_section_rows_and_is_reproducible() -> None:
    x = np.arange(8, dtype=float)
    first = section_bootstrap_spearman(x, x, resamples=100, seed=5)
    second = section_bootstrap_spearman(x, x, resamples=100, seed=5)
    assert first == second
    assert first["spearman_r"] == pytest.approx(1.0)
    assert first["bootstrap_unit"] == "seismic_section"
    assert first["n_sections"] == 8
