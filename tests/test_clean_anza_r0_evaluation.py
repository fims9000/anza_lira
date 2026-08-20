import numpy as np

from cracks_experiment.clean_anza_evaluation import _bootstrap, verify_clean_threshold_freeze


def test_clean_threshold_freeze_is_complete_and_expert_blind() -> None:
    result = verify_clean_threshold_freeze()
    assert result["status"] == "FROZEN"
    assert len(result["runs"]) == 3
    assert {row["section_count"] for row in result["runs"]} == {392}
    assert result["expert_scores_used"] is False


def test_section_bootstrap_is_reproducible_and_not_pixel_bootstrap() -> None:
    values = np.asarray([0.1, 0.2, 0.3, 0.4])
    assert _bootstrap(values, seed=17, resamples=100) == _bootstrap(values, seed=17, resamples=100)
    assert _bootstrap(values, seed=17, resamples=100)[0] == np.mean(values)

