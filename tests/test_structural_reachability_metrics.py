import numpy as np
import pytest

from structural_reachability.metrics import evaluate_low_fpr_curve, section_paired_bootstrap


def test_low_fpr_metric_and_matched_ranking_are_exact_on_separable_pairs() -> None:
    result = evaluate_low_fpr_curve(
        np.asarray([1, 0, 1, 0]),
        np.asarray([0.9, 0.1, 0.8, 0.2]),
        pair_ids=np.asarray(["a", "a", "b", "b"]),
    )
    assert result["tpr_at_fpr_max"] == pytest.approx(1.0)
    assert result["achieved_fpr"] == pytest.approx(0.0)
    assert result["matched_ranking_probability"] == pytest.approx(1.0)
    assert result["low_fpr_partial_auc_normalized"] == pytest.approx(1.0)


def test_section_bootstrap_resamples_whole_sections_and_is_reproducible() -> None:
    rows = []
    for section in range(1, 5):
        for seed in (41, 42, 43):
            for relation, positive, negative in (("base", 0.6, 0.4), ("candidate", 0.9, 0.1)):
                rows.extend([
                    {"section_id": section, "seed": seed, "relation": relation, "pair_id": f"{section}", "label": 1, "score": positive},
                    {"section_id": section, "seed": seed, "relation": relation, "pair_id": f"{section}", "label": 0, "score": negative},
                ])
    kwargs = dict(
        candidate_relation="candidate",
        baseline_relation="base",
        seeds=(41, 42, 43),
        metric="matched_ranking_probability",
        resamples=100,
        random_seed=7,
    )
    first = section_paired_bootstrap(rows, **kwargs)
    second = section_paired_bootstrap(rows, **kwargs)
    assert first == second
    assert first["resampling_unit"] == "section_id"
    assert first["section_count"] == 4
    assert first["point_delta"] == pytest.approx(0.0)
