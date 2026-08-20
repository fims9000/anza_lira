from __future__ import annotations

from anza_tracegraph.endgame_v1.p0.dataset import STATUS_MISS, STATUS_NONE, STATUS_PRESENT
from anza_tracegraph.endgame_v1.selector.calibration import calibrate_threshold
from anza_tracegraph.endgame_v1.selector.metrics import bootstrap_source_metrics, relation_metrics, source_decisions


def _fixture():
    sources = [
        {"split": "relation_calibration", "index": 0, "stratum": "straight_gap", "positive": 1, "status": STATUS_PRESENT, "candidate_count": 2},
        {"split": "relation_calibration", "index": 1, "stratum": "weak_branch_continue", "positive": 1, "status": STATUS_MISS, "candidate_count": 1},
        {"split": "relation_calibration", "index": 2, "stratum": "none_isolated_end", "positive": 0, "status": STATUS_NONE, "candidate_count": 1},
    ]
    candidates = [
        {"source_index": 0, "candidate_rank": 0, "correct": 1, "score": 0.90},
        {"source_index": 0, "candidate_rank": 1, "correct": 0, "score": 0.10},
        {"source_index": 1, "candidate_rank": 0, "correct": 0, "score": 0.80},
        {"source_index": 2, "candidate_rank": 0, "correct": 0, "score": 0.20},
    ]
    return sources, candidates


def test_one_max_candidate_and_none_threshold_rule() -> None:
    sources, candidates = _fixture()
    decisions = source_decisions(sources, candidates, 0.85)
    assert decisions[0]["correct_accepted"] == 1
    assert decisions[1]["selected_none"] == 1
    assert decisions[2]["selected_none"] == 1


def test_metric_denominators_include_candidate_miss_only_in_rr() -> None:
    sources, candidates = _fixture()
    metrics = relation_metrics(source_decisions(sources, candidates, 0.50))
    assert metrics["CCR"] == 1.0
    assert metrics["RelationRecovery"] == 0.5
    assert metrics["FalseBridge"] == 0.0
    assert metrics["WrongBranch"] == 0.0
    assert metrics["NONERecall"] == 1.0


def test_calibration_selects_one_safe_threshold_from_calibration_rows() -> None:
    sources, candidates = _fixture()
    result = calibrate_threshold(sources, candidates)
    assert result["status"] == "CALIBRATION_FEASIBLE"
    assert result["selected"]["RelationRecovery"] == 0.5
    assert result["selected"]["FalseBridge"] <= 0.02
    assert result["selected"]["WrongBranch"] <= 0.03
    assert isinstance(result["selected"]["threshold"], float)


def test_bootstrap_unit_is_source_not_candidate() -> None:
    sources, candidates = _fixture()
    decisions = source_decisions(sources, candidates, 0.50)
    result = bootstrap_source_metrics(decisions, resamples=100, seed=7)
    assert result["unit"] == "source_scene"
    assert result["resamples"] == 100
    assert set(result["intervals"]) == {"CCR", "RelationRecovery", "FalseBridge", "WrongBranch"}
