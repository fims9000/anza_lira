"""Synthetic exact global endpoint set packing tests."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_global import GlobalProposal, choose_global_repairs


def test_global_solution_beats_greedy_confidence():
    result = choose_global_repairs([
        GlobalProposal("large", ("e1", "e2"), .97),
        GlobalProposal("smaller_a", ("e1", "e3"), .67),
        GlobalProposal("smaller_b", ("e2", "e4"), .66),
    ])
    assert result.scene_ids == {"smaller_a", "smaller_b"}
    assert result.objective == pytest.approx(1.33)
    assert result.selected_proposals == 2
    assert result.optimality_proven


def test_independent_repair_same_trunk_can_both_survive():
    result = choose_global_repairs([
        GlobalProposal("a", ("e1", "e2"), .8),
        GlobalProposal("b", ("e3", "e4"), .82),
    ])
    assert result.scene_ids == {"a", "b"}


def test_empty_duplicate_invalid_proposals_refused():
    assert choose_global_repairs([]).scene_ids == frozenset()
    with pytest.raises(ValueError, match="Duplicate scene"):
        choose_global_repairs([
            GlobalProposal("a", ("e1", "e2"), .9),
            GlobalProposal("a", ("e3", "e4"), .8),
        ])
    for bad in (
        GlobalProposal("bad", ("e1", "e1"), .5),
        GlobalProposal("bad", ("e1", "e2"), float("nan")),
        GlobalProposal("bad", (), .5),
        GlobalProposal("bad", ("e1", "e2"), 1.3),
    ):
        with pytest.raises(ValueError):
            choose_global_repairs([bad])


def test_single_conflict_returns_best():
    r = choose_global_repairs([
        GlobalProposal("same_1", ("e1", "e2"), .99),
        GlobalProposal("same_2", ("e2", "e3"), .75),
    ])
    assert r.scene_ids == {"same_1"}
