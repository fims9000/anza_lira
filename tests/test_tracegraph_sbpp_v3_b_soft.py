from __future__ import annotations

import numpy as np

from anza_tracegraph.ports_v3.branches import Branch
from anza_tracegraph.ports_v3.terminal_ports import Port
from anza_tracegraph.ports_v3_b.candidates import propose_cluster_candidates
from anza_tracegraph.ports_v3_b.clustering import cluster_branches
from anza_tracegraph.ports_v3_b.soft_branches import extract_soft_branches, source_sector_mask


def _source() -> Port: return Port(0, (32.0, 16.0), (0.0, 1.0), 0.9, "terminal", -1)


def _hard_branch(branch_id: int, y: int, start: int, end: int) -> Branch:
    points = np.column_stack((np.full(end - start, y), np.arange(start, end))).astype(float)
    return Branch(branch_id, points, 0.8, 0.5, 1.0, False, "endpoint", "endpoint", None, None)


def _extract(probability: np.ndarray, tau: float, hard_mask: np.ndarray | None = None):
    hard = np.zeros_like(probability, bool) if hard_mask is None else hard_mask
    return extract_soft_branches(probability, probability, hard, _source(), tau_s=tau)


def test_soft_threshold_changes_support_before_skeletonization():
    probability = np.zeros((64, 80), float); probability[32, 34:54] = 0.29
    assert len(_extract(probability, 0.30)) == 0
    assert len(_extract(probability, 0.25)) >= 1


def test_three_soft_thresholds_can_produce_distinct_branch_sets():
    probability = np.zeros((72, 88), float)
    probability[24, 34:48] = 0.34; probability[36, 38:54] = 0.29; probability[48, 42:60] = 0.24
    counts = [len(_extract(probability, tau)) for tau in (0.30, 0.25, 0.20)]
    assert counts[0] < counts[1] < counts[2]


def test_soft_extraction_never_mutates_hard_mask_or_source_port():
    probability = np.zeros((64, 80), float); probability[32, 34:54] = 0.29
    hard = probability >= 0.35; before = hard.copy(); source = _source()
    extract_soft_branches(probability, probability, hard, source, tau_s=0.25)
    assert np.array_equal(hard, before) and source == _source()


def test_source_sector_rejects_behind_source_clutter():
    mask = source_sector_mask((64, 80), _source())
    assert not mask[32, 8] and mask[32, 32]
    probability = np.zeros((64, 80), float); probability[32, 2:14] = 0.5
    assert _extract(probability, 0.30) == ()


def test_h1_keeps_hard_anchored_soft_extension():
    probability = np.zeros((64, 80), float); probability[32, 30:42] = 0.36; probability[32, 42:49] = 0.26
    hard = probability >= 0.35; branches = _extract(probability, 0.25, hard)
    assert branches and all(branch.hysteresis_rule == "H1_hard_anchored" for branch in branches)


def test_h2_keeps_isolated_coherent_ridge_and_rejects_blob():
    ridge = np.zeros((64, 80), float); ridge[32, 34:54] = 0.30
    branches = _extract(ridge, 0.25); assert branches and all(branch.hysteresis_rule == "H2_self_supported" for branch in branches)
    blob = np.zeros((64, 80), float); blob[25:41, 36:52] = 0.30
    assert _extract(blob, 0.25) == ()


def test_soft_hard_overlap_deduplicates_and_soft_only_has_one_cluster():
    hard = _hard_branch(1, 32, 34, 54); probability = np.zeros((64, 80), float); probability[32, 32:57] = 0.30
    soft = _extract(probability, 0.25, probability >= 0.35)
    clusters = cluster_branches((hard,), soft)
    assert len(clusters) == 1 and clusters[0].hard_branch_ids == (1,)
    soft_only = cluster_branches((), soft)
    assert len(soft_only) == 1 and soft_only[0].hard_branch_ids == ()


def test_cluster_candidate_budget_is_after_deduplication():
    hard = _hard_branch(1, 32, 34, 54); probability = np.zeros((64, 80), float); probability[32, 32:57] = 0.30
    soft = _extract(probability, 0.25); clusters = cluster_branches((hard,), soft)
    candidates = propose_cluster_candidates(_source(), clusters)
    assert len(candidates) <= len(clusters)
    assert len({candidate.destination_branch_id for candidate in candidates}) == len(candidates)
