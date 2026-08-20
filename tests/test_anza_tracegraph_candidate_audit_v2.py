from __future__ import annotations

import numpy as np

from anza_tracegraph.candidate_audit_v2 import AUDIT_PROTOCOL, K_VALUES, _branch_match, _candidate_pool, _rank
from anza_tracegraph.tracelets import Endpoint, Tracelet


def _tracelet(tracelet_id: int, y: float, x0: float, x1: float) -> Tracelet:
    x = np.linspace(x0, x1, int(abs(x1 - x0)) + 1)
    return Tracelet(tracelet_id, np.column_stack((np.full_like(x, y), x)), 0.9, 1.0)


def test_branch_matching_is_not_endpoint_radius_matching():
    truth = _tracelet(1, 12.0, 30.0, 60.0).points_yx
    longitudinally_shifted = _tracelet(2, 12.0, 39.0, 69.0)
    matched, fraction, _ = _branch_match(longitudinally_shifted, truth)
    assert matched and fraction > 0.6
    assert np.linalg.norm(longitudinally_shifted.points_yx[0] - truth[0]) > 6.0


def test_directed_ports_reject_a_destination_pointing_away():
    source = Endpoint(0, -1, (10.0, 10.0), (0.0, 1.0), 1.0)
    correct = _tracelet(1, 10.0, 50.0, 70.0)
    pool = _candidate_pool(source, (correct,), correct.points_yx)
    near, far = sorted(pool, key=lambda row: row["endpoint"].point_yx[1])
    assert near["eligible"] and near["directed_eligible"]
    assert far["eligible"] and not far["directed_eligible"]


def test_top_k_is_a_view_of_an_unpruned_pool():
    source = Endpoint(0, -1, (10.0, 10.0), (0.0, 1.0), 1.0)
    truth = _tracelet(99, 10.0, 30.0, 60.0).points_yx
    tracelets = tuple(_tracelet(i + 1, 10.0 + i, 30.0 + i, 55.0 + i) for i in range(20))
    ranked = _rank(_candidate_pool(source, tracelets, truth))
    assert len(ranked) > 8
    assert all(len(ranked[:k]) <= k for k in K_VALUES)


def test_taxonomy_priority_is_exhaustive_contract():
    assert len(AUDIT_PROTOCOL["taxonomy_priority"]) == 5
    assert {item[0] for item in AUDIT_PROTOCOL["taxonomy_priority"]} == {"A", "B", "C", "D", "E"}
