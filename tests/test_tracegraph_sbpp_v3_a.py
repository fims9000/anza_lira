from __future__ import annotations

import numpy as np

from anza_tracegraph.ports_v3.branches import Branch
from anza_tracegraph.ports_v3.metrics import branch_match, wilson_interval
from anza_tracegraph.ports_v3.runner import K_VALUES, PROTOCOL, TAU_CANDIDATES, protocol_hash


def _branch(branch_id: int, y: float, x0: int, x1: int) -> Branch:
    points = np.column_stack((np.full(x1 - x0 + 1, y), np.arange(x0, x1 + 1)))
    return Branch(branch_id, points, 0.8, 0.5, 1.0, False, "endpoint", "endpoint", None, None)


def test_branch_matching_is_independent_of_endpoint_distance():
    truth = _branch(0, 10, 20, 60).points_yx; shifted = _branch(1, 10, 29, 69)
    matched, fraction, _ = branch_match(shifted, truth)
    assert matched and fraction >= 0.60
    assert np.linalg.norm(shifted.points_yx[-1] - truth[-1]) > 6


def test_endpoint_close_wrong_branch_does_not_match_lineage():
    truth = _branch(0, 10, 20, 60).points_yx; wrong = _branch(1, 14, 20, 60)
    matched, _, _ = branch_match(wrong, truth)
    assert not matched
    assert np.linalg.norm(wrong.points_yx[-1] - truth[-1]) <= 6


def test_protocol_has_one_bounded_calibration_parameter_and_frozen_k():
    assert TAU_CANDIDATES == (0.20, 0.25, 0.30, 0.35)
    assert K_VALUES == (4, 8, 12, 16)
    assert PROTOCOL["candidate"]["primary_k"] == 12
    assert len(protocol_hash()) == 64


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(973, 1024)
    assert low < 973 / 1024 < high
