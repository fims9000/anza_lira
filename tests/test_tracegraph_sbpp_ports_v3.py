from __future__ import annotations

import numpy as np

from anza_tracegraph.ports_v3.branches import Branch, extract_branches
from anza_tracegraph.ports_v3.candidates import propose_branch_candidates
from anza_tracegraph.ports_v3.curvature_split import split_at_curvature
from anza_tracegraph.ports_v3.junction_ports import junction_arm_ports
from anza_tracegraph.ports_v3.micro_branches import micro_branches
from anza_tracegraph.ports_v3.terminal_ports import Port, terminal_ports
from anza_tracegraph.ports_v3.valley_ports import confidence_valley_ports
from anza_tracegraph.ports_v3.virtual_landing import virtual_landing_ports


def _branch(branch_id: int, points: np.ndarray, *, start: str = "endpoint", end: str = "endpoint", candidate_only: bool = False) -> Branch:
    return Branch(branch_id, points.astype(float), 0.8, 0.5, 1.0, candidate_only, start, end, 1 if start == "junction" else None, 1 if end == "junction" else None)


def test_terminal_tangent_points_outward_and_is_axially_reversal_invariant():
    points = np.column_stack((np.full(20, 10), np.arange(20)))
    first, last = terminal_ports((_branch(0, points),))
    assert np.dot(first.tangent_yx, (0, -1)) > 0.99 and np.dot(last.tangent_yx, (0, 1)) > 0.99
    reversed_ports = terminal_ports((_branch(0, points[::-1]),))
    assert {tuple(np.round(port.point_yx, 5)) for port in (first, last)} == {tuple(np.round(port.point_yx, 5)) for port in reversed_ports}


def test_junction_creates_one_outward_port_per_incident_arm_and_x_has_four():
    arms = []
    for branch_id, end in enumerate(((10, 20), (10, 0), (20, 10), (0, 10))):
        points = np.linspace((10, 10), end, 12); arms.append(_branch(branch_id, points, start="junction"))
    ports = junction_arm_ports(tuple(arms))
    assert len(ports) == 4
    assert all(np.linalg.norm(np.asarray(port.point_yx) - np.asarray((10, 10))) >= 3.5 for port in ports)


def test_curvature_split_requires_persistent_discontinuity():
    straight = np.column_stack((np.full(20, 10), np.arange(20)))
    assert len(split_at_curvature(straight)) == 1
    corner = np.vstack((np.column_stack((np.full(9, 10), np.arange(9))), np.column_stack((np.arange(11, 21), np.full(10, 8)))))
    assert len(split_at_curvature(corner)) >= 2


def test_virtual_landing_is_in_end_bands_and_branch_deduplicates():
    points = np.column_stack((np.full(40, 10), np.arange(20, 60)))
    branch = _branch(1, points); landings = virtual_landing_ports(branch)
    assert all(min(abs(port.point_yx[1] - 20), abs(port.point_yx[1] - 59)) <= 12.01 for port in landings)
    source = Port(0, (10.0, 10.0), (0.0, 1.0), 0.9, "terminal", -1)
    candidates = propose_branch_candidates(source, (branch,), np.ones((80, 80)))
    assert len(candidates) == 1 and candidates[0].destination_branch_id == 1


def test_directed_compatibility_rejects_away_facing_destination():
    source = Port(0, (10.0, 10.0), (0.0, 1.0), 0.9, "terminal", -1)
    toward = _branch(1, np.column_stack((np.full(20, 10), np.arange(30, 50))))
    away = _branch(2, toward.points_yx[::-1])
    # Both branch geometries are equivalent; virtual end selection still finds exactly one facing landing per branch.
    candidates = propose_branch_candidates(source, (toward, away), np.ones((64, 64)))
    assert {candidate.destination_branch_id for candidate in candidates} == {1, 2}
    assert all(candidate.destination_angle < np.pi / 2 for candidate in candidates)


def test_valley_creates_two_logical_ports_but_flat_profile_creates_none():
    points = np.column_stack((np.full(30, 10), np.arange(10, 40))); branch = _branch(0, points)
    flat = np.full((50, 50), 0.8); assert confidence_valley_ports((branch,), flat) == ()
    valley = flat.copy(); valley[10, 23:27] = 0.2
    ports = confidence_valley_ports((branch,), valley)
    assert len(ports) == 2 and {port.port_type for port in ports} == {"valley_left", "valley_right"}


def test_micro_branch_is_candidate_only_and_normal_extraction_unchanged():
    mask = np.zeros((32, 32), bool); mask[8, 5:11] = True; mask[20, 4:20] = True
    probability = mask.astype(float); branches = extract_branches(mask, probability, probability, tau_micro=0.35)
    micro = micro_branches(branches)
    assert len(micro) == 1 and micro[0].candidate_only and 4 <= micro[0].length < 8
    assert sum(not branch.candidate_only for branch in branches) == 1
