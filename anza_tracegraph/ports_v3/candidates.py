"""Branch-deduplicated directed candidate proposal."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .branches import Branch
from .junction_ports import junction_arm_ports
from .terminal_ports import Port, terminal_ports
from .valley_ports import confidence_valley_ports
from .virtual_landing import virtual_landing_ports


@dataclass(frozen=True)
class BranchPortCandidate:
    destination_branch_id: int
    landing_point_yx: tuple[float, float]
    landing_tangent_yx: tuple[float, float]
    port_type: str
    distance: float
    source_angle: float
    destination_angle: float
    geometric_score: float
    mean_probability: float


def source_ports(branches: tuple[Branch, ...], probability: np.ndarray) -> tuple[Port, ...]:
    return terminal_ports(branches) + junction_arm_ports(branches) + confidence_valley_ports(branches, probability)


def select_source_port(ports: tuple[Port, ...], query_yx: tuple[float, float], query_tangent_yx: tuple[float, float]) -> Port | None:
    if not ports: return None
    query = np.asarray(query_yx); tangent = np.asarray(query_tangent_yx)
    return min(ports, key=lambda port: (float(np.linalg.norm(np.asarray(port.point_yx) - query)) + 2.0 * (1.0 - max(0.0, float(np.dot(np.asarray(port.tangent_yx), tangent)))), port.branch_id, port.port_type))


def _candidate(source: Port, landing: Port, branch: Branch) -> BranchPortCandidate | None:
    delta = np.asarray(landing.point_yx) - np.asarray(source.point_yx); distance = float(np.linalg.norm(delta))
    if not 6.0 <= distance <= 68.0: return None
    direction = delta / max(distance, 1e-8); source_dot = float(np.dot(np.asarray(source.tangent_yx), direction)); destination_dot = float(np.dot(np.asarray(landing.tangent_yx), -direction))
    if source_dot <= 0.0 or destination_dot <= 0.0: return None
    source_angle = math.acos(float(np.clip(source_dot, -1.0, 1.0))); destination_angle = math.acos(float(np.clip(destination_dot, -1.0, 1.0)))
    if max(source_angle, destination_angle) > math.radians(78.0): return None
    score = distance + 8.0 * max(source_angle, destination_angle)
    return BranchPortCandidate(branch.branch_id, landing.point_yx, landing.tangent_yx, landing.port_type, distance, source_angle, destination_angle, score, branch.mean_probability)


def propose_branch_candidates(source: Port, branches: tuple[Branch, ...], probability: np.ndarray) -> tuple[BranchPortCandidate, ...]:
    explicit = terminal_ports(branches) + junction_arm_ports(branches) + confidence_valley_ports(branches, probability)
    explicit_by_branch: dict[int, list[Port]] = {}
    for port in explicit: explicit_by_branch.setdefault(port.branch_id, []).append(port)
    best: dict[int, BranchPortCandidate] = {}
    for branch in branches:
        if branch.branch_id == source.branch_id: continue
        landings = virtual_landing_ports(branch) + tuple(explicit_by_branch.get(branch.branch_id, ()))
        for landing in landings:
            candidate = _candidate(source, landing, branch)
            if candidate is None: continue
            previous = best.get(branch.branch_id)
            if previous is None or (candidate.geometric_score, -candidate.mean_probability, candidate.port_type) < (previous.geometric_score, -previous.mean_probability, previous.port_type): best[branch.branch_id] = candidate
    return tuple(sorted(best.values(), key=lambda value: (value.geometric_score, -value.mean_probability, value.destination_branch_id)))
