"""One deterministic candidate per soft/hard branch cluster."""

from __future__ import annotations

import math

import numpy as np

from anza_tracegraph.ports_v3.candidates import BranchPortCandidate
from anza_tracegraph.ports_v3.terminal_ports import Port
from anza_tracegraph.ports_v3.virtual_landing import virtual_landing_ports

from .clustering import BranchCluster


def _candidate(source: Port, landing: Port, cluster: BranchCluster) -> BranchPortCandidate | None:
    delta = np.asarray(landing.point_yx) - np.asarray(source.point_yx); distance = float(np.linalg.norm(delta))
    if not 6.0 <= distance <= 68.0: return None
    direction = delta / max(distance, 1e-8); source_dot = float(np.dot(np.asarray(source.tangent_yx), direction)); destination_dot = float(np.dot(np.asarray(landing.tangent_yx), -direction))
    if source_dot <= 0.0 or destination_dot <= 0.0: return None
    source_angle = math.acos(float(np.clip(source_dot, -1.0, 1.0))); destination_angle = math.acos(float(np.clip(destination_dot, -1.0, 1.0)))
    if max(source_angle, destination_angle) > math.radians(78.0): return None
    return BranchPortCandidate(cluster.cluster_id, landing.point_yx, landing.tangent_yx, landing.port_type, distance, source_angle, destination_angle, distance + 8.0 * max(source_angle, destination_angle), cluster.mean_probability)


def propose_cluster_candidates(source: Port, clusters: tuple[BranchCluster, ...]) -> tuple[BranchPortCandidate, ...]:
    best: dict[int, BranchPortCandidate] = {}
    for cluster in clusters:
        if source.branch_id in cluster.hard_branch_ids: continue
        for member in cluster.members:
            for landing in virtual_landing_ports(member):
                candidate = _candidate(source, landing, cluster)
                if candidate is None: continue
                previous = best.get(cluster.cluster_id)
                if previous is None or (candidate.geometric_score, -candidate.mean_probability, candidate.landing_point_yx) < (previous.geometric_score, -previous.mean_probability, previous.landing_point_yx): best[cluster.cluster_id] = candidate
    return tuple(sorted(best.values(), key=lambda item: (item.geometric_score, -item.mean_probability, item.destination_branch_id)))
