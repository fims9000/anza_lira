"""Truth-free soft/hard branch clustering."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree

from anza_tracegraph.ports_v3.branches import Branch
from anza_tracegraph.ports_v3.curvature_split import robust_tangents
from .soft_branches import SoftBranch


@dataclass(frozen=True)
class BranchCluster:
    cluster_id: int
    canonical: Branch | SoftBranch
    members: tuple[Branch | SoftBranch, ...]
    hard_branch_ids: tuple[int, ...]

    @property
    def mean_probability(self) -> float: return float(max(member.mean_probability for member in self.members))


def _overlap(first: np.ndarray, second: np.ndarray) -> float:
    a = float(np.mean(cKDTree(second).query(first)[0] <= 2.0)); b = float(np.mean(cKDTree(first).query(second)[0] <= 2.0)); return max(a, b)


def _axial_mismatch(first: np.ndarray, second: np.ndarray) -> float:
    first_tangent = robust_tangents(first); second_tangent = robust_tangents(second); _, nearest = cKDTree(second).query(first)
    dots = np.abs(np.sum(first_tangent * second_tangent[np.asarray(nearest, dtype=int)], axis=1)); return float(np.median(np.arccos(np.clip(dots, 0.0, 1.0))))


def _mergeable(first: Branch | SoftBranch, second: Branch | SoftBranch) -> bool:
    return _overlap(first.points_yx, second.points_yx) >= 0.60 and _axial_mismatch(first.points_yx, second.points_yx) <= math.radians(30.0)


def cluster_branches(hard: tuple[Branch, ...], soft: tuple[SoftBranch, ...]) -> tuple[BranchCluster, ...]:
    clusters: list[BranchCluster] = [BranchCluster(branch.branch_id, branch, (branch,), (branch.branch_id,)) for branch in hard]
    next_id = max((branch.branch_id for branch in hard), default=-1) + 1
    for branch in soft:
        matching = [index for index, cluster in enumerate(clusters) if any(_mergeable(branch, member) for member in cluster.members)]
        if matching:
            index = matching[0]; cluster = clusters[index]; members = cluster.members + (branch,)
            hard_members = [member for member in members if isinstance(member, Branch)]
            canonical = hard_members[0] if hard_members else max(members, key=lambda item: (item.length, item.mean_probability))
            clusters[index] = BranchCluster(cluster.cluster_id, canonical, members, cluster.hard_branch_ids)
        else:
            clusters.append(BranchCluster(next_id, branch, (branch,), ())); next_id += 1
    return tuple(sorted(clusters, key=lambda item: item.cluster_id))
