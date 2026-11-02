"""Exact global endpoint-disjoint selection for accepted CCTA repair proposals.

The local K10/15 relation/identity prediction is frozen. This module only
resolves conflicts *between* overlapping real 3D windows by solving a binary
set-packing problem:

    maximize sum_p w_p * x_p
    subject to sum_{p: e in p} x_p <= 1 for every physical endpoint e.

Utility weights are heuristic ranking scores, NOT calibrated success chances.
The solver fails closed if optimality is not proven. All mask/CT path vetoes
must still run separately after graph selection.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix


@dataclass(frozen=True)
class GlobalProposal:
    scene_id: str
    endpoint_ids: tuple[str, ...]
    weight: float


@dataclass(frozen=True)
class SelectionResult:
    scene_ids: frozenset[str]
    objective: float
    candidate_proposals: int
    selected_proposals: int
    algorithm: str
    optimality_proven: bool


def choose_global_repairs(
    proposals: list[GlobalProposal], *, time_limit_s: float = 10.0
) -> SelectionResult:
    """Use an exact 0/1 MILP with a deterministic small lexical tie-break."""
    if not math.isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError("time_limit_s must be positive")
    if len({x.scene_id for x in proposals}) != len(proposals):
        raise ValueError("Duplicate scene proposal")
    if any(not p.scene_id or not p.endpoint_ids or len(set(p.endpoint_ids)) != len(p.endpoint_ids)
           or any(not e for e in p.endpoint_ids) for p in proposals):
        raise ValueError("Every proposal needs distinct nonblank physical endpoint IDs")
    if any(not math.isfinite(p.weight) or not (0 < p.weight <= 1.0) for p in proposals):
        raise ValueError("Weights must be finite in (0, 1]; no arbitrary overconfident scores")
    if not proposals:
        return SelectionResult(frozenset(), 0.0, 0, 0, "exact_binary_set_packing", True)

    ordered = sorted(proposals, key=lambda p: p.scene_id)
    labels = sorted({e for p in ordered for e in p.endpoint_ids})
    mapping = {label: index for index, label in enumerate(labels)}
    rr, cc = [], []
    for i, p in enumerate(ordered):
        for endpoint in p.endpoint_ids:
            rr.append(mapping[endpoint])
            cc.append(i)
    matrix = csc_matrix(
        (np.ones(len(rr), dtype=np.float64), (rr, cc)),
        shape=(len(labels), len(ordered)),
    )
    # Identity tie-break is far below a meaningful precision of weights.
    # Within an exact tie, keep earlier lexicographically named scenes.
    tie = 1e-10 * (len(ordered) - np.arange(len(ordered))) / len(ordered)
    costs = -(np.asarray([p.weight for p in ordered]) + tie)
    outcome = milp(
        c=costs,
        integrality=np.ones(len(ordered), dtype=np.int8),
        bounds=Bounds(np.zeros(len(ordered)), np.ones(len(ordered))),
        constraints=LinearConstraint(matrix, -np.inf, np.ones(len(labels))),
        options={"time_limit": float(time_limit_s), "mip_rel_gap": 0.0},
    )
    if not outcome.success or outcome.status != 0 or outcome.x is None:
        raise RuntimeError(
            "Global endpoint set-packing did not prove optimum; refusing "
            "greedy replacement or unverified partial repair"
        )
    chosen = [p for p, active in zip(ordered, outcome.x) if active > .5]
    selected_endpoints = [e for proposal in chosen for e in proposal.endpoint_ids]
    if len(selected_endpoints) != len(set(selected_endpoints)):
        raise RuntimeError("Solver returned endpoint-conflicting repairs")
    return SelectionResult(
        frozenset(p.scene_id for p in chosen),
        float(sum(p.weight for p in chosen)), len(proposals), len(chosen),
        "exact_binary_set_packing", True,
    )
