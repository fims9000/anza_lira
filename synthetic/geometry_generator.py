"""Exact lineage, topology, and gap geometry for CrossingTraceBench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


GEOMETRY_TYPES = (
    "single_straight",
    "curved_fault",
    "two_crossing_faults",
    "x_junction",
    "t_junction",
    "y_junction",
    "near_parallel",
    "close_non_intersecting",
    "fault_with_gap",
    "negative_gap",
    "different_throws",
    "curved_crossing",
    "asymmetric_crossing",
    "acute_angle_crossing",
    "similar_tangent_crossing",
    "weak_branch_crossing",
    "crossing_near_junction",
    "nontrivial_pairing",
    "short_distractor",
    "multiple_minor_branches",
)


NONTRIVIAL_PAIRING_CASES = (
    "curved_crossing",
    "asymmetric_crossing",
    "acute_angle_crossing",
    "similar_tangent_crossing",
    "weak_branch_crossing",
    "crossing_near_junction",
    "nontrivial_pairing",
)


@dataclass(frozen=True)
class Branch:
    branch_id: int
    instance_id: int
    points_xy: np.ndarray
    throw: float


@dataclass(frozen=True)
class Junction:
    junction_type: str
    point_xy: tuple[float, float]
    incident_branch_ids: tuple[int, ...]
    incident_instance_ids: tuple[int, ...]
    continuation_relation: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Gap:
    gap_type: str
    points_xy: np.ndarray
    latent_instance_id: int | None


@dataclass(frozen=True)
class GeometrySample:
    case: str
    branches: tuple[Branch, ...]
    junctions: tuple[Junction, ...] = ()
    gaps: tuple[Gap, ...] = ()
    strata: tuple[str, ...] = ()


def _curve(points: Sequence[tuple[float, float]], samples: int = 96) -> np.ndarray:
    controls = np.asarray(points, dtype=np.float32)
    if len(controls) == 2:
        t = np.linspace(0.0, 1.0, samples, dtype=np.float32)[:, None]
        return controls[0] * (1.0 - t) + controls[1] * t
    if len(controls) == 3:
        t = np.linspace(0.0, 1.0, samples, dtype=np.float32)[:, None]
        return (1.0 - t) ** 2 * controls[0] + 2.0 * (1.0 - t) * t * controls[1] + t**2 * controls[2]
    raise ValueError("Curves require two or three control points")


def _branch(branch_id: int, instance_id: int, points: Sequence[tuple[float, float]], throw: float = 4.0) -> Branch:
    return Branch(branch_id, instance_id, _curve(points), float(throw))


def _junction(
    junction_type: str,
    point_xy: tuple[float, float],
    branches: Sequence[Branch],
    incident_branch_ids: Sequence[int],
    continuation_relation: Sequence[tuple[int, int]],
) -> Junction:
    by_id = {branch.branch_id: branch for branch in branches}
    incident = tuple(int(value) for value in incident_branch_ids)
    return Junction(
        junction_type=junction_type,
        point_xy=point_xy,
        incident_branch_ids=incident,
        incident_instance_ids=tuple(by_id[value].instance_id for value in incident),
        continuation_relation=tuple((int(first), int(second)) for first, second in continuation_relation),
    )


def _x_geometry(
    case: str,
    center: tuple[float, float],
    branch_controls: Sequence[Sequence[tuple[float, float]]],
    throw1: float,
    throw2: float,
    *,
    strata: tuple[str, ...] = (),
) -> GeometrySample:
    branches = (
        _branch(1, 1, branch_controls[0], throw1),
        _branch(2, 1, branch_controls[1], throw1),
        _branch(3, 2, branch_controls[2], throw2),
        _branch(4, 2, branch_controls[3], throw2),
    )
    junction = _junction("x_crossing", center, branches, (1, 2, 3, 4), ((1, 2), (3, 4)))
    return GeometrySample(case, branches, (junction,), strata=strata)


def generate_geometry(case: str, rng: np.random.Generator) -> GeometrySample:
    """Generate normalized [0,1] geometry; continuation comes from instance lineage."""
    if case not in GEOMETRY_TYPES:
        raise ValueError(f"Unknown CrossingTraceBench case: {case}")
    jitter = lambda scale=0.025: float(rng.uniform(-scale, scale))
    center = (0.5 + jitter(0.02), 0.5 + jitter(0.02))
    cx, cy = center
    throw1 = float(rng.uniform(3.0, 7.0))
    throw2 = float(rng.uniform(2.0, 6.0))

    if case == "single_straight":
        return GeometrySample(case, (_branch(1, 1, ((0.08, 0.82), (0.92, 0.18)), throw1),))
    if case == "curved_fault":
        return GeometrySample(case, (_branch(1, 1, ((0.08, 0.80), (0.42, 0.20), (0.92, 0.35)), throw1),))
    if case in {"two_crossing_faults", "x_junction"}:
        return _x_geometry(
            case,
            center,
            (
                ((0.05, 0.85), center),
                (center, (0.95, 0.15)),
                ((0.08, 0.15), center),
                (center, (0.92, 0.85)),
            ),
            throw1,
            throw2,
        )
    if case == "t_junction":
        branches = (
            _branch(1, 1, ((0.05, cy + 0.03), center), throw1),
            _branch(2, 1, (center, (0.95, cy - 0.03)), throw1),
            _branch(3, 2, ((cx, 0.94), center), throw2),
        )
        junction = _junction("t_intersection", center, branches, (1, 2, 3), ((1, 2),))
        return GeometrySample(case, branches, (junction,))
    if case == "y_junction":
        branches = (
            _branch(1, 1, ((cx, 0.95), center), throw1),
            _branch(2, 1, (center, (0.10, 0.12)), throw1),
            _branch(3, 1, (center, (0.90, 0.12)), throw1),
        )
        junction = _junction("y_branch", center, branches, (1, 2, 3), ((1, 2), (1, 3)))
        return GeometrySample(case, branches, (junction,))
    if case == "near_parallel":
        return GeometrySample(
            case,
            (
                _branch(1, 1, ((0.10, 0.82), (0.90, 0.28)), throw1),
                _branch(2, 2, ((0.08, 0.70), (0.88, 0.16)), throw2),
            ),
        )
    if case == "close_non_intersecting":
        return GeometrySample(
            case,
            (
                _branch(1, 1, ((0.08, 0.72), (0.45, 0.35), (0.92, 0.42)), throw1),
                _branch(2, 2, ((0.08, 0.82), (0.45, 0.45), (0.92, 0.52)), throw2),
            ),
        )
    if case in {"fault_with_gap", "negative_gap"}:
        left = _branch(1, 1, ((0.06, 0.82), (0.45, 0.53)), throw1)
        right_instance = 1 if case == "fault_with_gap" else 2
        right = _branch(2, right_instance, ((0.55, 0.47), (0.94, 0.18)), throw1)
        gap = Gap(
            "positive" if case == "fault_with_gap" else "negative",
            _curve(((0.45, 0.53), (0.55, 0.47)), samples=20),
            1 if case == "fault_with_gap" else None,
        )
        return GeometrySample(case, (left, right), gaps=(gap,), strata=("positive_gap" if case == "fault_with_gap" else "negative_gap",))
    if case == "different_throws":
        return GeometrySample(
            case,
            (
                _branch(1, 1, ((0.12, 0.90), (0.42, 0.08)), throw1),
                _branch(2, 2, ((0.58, 0.90), (0.88, 0.08)), max(1.5, 0.45 * throw2)),
            ),
        )
    if case == "curved_crossing":
        return _x_geometry(
            case,
            center,
            (
                ((0.04, 0.78), (0.25, 0.22), center),
                (center, (0.72, 0.80), (0.96, 0.24)),
                ((0.06, 0.18), (0.30, 0.72), center),
                (center, (0.70, 0.18), (0.94, 0.82)),
            ),
            throw1,
            throw2,
            strata=("nontrivial_pairing", "curved"),
        )
    if case == "asymmetric_crossing":
        return _x_geometry(
            case,
            center,
            (((0.03, 0.91), center), (center, (0.91, 0.32)), ((0.20, 0.04), center), (center, (0.98, 0.68))),
            throw1,
            throw2,
            strata=("nontrivial_pairing", "asymmetric"),
        )
    if case == "acute_angle_crossing":
        return _x_geometry(
            case,
            center,
            (((0.03, 0.72), center), (center, (0.97, 0.28)), ((0.03, 0.60), center), (center, (0.97, 0.40))),
            throw1,
            throw2,
            strata=("nontrivial_pairing", "acute_angle"),
        )
    if case == "similar_tangent_crossing":
        return _x_geometry(
            case,
            center,
            (
                ((0.04, 0.75), (0.30, 0.45), center),
                (center, (0.72, 0.56), (0.96, 0.25)),
                ((0.04, 0.65), (0.30, 0.35), center),
                (center, (0.72, 0.66), (0.96, 0.35)),
            ),
            throw1,
            throw2,
            strata=("nontrivial_pairing", "similar_tangents"),
        )
    if case == "weak_branch_crossing":
        sample = _x_geometry(
            case,
            center,
            (((0.05, 0.85), center), (center, (0.95, 0.15)), ((0.08, 0.15), center), (center, (0.92, 0.85))),
            throw1,
            0.45 * throw2,
            strata=("nontrivial_pairing", "weak_branch"),
        )
        return sample
    if case == "crossing_near_junction":
        base = _x_geometry(
            case,
            center,
            (((0.05, 0.85), center), (center, (0.95, 0.15)), ((0.08, 0.15), center), (center, (0.92, 0.85))),
            throw1,
            throw2,
            strata=("nontrivial_pairing", "near_junction"),
        )
        second_center = (0.72, 0.32)
        # Split the first fault at the nearby attachment so this is an actual
        # three-branch T topology, not a two-ray relation mislabeled as Y.
        branches = (
            base.branches[0],
            _branch(2, 1, (center, second_center), throw1),
            base.branches[2],
            base.branches[3],
            _branch(5, 3, (second_center, (0.92, 0.48)), 0.6 * throw1),
            _branch(6, 1, (second_center, (0.95, 0.15)), throw1),
        )
        second = _junction("t_intersection", second_center, branches, (2, 5, 6), ((2, 6),))
        return GeometrySample(case, branches, base.junctions + (second,), strata=base.strata)
    if case == "nontrivial_pairing":
        # Branches 1 and 3 are locally the straightest pair but belong to
        # different instances. The exact relation remains (1,2) and (3,4).
        return _x_geometry(
            case,
            center,
            (((0.04, cy), center), (center, (0.92, 0.28)), ((0.96, cy), center), (center, (0.08, 0.72))),
            throw1,
            throw2,
            strata=("nontrivial_pairing", "minimum_angle_is_wrong"),
        )
    if case == "short_distractor":
        return GeometrySample(
            case,
            (
                _branch(1, 1, ((0.06, 0.80), (0.94, 0.22)), throw1),
                _branch(2, 2, ((0.68, 0.70), (0.78, 0.62)), 1.5),
            ),
        )
    if case == "multiple_minor_branches":
        branches = (
            _branch(1, 1, ((0.06, 0.85), center), throw1),
            _branch(2, 1, (center, (0.94, 0.18)), throw1),
            _branch(3, 2, (center, (0.78, 0.72)), throw2),
            _branch(4, 3, ((0.30, 0.65), (0.18, 0.32)), 2.0),
        )
        junction = _junction("t_intersection", center, branches, (1, 2, 3), ((1, 2),))
        return GeometrySample(case, branches, (junction,))
    raise AssertionError(case)


def scale_geometry(geometry: GeometrySample, image_size: int) -> GeometrySample:
    scale = float(image_size - 1)
    branches = tuple(
        Branch(branch.branch_id, branch.instance_id, branch.points_xy * scale, branch.throw)
        for branch in geometry.branches
    )
    junctions = tuple(
        Junction(
            junction.junction_type,
            (junction.point_xy[0] * scale, junction.point_xy[1] * scale),
            junction.incident_branch_ids,
            junction.incident_instance_ids,
            junction.continuation_relation,
        )
        for junction in geometry.junctions
    )
    gaps = tuple(Gap(gap.gap_type, gap.points_xy * scale, gap.latent_instance_id) for gap in geometry.gaps)
    return GeometrySample(geometry.case, branches, junctions, gaps, geometry.strata)
