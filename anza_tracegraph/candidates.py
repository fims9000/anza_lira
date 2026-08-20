"""Variant-independent endpoint candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .tracelets import Endpoint


@dataclass(frozen=True)
class Candidate:
    endpoint: Endpoint
    distance: float
    tangent_error: float
    geometric_score: float


def axial_error(a: tuple[float, float], b: tuple[float, float]) -> float:
    dot = abs(float(np.dot(np.asarray(a), np.asarray(b))))
    return float(math.acos(np.clip(dot, 0.0, 1.0)))


def generate_candidates(source: Endpoint, destinations: list[Endpoint], *, k_max: int = 8, min_distance: float = 6.0, max_distance: float = 68.0, max_tangent_error: float = math.radians(78.0)) -> tuple[Candidate, ...]:
    output = []
    for endpoint in destinations:
        if endpoint.tracelet_id == source.tracelet_id: continue
        delta = np.asarray(endpoint.point_yx) - np.asarray(source.point_yx); distance = float(np.linalg.norm(delta))
        if not min_distance <= distance <= max_distance: continue
        direction = tuple((delta / max(distance, 1e-8)).tolist())
        error = max(axial_error(source.outgoing_tangent_yx, direction), axial_error(endpoint.outgoing_tangent_yx, direction))
        if error > max_tangent_error: continue
        output.append(Candidate(endpoint, distance, error, distance + 8.0 * error))
    return tuple(sorted(output, key=lambda value: (value.geometric_score, value.endpoint.tracelet_id, value.endpoint.end_index))[:k_max])
