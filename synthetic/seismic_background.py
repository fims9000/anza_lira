"""Layered seismic-like rendering with controlled signed-distance fault throw."""

from __future__ import annotations

import numpy as np

from .geometry_generator import GeometrySample


def _instance_points(geometry: GeometrySample) -> list[tuple[np.ndarray, float]]:
    instances: list[tuple[np.ndarray, float]] = []
    for instance_id in sorted({branch.instance_id for branch in geometry.branches}):
        selected = [branch for branch in geometry.branches if branch.instance_id == instance_id]
        points = np.concatenate([branch.points_xy for branch in selected], axis=0)
        throw = float(np.mean([branch.throw for branch in selected]))
        instances.append((points, throw))
    return instances


def _signed_distance_to_points(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    points_xy: np.ndarray,
) -> np.ndarray:
    """Approximate polyline signed distance by its nearest sampled point/tangent."""
    tangents = np.gradient(points_xy, axis=0)
    tangent_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(tangent_norm, 1e-6)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    flat_x = grid_x.reshape(-1)
    flat_y = grid_y.reshape(-1)
    best_distance_sq = np.full(flat_x.shape, np.inf, dtype=np.float32)
    best_signed = np.zeros(flat_x.shape, dtype=np.float32)
    chunk = 32
    for start in range(0, len(points_xy), chunk):
        points = points_xy[start : start + chunk]
        local_normals = normals[start : start + chunk]
        dx = flat_x[:, None] - points[None, :, 0]
        dy = flat_y[:, None] - points[None, :, 1]
        distance_sq = dx * dx + dy * dy
        local_index = np.argmin(distance_sq, axis=1)
        local_best = distance_sq[np.arange(len(flat_x)), local_index]
        improve = local_best < best_distance_sq
        chosen_dx = dx[np.arange(len(flat_x)), local_index]
        chosen_dy = dy[np.arange(len(flat_x)), local_index]
        chosen_normals = local_normals[local_index]
        signed = chosen_dx * chosen_normals[:, 0] + chosen_dy * chosen_normals[:, 1]
        best_distance_sq[improve] = local_best[improve]
        best_signed[improve] = signed[improve]
    return best_signed.reshape(grid_x.shape)


def render_seismic(
    geometry: GeometrySample,
    image_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render a controlled structural image; this is not claimed as an F3 simulator."""
    size = int(image_size)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    warped_y = yy.copy()
    for points, throw in _instance_points(geometry):
        signed_distance = _signed_distance_to_points(xx, yy, points)
        tau = float(rng.uniform(1.2, 2.8))
        warped_y += 0.5 * throw * np.tanh(signed_distance / tau)

    normalized_y = warped_y / max(size - 1, 1)
    normalized_x = xx / max(size - 1, 1)
    image = np.zeros((size, size), dtype=np.float32)
    frequencies = rng.uniform(5.0, 15.0, size=5)
    amplitudes = rng.uniform(0.12, 0.30, size=5)
    phases = rng.uniform(-np.pi, np.pi, size=5)
    phase_slopes = rng.uniform(-1.2, 1.2, size=5)
    for frequency, amplitude, phase, slope in zip(frequencies, amplitudes, phases, phase_slopes):
        curved_phase = phase + slope * normalized_x + 0.35 * np.sin(2.0 * np.pi * normalized_x)
        image += amplitude * np.sin(2.0 * np.pi * frequency * normalized_y + curved_phase)
    image += 0.08 * rng.normal(size=image.shape).astype(np.float32)
    image += 0.04 * np.roll(image, 1, axis=0) - 0.03 * np.roll(image, 2, axis=0)
    low, high = np.percentile(image, [1.0, 99.0])
    image = np.clip((image - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    # Three channels retain amplitude plus two deterministic local contrast views.
    grad_y = np.gradient(image, axis=0)
    grad_x = np.gradient(image, axis=1)
    channels = np.stack(
        [image, np.clip(0.5 + 1.5 * grad_y, 0.0, 1.0), np.clip(0.5 + 1.5 * grad_x, 0.0, 1.0)],
        axis=0,
    )
    return channels.astype(np.float32)
