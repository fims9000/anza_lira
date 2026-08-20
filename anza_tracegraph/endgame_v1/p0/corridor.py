"""Historical six-channel semantics on directed 32x64 branch corridors."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def branch_landing_corridor(
    model_input: np.ndarray,
    probability: np.ndarray,
    source_yx: tuple[float, float],
    destination_yx: tuple[float, float],
    *,
    relation_corridor_x: tuple[int, int] = (35, 50),
    output_hw: tuple[int, int] = (32, 64),
    cross_extent: float = 16.0,
    padding: float = 12.0,
) -> np.ndarray:
    """Return RGB, visible evidence, endpoint markers, and corridor prior."""
    height, width = map(int, output_hw)
    source = np.asarray(source_yx, dtype=np.float64)
    destination = np.asarray(destination_yx, dtype=np.float64)
    vector = destination - source
    distance = float(np.linalg.norm(vector))
    if distance <= 0:
        raise ValueError("source and destination must differ")
    along = vector / distance
    across = np.asarray((-along[1], along[0]))
    midpoint = 0.5 * (source + destination)
    longitudinal_extent = 0.5 * distance + float(padding)
    longitudinal = np.linspace(-longitudinal_extent, longitudinal_extent, width)
    transverse = np.linspace(-float(cross_extent), float(cross_extent), height)
    grid_longitudinal, grid_transverse = np.meshgrid(longitudinal, transverse)
    grid_y = midpoint[0] + along[0] * grid_longitudinal + across[0] * grid_transverse
    grid_x = midpoint[1] + along[1] * grid_longitudinal + across[1] * grid_transverse
    channels = [map_coordinates(np.asarray(channel), (grid_y, grid_x), order=1, mode="reflect") for channel in np.asarray(model_input)]
    visible = np.asarray(probability, dtype=np.float32).copy()
    start, end = relation_corridor_x
    visible[:, int(start) : int(end)] = 0.0
    channels.append(map_coordinates(visible, (grid_y, grid_x), order=1, mode="constant", cval=0.0))
    endpoint_position = distance / (2.0 * longitudinal_extent) * ((width - 1) / 2.0)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    yy, xx = np.mgrid[:height, :width]
    source_marker = np.exp(-((xx - (center_x - endpoint_position)) ** 2 + (yy - center_y) ** 2) / 4.0)
    destination_marker = np.exp(-((xx - (center_x + endpoint_position)) ** 2 + (yy - center_y) ** 2) / 4.0)
    channels.append(np.maximum(source_marker, destination_marker).astype(np.float32))
    channels.append(np.exp(-((yy - center_y) ** 2) / 8.0).astype(np.float32))
    result = np.stack(channels).astype(np.float32)
    if result.shape != (6, height, width) or not np.isfinite(result).all():
        raise AssertionError("invalid P0 corridor")
    return result
