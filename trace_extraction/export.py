"""Deterministic fracture trace object and GeoJSON serialization."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .graph import TraceSegment


def _mean_axial_orientation(segment: TraceSegment) -> float:
    angles = [
        math.atan2(y1 - y0, x1 - x0)
        for (y0, x0), (y1, x1) in zip(segment.pixels, segment.pixels[1:])
        if (y0, x0) != (y1, x1)
    ]
    if not angles:
        return 0.0
    return 0.5 * math.atan2(sum(math.sin(2 * value) for value in angles), sum(math.cos(2 * value) for value in angles))


def _sample_mean(array: np.ndarray | None, segment: TraceSegment, default: float) -> float:
    if array is None:
        return default
    values = [float(array[y, x]) for y, x in segment.pixels]
    return float(np.mean(values)) if values else default


def traces_to_geojson(
    segments: Sequence[TraceSegment],
    *,
    source_image_id: str,
    patch_id: str,
    model: str,
    seed: int,
    probability: np.ndarray | None = None,
    coherence: np.ndarray | None = None,
    anisotropy: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
) -> dict:
    features = []
    for segment in sorted(segments, key=lambda item: item.trace_id):
        chord = segment.chord_length
        properties = {
            "trace_id": segment.trace_id,
            "source_image_id": source_image_id,
            "patch_id": patch_id,
            "model": model,
            "seed": int(seed),
            "confidence": _sample_mean(confidence, segment, _sample_mean(probability, segment, 1.0)),
            "orientation_deg": math.degrees(_mean_axial_orientation(segment)) % 180.0,
            "orientation_coherence": _sample_mean(coherence, segment, 1.0),
            "pixel_length": segment.pixel_length,
            "chord_length": chord,
            "sinuosity": segment.pixel_length / chord if chord > 0 else 1.0,
            "anisotropy": _sample_mean(anisotropy, segment, 0.0),
            "mean_probability": _sample_mean(probability, segment, 1.0),
            "start_type": segment.start_type,
            "end_type": segment.end_type,
            "start_border_truncated": segment.start_border_truncated,
            "end_border_truncated": segment.end_border_truncated,
            "border_truncated": segment.start_border_truncated or segment.end_border_truncated,
        }
        if not all(math.isfinite(float(value)) for value in properties.values() if isinstance(value, (int, float))):
            raise ValueError(f"Non-finite GeoJSON properties for trace {segment.trace_id}")
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[int(x), int(y)] for y, x in segment.pixels]},
                "properties": properties,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def write_geojson(path: str | Path, payload: dict) -> None:
    if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
        raise ValueError("Expected a GeoJSON FeatureCollection")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
