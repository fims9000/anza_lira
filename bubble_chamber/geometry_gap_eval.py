"""Held-out coordinate recovery on actual expert measurements (NO photo segmentation).

No model training or threshold tuning. One interior blinded span per measured
track and projection. Metrics describe only reconstruction of clicked points.
"""
from __future__ import annotations

from collections import defaultdict
import json
import numpy as np

from .visualmetro_import import SCHEMA


def _predict(points: np.ndarray, start: int, width: int) -> dict[str, np.ndarray]:
    n = len(points)
    if not (start >= 2 and width >= 1 and start + width + 1 < n):
        raise ValueError("gap needs two measured neighbors on each side")
    l, r = start - 1, start + width
    ts = np.arange(start, start + width, dtype=float)
    linear = points[l] + (points[r] - points[l]) * ((ts - l) / (r - l))[:, None]
    i = np.asarray([l - 1, l, r, r + 1], dtype=float)
    control = points[i.astype(int)]
    try:
        poly = np.column_stack([np.polyval(np.polyfit(i - l, control[:, axis], 2), ts - l) for axis in range(2)])
    except np.linalg.LinAlgError:
        poly = linear.copy()
    v0 = points[l] - points[l - 1]
    v1 = points[r + 1] - points[r]
    length = r - l
    t = (ts - l) / length
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    hermite = h00[:, None] * points[l] + h10[:, None] * length * v0 + h01[:, None] * points[r] + h11[:, None] * length * v1
    return {"linear": linear, "local_quadratic": poly, "local_hermite": hermite}


def evaluate_geometry(manifest: dict) -> dict:
    if manifest.get("schema") != SCHEMA:
        raise ValueError("invalid visualMetro manifest")
    rows = []
    for frame in manifest["frames"]:
        for track in frame["tracks"]:
            n = track["n_points"]
            if n < 6:
                continue
            width = 2 if n >= 8 else 1
            start = min(max(2, n // 2 - width // 2), n - width - 2)
            for view in (1, 2, 3):
                points = np.asarray(track["measured_pixel_xy"][str(view)], dtype=float)
                expected = points[start:start + width]
                try:
                    predictions = _predict(points, start, width)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                err = {key: float(np.mean(np.linalg.norm(pred - expected, axis=1))) for key, pred in predictions.items()}
                rows.append({"film_id": frame["film_id"], "frame_id": frame["frame_id"], "track_id": track["track_id"],
                             "view": view, "withheld_indices": list(range(start, start + width)),
                             "mean_position_error_px": err})
    if not rows:
        raise ValueError("no eligible tracks")
    methods = ("linear", "local_quadratic", "local_hermite")
    result = {k:{"mean_px":round(float(np.mean([row["mean_position_error_px"][k] for row in rows])),4),
                 "median_px":round(float(np.median([row["mean_position_error_px"][k] for row in rows])),4),
                 "p90_px":round(float(np.quantile([row["mean_position_error_px"][k] for row in rows],.9)),4)} for k in methods}
    return {"status":"ANNOTATED_COORDINATE_GAP_RECOVERY_ONLY_NOT_IMAGE_SEGMENTATION",
            "protocol": "one held-out contiguous span near midpoint per track and view; same span for all methods; no learning; no test tuning",
            "frame_count":len(manifest["frames"]), "evaluation_traces":len(rows),
            "independent_film_count":len({row["film_id"] for row in rows}),
            "per_method":result,"per_trace":rows,"no_real_segmentation_metric":True}
