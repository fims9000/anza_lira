"""Complete CrossingTraceBench-v4 case and target manifest."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import distance_transform_edt

from models.anza2.affinity import LOCAL8_OFFSETS
from synthetic.affinity_targets import build_affinity_targets
from synthetic.crossing_trace_bench_v4 import SPLIT_SIZES_V4, generate_sample_v4

from .visible_mode_targets import target_audit_row


AUDIT_SPLITS = ("train", "validation", "confirm")


def _gap_length(sample: dict[str, Any]) -> float | None:
    values = []
    for gap in sample.get("gaps", []):
        endpoints = np.asarray(gap["endpoint_xy"], dtype=np.float64)
        values.append(float(np.linalg.norm(endpoints[1] - endpoints[0])))
    return float(np.mean(values)) if values else None


def _parallel_separation(sample: dict[str, Any]) -> float | None:
    centerlines = np.asarray(sample["branch_centerlines"], dtype=bool)
    if centerlines.shape[0] < 2:
        return None
    distances = []
    for first in range(centerlines.shape[0]):
        if not centerlines[first].any():
            continue
        field = distance_transform_edt(~centerlines[first])
        for second in range(first + 1, centerlines.shape[0]):
            if centerlines[second].any():
                distances.append(float(field[centerlines[second]].min()))
    return min(distances) if distances else None


def _junction_diameter(sample: dict[str, Any]) -> float | None:
    mask = np.asarray(sample["junction_map"], dtype=bool)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return float(max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)))


def _curvature_dispersion(sample: dict[str, Any]) -> float | None:
    theta = np.asarray(sample["gt_branch_theta"], dtype=np.float64)
    valid = np.asarray(sample["gt_branch_theta_valid"], dtype=bool)
    values = []
    for branch_theta, branch_valid in zip(theta, valid, strict=True):
        selected = branch_theta[branch_valid]
        if selected.size:
            resultant = abs(np.mean(np.exp(2j * selected)))
            values.append(float(1.0 - resultant))
    return float(np.mean(values)) if values else None


def manifest_row(split: str, index: int, *, image_size: int = 64) -> dict[str, Any]:
    sample = generate_sample_v4(split, index, image_size=image_size)
    edges = build_affinity_targets(sample, LOCAL8_OFFSETS)
    mode_count = np.asarray(sample["gt_mode_count"], dtype=np.uint8)
    unique, counts = np.unique(mode_count, return_counts=True)
    target_hist = {str(int(key)): int(value) for key, value in zip(unique, counts, strict=True)}
    centerline_pixels = int(np.asarray(sample["visible_centerline_map"], dtype=bool).sum())
    visible_pixels = int(np.asarray(sample["visible_fault_mask"], dtype=bool).sum())
    target_audit = target_audit_row(sample)
    junction_types = sorted({str(item["junction_type"]) for item in sample.get("junctions", [])})
    gap_types = sorted({str(item["gap_type"]) for item in sample.get("gaps", [])})
    return {
        "split": split,
        "index": int(index),
        "sample_id": f"{split}:{index}",
        "seed": int(sample["seed"]),
        "case": str(sample["case"]),
        "structure_type": str(sample["case"]),
        "gap_status": "+".join(gap_types) if gap_types else "none",
        "junction_type": "+".join(junction_types) if junction_types else "none",
        "branch_count": len(sample["branch_ids"]),
        "instance_count": len(sample["fault_instance_ids"]),
        "visible_support_pixels": visible_pixels,
        "latent_support_pixels": int(np.asarray(sample["latent_fault_mask"], dtype=bool).sum()),
        "positive_gap_pixels": int(np.asarray(sample["positive_gap_mask"], dtype=bool).sum()),
        "negative_gap_pixels": int(np.asarray(sample["negative_gap_mask"], dtype=bool).sum()),
        "gap_length_px": _gap_length(sample),
        "effective_width_px": visible_pixels / max(centerline_pixels, 1),
        "junction_diameter_px": _junction_diameter(sample),
        "parallel_separation_px": _parallel_separation(sample),
        "curvature_dispersion": _curvature_dispersion(sample),
        "target_mode_count_histogram": json.dumps(target_hist, sort_keys=True),
        "max_target_mode_count": int(mode_count.max()),
        "positive_edges": int(np.asarray(edges["affinity_positive"]).sum()),
        "hard_negative_edges": int(np.asarray(edges["affinity_hard_negative"]).sum()),
        **target_audit,
    }


def build_complete_manifest(*, image_size: int = 64) -> list[dict[str, Any]]:
    rows = []
    for split in AUDIT_SPLITS:
        for index in range(SPLIT_SIZES_V4[split]):
            rows.append(manifest_row(split, index, image_size=image_size))
    return rows


def write_manifest(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError("manifest cannot be empty")
    fields = list(values[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(values)


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in AUDIT_SPLITS:
        selected = [row for row in rows if row["split"] == split]
        cases: dict[str, int] = {}
        for row in selected:
            cases[row["case"]] = cases.get(row["case"], 0) + 1
        summary[split] = {
            "count": len(selected),
            "index_min": min(int(row["index"]) for row in selected),
            "index_max": max(int(row["index"]) for row in selected),
            "seed_min": min(int(row["seed"]) for row in selected),
            "seed_max": max(int(row["seed"]) for row in selected),
            "case_counts": dict(sorted(cases.items())),
        }
    seed_sets = {
        split: {int(row["seed"]) for row in rows if row["split"] == split}
        for split in AUDIT_SPLITS
    }
    pairwise_overlap = {
        f"{first}__{second}": len(seed_sets[first] & seed_sets[second])
        for position, first in enumerate(AUDIT_SPLITS)
        for second in AUDIT_SPLITS[position + 1:]
    }
    return {"splits": summary, "seed_overlap": pairwise_overlap}

