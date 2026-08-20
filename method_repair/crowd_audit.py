"""Crowd-only target audit for thin, spatially displaced CRACKS traces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

from datasets.cracks import BLUE, GREEN, load_rgb_mask
from method_repair.audit import PROJECT_ROOT


PROTOCOL_PATH = PROJECT_ROOT / "results" / "anza_v2_study" / "protocol.json"
ANNOTATION_ROOT = PROJECT_ROOT / "data" / "cracks" / "annotations"
TARGET_ROOT = PROJECT_ROOT / "data" / "cracks" / "crowd_targets" / "paper_like"
AUDIT_SALT = "method-repair-crowd-only-spatial-audit-v1"


def positive_trace_mask(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb, dtype=np.uint8)
    return np.all(array == BLUE, axis=-1) | np.all(array == GREEN, axis=-1)


def stable_section_sample(section_ids: Sequence[int], *, count: int = 40) -> list[int]:
    ranked = sorted(
        {int(value) for value in section_ids},
        key=lambda value: hashlib.sha256(f"{AUDIT_SALT}:{value}".encode()).hexdigest(),
    )
    return sorted(ranked[: min(int(count), len(ranked))])


def directed_positive_distances(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    source_mask = np.asarray(source, dtype=bool)
    destination_mask = np.asarray(destination, dtype=bool)
    if source_mask.shape != destination_mask.shape:
        raise ValueError("trace masks must have equal shape")
    if not source_mask.any():
        return np.empty(0, dtype=np.float32)
    if not destination_mask.any():
        return np.full(int(source_mask.sum()), np.inf, dtype=np.float32)
    return distance_transform_edt(~destination_mask)[source_mask].astype(np.float32)


def tolerant_pair_statistics(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    first_mask = np.asarray(first, dtype=bool)
    second_mask = np.asarray(second, dtype=bool)
    forward = directed_positive_distances(first_mask, second_mask)
    backward = directed_positive_distances(second_mask, first_mask)
    distances = np.concatenate([forward, backward])
    overlap = int((first_mask & second_mask).sum())
    denominator = int(first_mask.sum() + second_mask.sum())
    finite = distances[np.isfinite(distances)]
    return {
        "first_positive_pixels": int(first_mask.sum()),
        "second_positive_pixels": int(second_mask.sum()),
        "pixel_dice": 2.0 * overlap / denominator if denominator else 1.0,
        "symmetric_distance_count": int(distances.size),
        "finite_distance_fraction": float(np.isfinite(distances).mean()) if distances.size else 1.0,
        "distance_median_px": float(np.median(finite)) if finite.size else 0.0,
        "distance_p90_px": float(np.quantile(finite, 0.9)) if finite.size else 0.0,
        "within_2px_fraction": float((finite <= 2.0).mean()) if finite.size else 1.0,
        "within_5px_fraction": float((finite <= 5.0).mean()) if finite.size else 1.0,
        "displaced_2_to_5px_fraction": float(((finite > 0.0) & (finite <= 5.0)).mean()) if finite.size else 0.0,
    }


def _target_distribution(split: str) -> dict[str, Any]:
    paths = sorted((TARGET_ROOT / split).glob("section_*.npz"))
    bins = np.asarray([0.0, 1e-6, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.000001])
    histogram = np.zeros(len(bins) - 1, dtype=np.int64)
    positive_mass = []
    positive_fraction_at_half = []
    for path in paths:
        with np.load(path) as payload:
            target = payload["target"].astype(np.float32)
            valid = payload["valid"].astype(bool)
        values = target[valid]
        histogram += np.histogram(values, bins=bins)[0]
        positive_mass.append(float(values.mean()) if values.size else 0.0)
        positive_fraction_at_half.append(float((values >= 0.5).mean()) if values.size else 0.0)
    return {
        "section_count": len(paths),
        "histogram_edges": bins.tolist(),
        "histogram_counts": histogram.tolist(),
        "section_positive_mass_mean": float(np.mean(positive_mass)),
        "section_positive_mass_median": float(np.median(positive_mass)),
        "section_positive_fraction_at_0_5_mean": float(np.mean(positive_fraction_at_half)),
        "section_positive_fraction_at_0_5_median": float(np.median(positive_fraction_at_half)),
    }


def run_crowd_target_audit(output_path: Path, *, sample_count: int = 40) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    heldout = protocol["setting_a"]["held_out_annotators"]
    annotators = list(heldout["all"])
    if "expert" in annotators:
        raise ValueError("expert annotations are forbidden in crowd target audit")
    available = []
    for section_id in protocol["setting_a"]["held_out_validation_section_ids"]:
        if all((ANNOTATION_ROOT / name / f"section_{int(section_id):03d}.png").is_file() for name in annotators):
            available.append(int(section_id))
    selected = stable_section_sample(available, count=sample_count)
    pair_rows: list[dict[str, Any]] = []
    agreement_counts = np.zeros(len(annotators) + 1, dtype=np.int64)
    positive_fractions: dict[str, list[float]] = {name: [] for name in annotators}
    for section_id in selected:
        masks: dict[str, np.ndarray] = {}
        for name in annotators:
            rgb = load_rgb_mask(ANNOTATION_ROOT / name / f"section_{section_id:03d}.png")
            masks[name] = positive_trace_mask(rgb)
            positive_fractions[name].append(float(masks[name].mean()))
        count_map = sum(mask.astype(np.uint8) for mask in masks.values())
        agreement_counts += np.bincount(count_map.ravel(), minlength=len(annotators) + 1)
        for first_index, first in enumerate(annotators):
            for second in annotators[first_index + 1 :]:
                pair_rows.append({
                    "section_id": section_id,
                    "annotator_a": first,
                    "annotator_b": second,
                    "expertise_pair": "practitioner_novice"
                    if (first.startswith("practitioner") != second.startswith("practitioner"))
                    else "novice_novice",
                    **tolerant_pair_statistics(masks[first], masks[second]),
                })
    finite_rows = [row for row in pair_rows if row["finite_distance_fraction"] > 0]
    payload = {
        "status": "CROWD_ONLY_TARGET_AUDIT_COMPLETE",
        "expert_data_accessed": False,
        "selection": {
            "salt": AUDIT_SALT,
            "available_all_three_heldout": len(available),
            "sample_count": len(selected),
            "section_ids": selected,
        },
        "annotators": annotators,
        "paper_like_target_distribution": {
            "train": _target_distribution("train"),
            "heldout": _target_distribution("heldout"),
        },
        "agreement_count_pixels": agreement_counts.tolist(),
        "positive_fraction_by_annotator": {
            name: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
            }
            for name, values in positive_fractions.items()
        },
        "pair_summary": {
            "pair_section_rows": len(pair_rows),
            "pixel_dice_mean": float(np.mean([row["pixel_dice"] for row in pair_rows])),
            "distance_median_px_median": float(np.median([row["distance_median_px"] for row in finite_rows])),
            "distance_p90_px_median": float(np.median([row["distance_p90_px"] for row in finite_rows])),
            "within_5px_fraction_mean": float(np.mean([row["within_5px_fraction"] for row in finite_rows])),
            "displaced_2_to_5px_fraction_mean": float(np.mean([row["displaced_2_to_5px_fraction"] for row in finite_rows])),
            "practitioner_novice_pixel_dice_mean": float(np.mean([
                row["pixel_dice"] for row in pair_rows if row["expertise_pair"] == "practitioner_novice"
            ])),
            "novice_novice_pixel_dice_mean": float(np.mean([
                row["pixel_dice"] for row in pair_rows if row["expertise_pair"] == "novice_novice"
            ])),
        },
        "pair_rows": pair_rows,
        "interpretation_guard": "DISTANCE_AUDIT_ONLY_DOES_NOT_SELECT_ARCHITECTURE_OR_READ_EXPERT",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
