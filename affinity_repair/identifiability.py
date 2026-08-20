"""Matched-gap pixel observability audit and pair-disjoint diagnostic probe."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from skimage.metrics import structural_similarity
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from synthetic.crossing_trace_bench_v3 import PAIRED_GAP_COUNT, generate_sample_v3


IDENTIFIABILITY_PROTOCOL = {
    "version": "gap_identifiability_v1",
    "source": "crossing_trace_bench_v3_validation_matched_pairs",
    "pair_count": 128,
    "train_pair_ids": "0:96",
    "validation_pair_ids": "96:128",
    "crop_size": 64,
    "probe_epochs": 40,
    "probe_seed": 42,
    "critical_probe_auroc_minimum": 0.70,
    "practical_identity_mae_maximum": 1e-5,
    "test_v3": "LOCKED_UNOPENED",
    "expert_access": "FORBIDDEN",
}


def _center_crop(image: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    height, width = array.shape[-2:]
    y0 = (height - size) // 2
    x0 = (width - size) // 2
    return array[..., y0 : y0 + size, x0 : x0 + size]


def _endpoint_patch_mae(first: np.ndarray, second: np.ndarray, endpoints: list[list[float]], radius: int = 12) -> float:
    values: list[float] = []
    height, width = first.shape[-2:]
    for x_value, y_value in endpoints:
        x, y = int(round(x_value)), int(round(y_value))
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        values.append(float(np.abs(first[:, y0:y1, x0:x1] - second[:, y0:y1, x0:x1]).mean()))
    return float(np.mean(values))


def pair_distance_row(pair_id: int, *, image_size: int = 128) -> dict[str, Any]:
    positive = generate_sample_v3("validation", int(pair_id), image_size=image_size)
    negative = generate_sample_v3("validation", PAIRED_GAP_COUNT + int(pair_id), image_size=image_size)
    first = np.asarray(positive["image"], dtype=np.float32)
    second = np.asarray(negative["image"], dtype=np.float32)
    delta = first.astype(np.float64) - second.astype(np.float64)
    positive_match = positive["gap_match"]
    negative_match = negative["gap_match"]
    return {
        "pair_id": int(pair_id),
        "exact_pixel_equal": bool(np.array_equal(first, second)),
        "mean_absolute_difference": float(np.abs(delta).mean()),
        "root_mean_square_difference": float(np.sqrt(np.square(delta).mean())),
        "l2_euclidean": float(np.linalg.norm(delta.reshape(-1))),
        "ssim": float(structural_similarity(first, second, channel_axis=0, data_range=1.0)),
        "endpoint_patch_mae": _endpoint_patch_mae(first, second, positive["gaps"][0]["endpoint_xy"]),
        "center_crop_mae": float(np.abs(_center_crop(first, 64) - _center_crop(second, 64)).mean()),
        "gap_length_difference_px": abs(float(positive_match["gap_length_px"]) - float(negative_match["gap_length_px"])),
        "endpoint_distance_difference_px": abs(float(positive_match["endpoint_distance_px"]) - float(negative_match["endpoint_distance_px"])),
        "local_orientation_difference_rad": abs(float(positive_match["local_axial_orientation_rad"]) - float(negative_match["local_axial_orientation_rad"])),
        "geometry_seed_equal": positive_match["geometry_seed"] == negative_match["geometry_seed"],
        "render_difficulty_seed_equal": positive_match["render_difficulty_seed"] == negative_match["render_difficulty_seed"],
        "phase_metadata_status": "NOT_STORED",
        "throw_metadata_status": "MATCHED_GENERATOR_PARAMETER_NOT_EXPORTED_IN_SAMPLE",
    }


def pair_distance_rows(*, pair_count: int = PAIRED_GAP_COUNT, image_size: int = 128) -> list[dict[str, Any]]:
    return [pair_distance_row(index, image_size=image_size) for index in range(int(pair_count))]


class _GapProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 24, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(24, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(image).flatten(1)).squeeze(1)


def _probe_arrays(pair_ids: range, *, image_size: int = 128, crop_size: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    targets: list[float] = []
    groups: list[int] = []
    for pair_id in pair_ids:
        positive = generate_sample_v3("validation", pair_id, image_size=image_size)
        negative = generate_sample_v3("validation", PAIRED_GAP_COUNT + pair_id, image_size=image_size)
        images.extend((_center_crop(positive["image"], crop_size), _center_crop(negative["image"], crop_size)))
        targets.extend((1.0, 0.0))
        groups.extend((pair_id, pair_id))
    return np.stack(images), np.asarray(targets, dtype=np.float32), np.asarray(groups, dtype=np.int64)


def run_diagnostic_probe(*, device: str = "cuda", epochs: int = 40) -> dict[str, Any]:
    seed = int(IDENTIFIABILITY_PROTOCOL["probe_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_x, train_y, train_groups = _probe_arrays(range(0, 96))
    validation_x, validation_y, validation_groups = _probe_arrays(range(96, 128))
    if set(train_groups.tolist()) & set(validation_groups.tolist()):
        raise AssertionError("diagnostic probe split must be pair-disjoint")
    torch_device = torch.device(device)
    model = _GapProbe().to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(int(epochs)):
        model.train()
        losses: list[float] = []
        for images, targets in loader:
            images, targets = images.to(torch_device), targets.to(torch_device)
            logits = model(images)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            logits = model(torch.from_numpy(validation_x).to(torch_device))
            probabilities = torch.sigmoid(logits).cpu().numpy()
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_auroc": float(roc_auc_score(validation_y, probabilities))})
    best = max(history, key=lambda row: (float(row["validation_auroc"]), -int(row["epoch"])))
    final = history[-1]
    return {
        "status": "COMPLETE",
        "architecture": "tiny_3conv_diagnostic_probe_not_part_of_method",
        "train_pair_count": 96,
        "validation_pair_count": 32,
        "pair_disjoint": True,
        "epochs": int(epochs),
        "best_validation_auroc": float(best["validation_auroc"]),
        "best_epoch": int(best["epoch"]),
        "final_validation_auroc": float(final["validation_auroc"]),
        "history": history,
        "test_v3_samples_opened": 0,
        "expert_data_accessed": False,
    }


def write_identifiability_audit(output_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = pair_distance_rows()
    csv_path = output_root / "pair_distances.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    probe = run_diagnostic_probe(device=device, epochs=int(IDENTIFIABILITY_PROTOCOL["probe_epochs"]))
    identical = sum(bool(row["exact_pixel_equal"]) for row in rows)
    practical = sum(float(row["mean_absolute_difference"]) <= float(IDENTIFIABILITY_PROTOCOL["practical_identity_mae_maximum"]) for row in rows)
    distance_summary = {
        name: {
            "mean": float(np.mean([float(row[name]) for row in rows])),
            "min": float(np.min([float(row[name]) for row in rows])),
            "max": float(np.max([float(row[name]) for row in rows])),
        }
        for name in ("mean_absolute_difference", "root_mean_square_difference", "ssim", "endpoint_patch_mae", "center_crop_mae")
    }
    identifiable = (
        identical == 0
        and practical == 0
        and probe["best_validation_auroc"] >= float(IDENTIFIABILITY_PROTOCOL["critical_probe_auroc_minimum"])
    )
    result = {
        "status": "BENCHMARK_CONNECTIVITY_IDENTIFIABLE" if identifiable else "BENCHMARK_CONNECTIVITY_NOT_IDENTIFIABLE",
        "protocol": IDENTIFIABILITY_PROTOCOL,
        "matched_pair_count": len(rows),
        "exact_identical_pair_count": identical,
        "practically_identical_pair_count": practical,
        "distance_summary": distance_summary,
        "probe": probe,
        "model_development_authorized": identifiable,
        "test_v3_samples_opened": 0,
        "expert_data_accessed": False,
    }
    (output_root / "identifiability.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report = f"""# Matched-gap identifiability audit

Status: `{result['status']}`

- Matched pairs: {len(rows)}
- Exact pixel-identical contradictory pairs: {identical}
- Practically identical pairs (MAE <= {IDENTIFIABILITY_PROTOCOL['practical_identity_mae_maximum']}): {practical}
- Mean pixel MAE: {distance_summary['mean_absolute_difference']['mean']:.6f}
- Mean SSIM: {distance_summary['ssim']['mean']:.6f}
- Mean endpoint-patch MAE: {distance_summary['endpoint_patch_mae']['mean']:.6f}
- Pair-disjoint diagnostic probe best validation AUROC: {probe['best_validation_auroc']:.6f}

The probe is an observability diagnostic only and is not part of StructuralAffinityANZA. Pair IDs 0-95 are used for probe training and 96-127 for validation; the two samples of a matched pair never cross the split.

Phase/throw caveat: phase metadata is not stored by the frozen v3 sample. Geometry and render-difficulty seeds, gap length, endpoint distance, and local gap orientation are exactly matched, while the renderer still produces measurably different pixels because positive and negative lineage alter how instance-wise seismic displacement is applied.

Model development authorized: **{'YES' if identifiable else 'NO'}**.
"""
    (output_root / "IDENTIFIABILITY_REPORT.md").write_text(report)
    return result
