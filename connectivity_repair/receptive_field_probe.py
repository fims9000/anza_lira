"""Pair-disjoint, capacity-matched receptive-field observability diagnostic."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from synthetic.crossing_trace_bench_v3 import PAIRED_GAP_COUNT
from synthetic.crossing_trace_bench_v5 import generate_sample_v5


RECEPTIVE_FIELDS = (3, 5, 9, 17, 33)
RF_PROBE_PROTOCOL = {
    "version": "connectivity_rf_probe_v2_dense_shared_recurrence",
    "receptive_fields": list(RECEPTIVE_FIELDS),
    "architecture": "shared-weight recurrent dense 3x3 update plus identical 1x1 stem/projection and center readout",
    "parameter_count_equal": True,
    "image_size": 128,
    "crop_size": 64,
    "train": "v5 train matched pair ids 0:96",
    "validation": "v5 validation matched pair ids 96:128",
    "epochs": 60,
    "seed": 42,
    "learning_rate": 0.001,
    "batch_size": 32,
    "selection": "minimum RF with final validation AUROC >=0.80; no best-epoch selection",
    "validation_auroc_gate": 0.80,
    "test_v5": "LOCKED_UNOPENED",
    "cracks": "FORBIDDEN",
    "expert": "FORBIDDEN",
}


def _center_crop(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[-2:]
    y0, x0 = (height - size) // 2, (width - size) // 2
    return np.asarray(image, dtype=np.float32)[..., y0 : y0 + size, x0 : x0 + size]


def probe_arrays(split: str, pair_ids: range, *, image_size: int = 128, crop_size: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[float] = []
    groups: list[int] = []
    for pair_id in pair_ids:
        positive = generate_sample_v5(split, int(pair_id), image_size=image_size)
        negative = generate_sample_v5(split, PAIRED_GAP_COUNT + int(pair_id), image_size=image_size)
        images.extend((_center_crop(positive["image"], crop_size), _center_crop(negative["image"], crop_size)))
        labels.extend((1.0, 0.0))
        # Split-qualified IDs prove that neither a pair nor its render seed is shared.
        group = (0 if split == "train" else 1_000) + int(pair_id)
        groups.extend((group, group))
    return np.stack(images), np.asarray(labels, dtype=np.float32), np.asarray(groups, dtype=np.int64)


class ReceptiveFieldProbe(nn.Module):
    """Dense exact RF with constant parameters and shared recurrent updates."""

    def __init__(self, receptive_field: int) -> None:
        super().__init__()
        if receptive_field not in RECEPTIVE_FIELDS:
            raise ValueError(receptive_field)
        self.receptive_field = int(receptive_field)
        self.steps = (self.receptive_field - 1) // 2
        self.stem = nn.Conv2d(3, 16, 1)
        self.shared_update = nn.Conv2d(16, 16, 3, padding=1)
        self.projection = nn.Conv2d(16, 24, 1)
        self.head = nn.Linear(24, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = nn.functional.gelu(self.stem(image))
        for _ in range(self.steps):
            features = nn.functional.gelu(features + 0.25 * self.shared_update(features))
        features = nn.functional.gelu(self.projection(features))
        y, x = features.shape[-2] // 2, features.shape[-1] // 2
        return self.head(features[:, :, y, x]).squeeze(1)


def parameter_count(receptive_field: int) -> int:
    return sum(parameter.numel() for parameter in ReceptiveFieldProbe(receptive_field).parameters())


def _balanced_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    return float(max(candidates, key=lambda value: (balanced_accuracy_score(labels, probabilities >= value), -value)))


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = len(labels)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        selected = (probabilities >= left) & (probabilities < right if right < 1.0 else probabilities <= right)
        if selected.any():
            value += selected.mean() * abs(float(labels[selected].mean()) - float(probabilities[selected].mean()))
    return float(value if total else 0.0)


def _pairwise_ranking(probabilities: np.ndarray) -> float:
    positive, negative = probabilities[0::2], probabilities[1::2]
    return float(np.mean((positive > negative).astype(float) + 0.5 * (positive == negative)))


def run_rf_probe(*, device: str = "cuda", epochs: int | None = None) -> dict[str, Any]:
    epoch_budget = int(RF_PROBE_PROTOCOL["epochs"] if epochs is None else epochs)
    seed = int(RF_PROBE_PROTOCOL["seed"])
    train_x, train_y, train_groups = probe_arrays("train", range(0, 96))
    validation_x, validation_y, validation_groups = probe_arrays("validation", range(96, 128))
    if set(train_groups.tolist()) & set(validation_groups.tolist()):
        raise AssertionError("RF probe must be pair-disjoint")
    rows: list[dict[str, Any]] = []
    for receptive_field in RECEPTIVE_FIELDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = ReceptiveFieldProbe(receptive_field).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(RF_PROBE_PROTOCOL["learning_rate"]))
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
            batch_size=int(RF_PROBE_PROTOCOL["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        losses: list[float] = []
        for _epoch in range(epoch_budget):
            model.train()
            losses = []
            for images, labels in loader:
                logits = model(images.to(device))
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            train_probability = torch.sigmoid(model(torch.from_numpy(train_x).to(device))).cpu().numpy()
            validation_probability = torch.sigmoid(model(torch.from_numpy(validation_x).to(device))).cpu().numpy()
        threshold = _balanced_threshold(train_y, train_probability)
        rows.append({
            "receptive_field": int(receptive_field),
            "shared_update_steps": (int(receptive_field) - 1) // 2,
            "parameter_count": parameter_count(receptive_field),
            "epochs": epoch_budget,
            "final_train_loss": float(np.mean(losses)),
            "train_frozen_threshold": threshold,
            "validation_auroc": float(roc_auc_score(validation_y, validation_probability)),
            "validation_balanced_auprc": float(average_precision_score(validation_y, validation_probability)),
            "validation_balanced_accuracy": float(balanced_accuracy_score(validation_y, validation_probability >= threshold)),
            "validation_pairwise_ranking": _pairwise_ranking(validation_probability),
            "validation_brier": float(brier_score_loss(validation_y, validation_probability)),
            "validation_ece_10bin": _ece(validation_y, validation_probability),
            "validation_prevalence": float(validation_y.mean()),
        })
    counts = {row["parameter_count"] for row in rows}
    if len(counts) != 1:
        raise AssertionError("RF probes are not capacity matched")
    passing = [row for row in rows if row["validation_auroc"] >= float(RF_PROBE_PROTOCOL["validation_auroc_gate"])]
    minimum = min((int(row["receptive_field"]) for row in passing), default=None)
    return {
        "status": "CONNECTIVITY_CONTEXT_OBSERVABLE" if minimum is not None else "CONNECTIVITY_CONTEXT_NOT_OBSERVABLE",
        "protocol": RF_PROBE_PROTOCOL,
        "rows": rows,
        "minimum_passing_receptive_field": minimum,
        "train_pair_count": 96,
        "validation_pair_count": 32,
        "pair_disjoint": True,
        "test_v5_samples_opened": 0,
        "expert_data_accessed": False,
        "cracks_samples_opened": 0,
    }


def write_rf_probe(output_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    result = run_rf_probe(device=device)
    (output_root / "receptive_field_probe.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output_root / "receptive_field_probe_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([row["receptive_field"] for row in result["rows"]], [row["validation_auroc"] for row in result["rows"]], marker="o", label="AUROC")
    ax.plot([row["receptive_field"] for row in result["rows"]], [row["validation_pairwise_ranking"] for row in result["rows"]], marker="s", label="pairwise ranking")
    ax.axhline(float(RF_PROBE_PROTOCOL["validation_auroc_gate"]), color="black", linestyle="--", linewidth=1, label="frozen AUROC gate")
    ax.set_xscale("log", base=2)
    ax.set_xticks(RECEPTIVE_FIELDS, [str(value) for value in RECEPTIVE_FIELDS])
    ax.set_ylim(0, 1)
    ax.set_xlabel("effective receptive field")
    ax.set_ylabel("validation score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "receptive_field_probe_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_root / "receptive_field_probe_curve.svg", bbox_inches="tight")
    plt.close(fig)
    lines = [
        "# Receptive-field observability probe",
        "",
        f"Status: `{result['status']}`",
        "",
        "All probes have the same parameter count and differ only by the number of shared-weight dense 3x3 recurrence steps, which sets the effective receptive field. The validation set is exactly balanced and pair-disjoint.",
        "",
        "| RF | AUROC | balanced AUPRC | balanced accuracy | pairwise ranking | Brier | ECE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(f"| {row['receptive_field']} | {row['validation_auroc']:.4f} | {row['validation_balanced_auprc']:.4f} | {row['validation_balanced_accuracy']:.4f} | {row['validation_pairwise_ranking']:.4f} | {row['validation_brier']:.4f} | {row['validation_ece_10bin']:.4f} |")
    lines.extend(["", f"Minimum RF passing AUROC >=0.80: `{result['minimum_passing_receptive_field']}`."])
    (output_root / "RECEPTIVE_FIELD_REPORT.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    from synthetic.crossing_trace_bench_v5 import freeze_benchmark_v5_config
    result_root = root / "results" / "connectivity_repair" / "pretraining"
    freeze_benchmark_v5_config(result_root / "benchmark_v5_config.json")
    result = write_rf_probe(result_root, device="cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(result, indent=2, sort_keys=True))
