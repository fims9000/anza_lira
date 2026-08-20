"""Frozen balanced endpoint-pair classifier for max-min path completion.

This module deliberately does not train a segmentation network.  It asks the
single pre-path question: can observed seismic context distinguish a true gap
from its matched hard negative?
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from sklearn.metrics import balanced_accuracy_score
import torch
import torch.nn as nn

from connectivity_repair.balanced_metrics import balanced_matched_pair_metrics
from path_completion.oracle import PATH_PROTOCOL, freeze_train_geometry
from path_completion.widest_path import EndpointPair, candidate_endpoint_pairs
from synthetic.crossing_trace_bench_v3 import PAIRED_GAP_COUNT
from synthetic.crossing_trace_bench_v5 import benchmark_v5_config, generate_sample_v5


PAIR_PROTOCOL = {
    "version": "anza_endpoint_pair_classifier_v1_frozen",
    "benchmark_sha256": benchmark_v5_config()["sha256"],
    "train": "v5 train matched pair ids 0:128",
    "validation": "v5 validation matched pair ids 0:128",
    "confirm": "LOCKED_UNOPENED",
    "test": "LOCKED_UNOPENED",
    "cracks_expert": "FORBIDDEN",
    "input": "oriented endpoint corridor: RGB, visible mask, endpoint markers, candidate corridor",
    "crop_hw": [33, 49],
    "cross_extent_px": 16.0,
    "longitudinal_padding_px": 8.0,
    "local_encoder_receptive_field_px": 24,
    "architecture": "five-convolution capacity-bounded corridor classifier with global mean/max readout",
    "train_augmentation": "four deterministic longitudinal/cross-axis flips",
    "epochs": 120,
    "batch_pairs": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "ranking_weight": 0.5,
    "ranking_temperature": 0.2,
    "seed": 42,
    "threshold_selection": "maximize train balanced accuracy; validation never selects threshold",
    "validation_auroc_gate": 0.85,
}


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _single_pair(sample: dict[str, Any], d_max: float) -> EndpointPair:
    pairs = candidate_endpoint_pairs(
        sample["visible_fault_mask"],
        d_min=float(PATH_PROTOCOL["d_min_px"]),
        d_max=float(d_max),
    )
    if len(pairs) != 1:
        raise ValueError(f"expected one endpoint pair, found {len(pairs)}")
    return pairs[0]


def oriented_pair_crop(
    sample: dict[str, Any],
    pair: EndpointPair,
    *,
    crop_hw: tuple[int, int] = (33, 49),
    cross_extent: float = 16.0,
    longitudinal_padding: float = 8.0,
) -> np.ndarray:
    """Put the unordered endpoint pair on a common horizontal coordinate frame."""

    height, width = (int(crop_hw[0]), int(crop_hw[1]))
    first = np.asarray(pair.first, dtype=np.float64)
    second = np.asarray(pair.second, dtype=np.float64)
    vector = second - first
    distance = float(np.linalg.norm(vector))
    if distance <= 0:
        raise ValueError("endpoint pair must contain distinct points")
    along = vector / distance
    across = np.asarray([-along[1], along[0]])
    midpoint = 0.5 * (first + second)
    longitudinal_extent = 0.5 * distance + float(longitudinal_padding)
    longitudinal = np.linspace(-longitudinal_extent, longitudinal_extent, width)
    transverse = np.linspace(-float(cross_extent), float(cross_extent), height)
    grid_longitudinal, grid_transverse = np.meshgrid(longitudinal, transverse)
    grid_y = midpoint[0] + along[0] * grid_longitudinal + across[0] * grid_transverse
    grid_x = midpoint[1] + along[1] * grid_longitudinal + across[1] * grid_transverse
    image = np.asarray(sample["image"], dtype=np.float32)
    visible = np.asarray(sample["visible_fault_mask"], dtype=np.float32)
    channels = [
        map_coordinates(channel, (grid_y, grid_x), order=1, mode="reflect")
        for channel in image
    ]
    channels.append(map_coordinates(visible, (grid_y, grid_x), order=0, mode="constant", cval=0.0))
    endpoint_position = distance / (2.0 * longitudinal_extent) * ((width - 1) / 2.0)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    yy, xx = np.mgrid[:height, :width]
    markers = np.maximum(
        np.exp(-((xx - (center_x - endpoint_position)) ** 2 + (yy - center_y) ** 2) / 4.0),
        np.exp(-((xx - (center_x + endpoint_position)) ** 2 + (yy - center_y) ** 2) / 4.0),
    )
    corridor = np.exp(-((yy - center_y) ** 2) / 8.0)
    channels.extend((markers.astype(np.float32), corridor.astype(np.float32)))
    output = np.stack(channels).astype(np.float32)
    if output.shape != (6, height, width) or not np.isfinite(output).all():
        raise AssertionError("invalid oriented pair crop")
    return output


def pair_arrays(
    split: str,
    pair_ids: Iterable[int],
    *,
    d_max: float,
    augment_train: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs: list[np.ndarray] = []
    labels: list[list[float]] = []
    groups: list[int] = []
    for pair_id in pair_ids:
        positive = generate_sample_v5(split, int(pair_id))
        negative = generate_sample_v5(split, PAIRED_GAP_COUNT + int(pair_id))
        positive_crop = oriented_pair_crop(positive, _single_pair(positive, d_max))
        negative_crop = oriented_pair_crop(negative, _single_pair(negative, d_max))
        variants = ((False, False),)
        if augment_train:
            variants = ((False, False), (False, True), (True, False), (True, True))
        for flip_y, flip_x in variants:
            pair = np.stack((positive_crop, negative_crop))
            if flip_y:
                pair = pair[:, :, ::-1, :]
            if flip_x:
                pair = pair[:, :, :, ::-1]
            pairs.append(np.ascontiguousarray(pair))
            labels.append([1.0, 0.0])
            groups.append((0 if split == "train" else 1_000_000) + int(pair_id))
    return (
        np.stack(pairs).astype(np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(groups, dtype=np.int64),
    )


class EndpointPairClassifier(nn.Module):
    """Small corridor classifier; its local encoder RF is fixed before validation."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(192, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(inputs)
        pooled = torch.cat((features.mean(dim=(-2, -1)), features.amax(dim=(-2, -1))), dim=1)
        return self.head(pooled).squeeze(1)


def _frozen_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    return float(max(candidates, key=lambda value: (balanced_accuracy_score(labels, probabilities >= value), -value)))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scores(model: nn.Module, arrays: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    flat = torch.from_numpy(arrays.reshape(-1, *arrays.shape[2:])).to(device)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(flat), 64):
            outputs.append(torch.sigmoid(model(flat[start : start + 64])).cpu())
    return torch.cat(outputs).numpy().reshape(len(arrays), 2)


def run_pair_classifier(
    *,
    device: str = "cuda",
    epochs: int | None = None,
    pair_count: int = PAIRED_GAP_COUNT,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    epoch_budget = int(PAIR_PROTOCOL["epochs"] if epochs is None else epochs)
    frozen_geometry = freeze_train_geometry()
    d_max = float(frozen_geometry["d_max_px"])
    train_x, train_y, train_groups = pair_arrays("train", range(pair_count), d_max=d_max, augment_train=True)
    validation_x, validation_y, validation_groups = pair_arrays("validation", range(pair_count), d_max=d_max)
    if set(train_groups.tolist()) & set(validation_groups.tolist()):
        raise AssertionError("endpoint pair train/validation groups overlap")
    seed = int(PAIR_PROTOCOL["seed"])
    _set_seed(seed)
    torch_device = torch.device(device)
    model = EndpointPairClassifier().to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(PAIR_PROTOCOL["learning_rate"]),
        weight_decay=float(PAIR_PROTOCOL["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    for _epoch in range(epoch_budget):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        losses = []
        for start in range(0, len(order), int(PAIR_PROTOCOL["batch_pairs"])):
            indices = order[start : start + int(PAIR_PROTOCOL["batch_pairs"])]
            batch = torch.from_numpy(train_x[indices.numpy()]).to(torch_device)
            logits = model(batch.flatten(0, 1)).reshape(len(indices), 2)
            targets = torch.from_numpy(train_y[indices.numpy()]).to(torch_device)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
            ranking = nn.functional.softplus(
                -(logits[:, 0] - logits[:, 1]) / float(PAIR_PROTOCOL["ranking_temperature"])
            ).mean()
            loss = bce + float(PAIR_PROTOCOL["ranking_weight"]) * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    train_scores = _scores(model, train_x[::4], torch_device)
    validation_scores = _scores(model, validation_x, torch_device)
    threshold = _frozen_threshold(train_y[::4].reshape(-1), train_scores.reshape(-1))
    train_metrics = balanced_matched_pair_metrics(train_scores[:, 0], train_scores[:, 1], threshold=threshold)
    validation_metrics = balanced_matched_pair_metrics(
        validation_scores[:, 0], validation_scores[:, 1], threshold=threshold
    )
    status = (
        "ENDPOINT_PAIR_CLASSIFIER_PASS"
        if validation_metrics["auroc"] >= float(PAIR_PROTOCOL["validation_auroc_gate"])
        else "ENDPOINT_PAIR_CLASSIFIER_GATE_FAIL"
    )
    result = {
        "status": status,
        "protocol": PAIR_PROTOCOL,
        "protocol_sha256": _canonical_hash(PAIR_PROTOCOL),
        "train_frozen_geometry": frozen_geometry,
        "epochs_completed": epoch_budget,
        "final_train_loss": float(np.mean(losses)),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "threshold_frozen_from_train": threshold,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "train_pair_count": int(pair_count),
        "validation_pair_count": int(pair_count),
        "balanced_pairs": True,
        "pair_disjoint": True,
        "confirm_v5_samples_opened": 0,
        "test_v5_samples_opened": 0,
        "expert_data_accessed": False,
        "cracks_samples_opened": 0,
    }
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "protocol_sha256": result["protocol_sha256"],
        "threshold": threshold,
    }
    return result, payload


def write_pair_classifier(output_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    result, checkpoint = run_pair_classifier(device=device)
    torch.save(checkpoint, output_root / "checkpoint.pt")
    (output_root / "protocol.json").write_text(json.dumps(PAIR_PROTOCOL, indent=2, sort_keys=True) + "\n")
    (output_root / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    metrics_rows = [
        {"split": split, **metrics}
        for split, metrics in (("train", result["train_metrics"]), ("validation", result["validation_metrics"]))
    ]
    with (output_root / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics_rows)
    metrics = result["validation_metrics"]
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["AUROC", "balanced\nAUPRC", "pairwise\nranking", "balanced\naccuracy"]
    values = [metrics["auroc"], metrics["balanced_auprc"], metrics["matched_pair_ranking_probability"], metrics["balanced_accuracy"]]
    ax.bar(labels, values, color=["#355C7D", "#6C5B7B", "#C06C84", "#F67280"])
    ax.axhline(float(PAIR_PROTOCOL["validation_auroc_gate"]), color="black", linestyle="--", linewidth=1, label="AUROC gate")
    ax.set_ylim(0, 1)
    ax.set_ylabel("pair-disjoint validation score")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_root / "fig_pair_classifier.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_root / "fig_pair_classifier.svg", bbox_inches="tight")
    plt.close(fig)
    report = [
        "# Balanced endpoint-pair classifier",
        "",
        f"Status: `{result['status']}`",
        "",
        "This is a balanced, pair-disjoint synthetic validation result. It is not a CRACKS or expert result.",
        "",
        f"- validation AUROC: `{metrics['auroc']:.6f}`",
        f"- balanced AUPRC: `{metrics['balanced_auprc']:.6f}`",
        f"- matched-pair ranking: `{metrics['matched_pair_ranking_probability']:.6f}`",
        f"- balanced accuracy at train-frozen threshold: `{metrics['balanced_accuracy']:.6f}`",
        f"- frozen threshold: `{result['threshold_frozen_from_train']:.6f}`",
        "- v5 confirm/test: `LOCKED_UNOPENED`",
        "- CRACKS expert: `FORBIDDEN_NOT_ACCESSED`",
    ]
    (output_root / "PAIR_CLASSIFIER_REPORT.md").write_text("\n".join(report) + "\n")
    return result

