"""Expert-blind full-section validation and threshold freezing for Setting A."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from cracks_experiment.matrix import (
    CRACKSRunSpec,
    PROJECT_ROOT,
    SETTING_A_PROTOCOL,
    setting_a_protocol_hash,
    setting_a_matrix,
)
from cracks_experiment.training import NORMALIZATION, build_real_model, load_real_checkpoint
from datasets.cracks import CRACKSSectionDataset


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _tile_starts(length: int, tile_size: int, overlap: int) -> tuple[int, ...]:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Tiling requires tile_size > overlap >= 0")
    if length <= tile_size:
        return (0,)
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def _blend_window(tile_size: int, overlap: int, device: torch.device) -> torch.Tensor:
    if overlap == 0:
        return torch.ones((1, 1, tile_size, tile_size), device=device)
    ramp = torch.ones(tile_size, dtype=torch.float32, device=device)
    edge = min(overlap, tile_size // 2)
    values = torch.linspace(1.0 / (edge + 1), 1.0, edge, device=device)
    ramp[:edge] = values
    ramp[-edge:] = values.flip(0)
    return (ramp[:, None] * ramp[None, :]).view(1, 1, tile_size, tile_size)


@torch.no_grad()
def tiled_probability(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    tile_size: int = 256,
    overlap: int = 64,
) -> torch.Tensor:
    """Predict one normalized CHW section using deterministic weighted tiles."""
    if image.ndim != 3:
        raise ValueError(f"Expected normalized CHW image, got {tuple(image.shape)}")
    height, width = image.shape[-2:]
    if height < tile_size or width < tile_size:
        raise ValueError("Full-section input must already be padded to at least one tile")
    device = next(model.parameters()).device
    window = _blend_window(tile_size, overlap, device)
    probability_sum = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)
    weight_sum = torch.zeros_like(probability_sum)
    for top in _tile_starts(height, tile_size, overlap):
        for left in _tile_starts(width, tile_size, overlap):
            tile = image[:, top : top + tile_size, left : left + tile_size].unsqueeze(0).to(device)
            probability = torch.sigmoid(model(tile)).float()
            if probability.shape != window.shape:
                raise ValueError(f"Model changed tile shape: {tuple(probability.shape)}")
            probability_sum[:, :, top : top + tile_size, left : left + tile_size] += probability * window
            weight_sum[:, :, top : top + tile_size, left : left + tile_size] += window
    if torch.any(weight_sum <= 0):
        raise AssertionError("Tiled inference left uncovered pixels")
    output = (probability_sum / weight_sum).squeeze(0).squeeze(0).cpu()
    if output.shape != (height, width) or not torch.isfinite(output).all():
        raise ValueError("Tiled inference produced invalid probabilities")
    return output.clamp(0.0, 1.0)


def _binary_metrics(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(prediction, dtype=bool)[valid]
    truth = np.asarray(target, dtype=bool)[valid]
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    tn = int(np.count_nonzero(~pred & ~truth))

    def ratio(numerator: int, denominator: int, empty_value: float) -> float:
        return float(numerator / denominator) if denominator else empty_value

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": ratio(2 * tp, 2 * tp + fp + fn, 1.0),
        "iou": ratio(tp, tp + fp + fn, 1.0),
        "precision": ratio(tp, tp + fp, 1.0 if not truth.any() else 0.0),
        "recall": ratio(tp, tp + fn, 1.0),
    }


def select_threshold(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Select highest macro section Dice; deterministic ties prefer lower threshold."""
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(float(row["threshold"]), []).append(row)
    if not grouped:
        raise ValueError("Threshold selection requires validation rows")
    sweep = []
    for threshold, section_rows in sorted(grouped.items()):
        counts = {name: sum(int(row[name]) for row in section_rows) for name in ("tp", "fp", "fn", "tn")}
        micro = _binary_metrics_from_counts(**counts)
        sweep.append(
            {
                "threshold": threshold,
                "section_count": len(section_rows),
                "macro_dice": float(np.mean([float(row["dice"]) for row in section_rows])),
                "macro_iou": float(np.mean([float(row["iou"]) for row in section_rows])),
                "micro_dice": micro["dice"],
                "micro_iou": micro["iou"],
            }
        )
    selected = max(sweep, key=lambda row: (row["macro_dice"], -row["threshold"]))
    return {"selected_threshold": selected["threshold"], "selection_metric": "macro_section_dice", "sweep": sweep}


def _binary_metrics_from_counts(*, tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    del tn
    return {
        "dice": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 1.0,
        "iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0,
    }


def run_setting_a_validation(
    spec: CRACKSRunSpec,
    training_root: Path,
    *,
    policy: str = "paper_like",
    device: str = "cuda",
    max_sections: int | None = None,
) -> dict[str, Any]:
    """Freeze a threshold using only held-out non-expert crowd annotations."""
    run_dir = Path(training_root) / f"{spec.run_id}-{spec.run_hash}"
    checkpoint_path = run_dir / "checkpoint-last.pt"
    status_path = run_dir / "status.json"
    if not checkpoint_path.exists() or not status_path.exists():
        raise FileNotFoundError(f"Training artifacts missing for {spec.run_id}")
    training_status = json.loads(status_path.read_text())
    if training_status.get("status") != "COMPLETE":
        raise ValueError(f"Cannot validate incomplete training run {spec.run_id}")
    if training_status.get("expert_scores_used") is not False:
        raise ValueError("Expert lock provenance failed before threshold selection")

    checkpoint_sha256 = _sha256(checkpoint_path)
    validation_path = run_dir / "crowd_validation.json"
    rows_path = run_dir / "crowd_validation_sections.csv"
    if validation_path.exists():
        existing = json.loads(validation_path.read_text())
        if (
            existing.get("status") == "COMPLETE"
            and existing.get("checkpoint_sha256") == checkpoint_sha256
            and existing.get("protocol_hash") == setting_a_protocol_hash()
            and existing.get("policy") == policy
            and existing.get("section_limit") == max_sections
        ):
            return {**existing, "action": "SKIP"}

    model = build_real_model(spec).to(torch.device(device))
    load_real_checkpoint(checkpoint_path, spec.run_hash, model)
    model.eval()
    protocol = json.loads((PROJECT_ROOT / "results" / "anza_v2_study" / "protocol.json").read_text())
    section_ids = list(protocol["setting_a"]["held_out_validation_section_ids"])
    if max_sections is not None:
        section_ids = section_ids[: int(max_sections)]
    dataset = CRACKSSectionDataset(
        PROJECT_ROOT / "data" / "cracks" / "images",
        PROJECT_ROOT / "data" / "cracks" / "crowd_targets" / policy / "heldout",
        section_ids,
        mean=NORMALIZATION["mean"],
        std=NORMALIZATION["std"],
    )
    rows: list[dict[str, Any]] = []
    thresholds = tuple(float(value) for value in SETTING_A_PROTOCOL["threshold_candidates"])
    for index in range(len(dataset)):
        batch = dataset[index]
        probability = tiled_probability(model, batch["image"], tile_size=256, overlap=64).numpy()
        height, width = batch["original_hw"]
        probability = probability[:height, :width]
        target = batch["target"][0, :height, :width].numpy() >= 0.5
        valid = batch["valid"][0, :height, :width].numpy().astype(bool)
        for threshold in thresholds:
            metrics = _binary_metrics(probability >= threshold, target, valid)
            rows.append({"section_id": int(batch["section_id"]), "threshold": threshold, **metrics})
        print(
            f"phase=cracks_crowd_validation model={spec.run_id} section={index + 1}/{len(dataset)} "
            "expert=LOCKED status=RUNNING"
        )
    selection = select_threshold(rows)
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "COMPLETE",
        "action": "RUN",
        "run_id": spec.run_id,
        "run_hash": spec.run_hash,
        "protocol_hash": setting_a_protocol_hash(),
        "checkpoint_sha256": checkpoint_sha256,
        "policy": policy,
        "section_limit": max_sections,
        "section_count": len(dataset),
        "expert_scores_used": False,
        "expert_scores": "LOCKED",
        "tile_size": 256,
        "tile_overlap": 64,
        **selection,
    }
    validation_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def freeze_setting_a_thresholds(training_root: Path) -> dict[str, Any]:
    """Create the expert-unlock prerequisite after every full crowd validation."""
    training_root = Path(training_root)
    frozen_runs = []
    for spec in setting_a_matrix():
        run_dir = training_root / f"{spec.run_id}-{spec.run_hash}"
        validation_path = run_dir / "crowd_validation.json"
        if not validation_path.exists():
            raise FileNotFoundError(f"Crowd validation missing for {spec.run_id}")
        result = json.loads(validation_path.read_text())
        required = {
            "status": "COMPLETE",
            "run_hash": spec.run_hash,
            "protocol_hash": setting_a_protocol_hash(),
            "policy": SETTING_A_PROTOCOL["policy"],
            "section_count": int(SETTING_A_PROTOCOL["heldout_annotator_validation_section_count"]),
            "section_limit": None,
            "expert_scores_used": False,
            "expert_scores": "LOCKED",
        }
        mismatches = {key: (result.get(key), value) for key, value in required.items() if result.get(key) != value}
        if mismatches:
            raise ValueError(f"Cannot freeze {spec.run_id}: {mismatches}")
        checkpoint_path = run_dir / "checkpoint-last.pt"
        if _sha256(checkpoint_path) != result.get("checkpoint_sha256"):
            raise ValueError(f"Checkpoint changed after validation for {spec.run_id}")
        frozen_runs.append(
            {
                "run_id": spec.run_id,
                "run_hash": spec.run_hash,
                "checkpoint_sha256": result["checkpoint_sha256"],
                "selected_threshold": float(result["selected_threshold"]),
                "validation_sha256": _sha256(validation_path),
            }
        )
    receipt_core = {
        "status": "FROZEN",
        "protocol_hash": setting_a_protocol_hash(),
        "selection_source": "held-out non-expert crowd annotators only",
        "expert_scores_used": False,
        "runs": frozen_runs,
    }
    encoded = json.dumps(receipt_core, sort_keys=True, separators=(",", ":")).encode()
    receipt = {**receipt_core, "freeze_sha256": hashlib.sha256(encoded).hexdigest()}
    path = training_root / "threshold_freeze.json"
    if path.exists() and json.loads(path.read_text()) != receipt:
        raise ValueError("Existing threshold freeze receipt differs; refusing to overwrite")
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
