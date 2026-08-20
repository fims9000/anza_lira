"""Validation-only operating-point calibration for the frozen pair classifier."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
import torch

from path_completion.learned_confirm import load_frozen_pair_classifier
from path_completion.pair_classifier import pair_arrays
from synthetic.crossing_trace_bench_v3 import PAIRED_GAP_COUNT


CALIBRATION_PROTOCOL = {
    "version": "anza_path_validation_calibration_v1_frozen",
    "classifier_weights": "UNCHANGED",
    "calibration_stream": "CrossingTraceBench-v5 validation pairs 0:128",
    "temperature_fit": "single positive scalar minimizing balanced validation binary NLL",
    "operating_point_rule": "maximize TPR subject to FPR <= 0.02; ties choose highest threshold",
    "fpr_max": 0.02,
    "primary_space": "temperature_scaled_probability",
    "raw_logit_operating_point_also_reported": True,
    "v5_test": "LOCKED_UNOPENED",
    "old_confirm": "DEVELOPMENT_EVIDENCE_NOT_USED",
    "expert": "FORBIDDEN",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    if values.shape != truth.shape or not len(values) or not np.isfinite(values).all():
        raise ValueError("temperature fitting requires matching finite logits and labels")

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(values / temperature, -50.0, 50.0)))
        return float(log_loss(truth, probabilities, labels=[0.0, 1.0]))

    result = minimize_scalar(objective, bounds=(-6.0, 6.0), method="bounded", options={"xatol": 1e-10})
    if not result.success:
        raise RuntimeError(f"temperature scaling failed: {result.message}")
    return float(np.exp(result.x))


def select_constrained_operating_point(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    fpr_max: float,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=bool).reshape(-1)
    if values.shape != truth.shape or not truth.any() or truth.all() or not np.isfinite(values).all():
        raise ValueError("operating-point selection requires finite binary-class scores")
    candidates = np.unique(np.concatenate(([np.nextafter(values.max(), np.inf)], values)))
    rows = []
    for threshold in candidates:
        prediction = values >= threshold
        tp = int(np.count_nonzero(prediction & truth))
        fp = int(np.count_nonzero(prediction & ~truth))
        positives = int(truth.sum())
        negatives = int((~truth).sum())
        rows.append({
            "threshold": float(threshold),
            "tpr": float(tp / positives),
            "fpr": float(fp / negatives),
            "tp": tp,
            "fp": fp,
            "positive_count": positives,
            "negative_count": negatives,
        })
    eligible = [row for row in rows if row["fpr"] <= float(fpr_max) + 1e-12]
    if not eligible:
        raise AssertionError("no operating point satisfies the FPR constraint")
    selected = max(eligible, key=lambda row: (row["tpr"], row["threshold"]))
    return {**selected, "rule": CALIBRATION_PROTOCOL["operating_point_rule"]}


def _validation_logits(project_root: Path, *, device: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    torch_device = torch.device(device)
    model, frozen = load_frozen_pair_classifier(project_root, torch_device)
    arrays, labels, groups = pair_arrays(
        "validation",
        range(PAIRED_GAP_COUNT),
        d_max=float(frozen["d_max_px"]),
        augment_train=False,
    )
    if len(np.unique(groups)) != PAIRED_GAP_COUNT:
        raise AssertionError("validation pair groups are not unique")
    flat = torch.from_numpy(arrays.reshape(-1, *arrays.shape[2:])).to(torch_device)
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(flat), 64):
            chunks.append(model(flat[start : start + 64]).cpu())
    return torch.cat(chunks).numpy().reshape(PAIRED_GAP_COUNT, 2), labels, frozen


def run_validation_calibration(project_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    root = Path(project_root)
    checkpoint = root / "results/path_completion/pair_classifier/checkpoint.pt"
    checkpoint_before = _sha256(checkpoint)
    logits, pair_labels, frozen = _validation_logits(root, device=device)
    labels = pair_labels.reshape(-1)
    flat_logits = logits.reshape(-1).astype(np.float64)
    raw_probabilities = 1.0 / (1.0 + np.exp(-np.clip(flat_logits, -50.0, 50.0)))
    temperature = fit_temperature(flat_logits, labels)
    calibrated = 1.0 / (1.0 + np.exp(-np.clip(flat_logits / temperature, -50.0, 50.0)))
    raw_logit_point = select_constrained_operating_point(
        flat_logits, labels, fpr_max=float(CALIBRATION_PROTOCOL["fpr_max"])
    )
    raw_probability_point = select_constrained_operating_point(
        raw_probabilities, labels, fpr_max=float(CALIBRATION_PROTOCOL["fpr_max"])
    )
    calibrated_point = select_constrained_operating_point(
        calibrated, labels, fpr_max=float(CALIBRATION_PROTOCOL["fpr_max"])
    )
    checkpoint_after = _sha256(checkpoint)
    if checkpoint_before != checkpoint_after or checkpoint_before != frozen["checkpoint_sha256"]:
        raise PermissionError("calibration modified or mismatched the frozen classifier checkpoint")
    rows = []
    for pair_id in range(PAIRED_GAP_COUNT):
        for class_index, case in enumerate(("fault_with_gap", "negative_gap")):
            rows.append({
                "pair_id": pair_id,
                "case": case,
                "label": int(pair_labels[pair_id, class_index]),
                "raw_logit": float(logits[pair_id, class_index]),
                "raw_probability": float(raw_probabilities.reshape(PAIRED_GAP_COUNT, 2)[pair_id, class_index]),
                "calibrated_probability": float(calibrated.reshape(PAIRED_GAP_COUNT, 2)[pair_id, class_index]),
            })
    core = {
        "status": "CALIBRATION_FROZEN",
        "protocol": CALIBRATION_PROTOCOL,
        "protocol_sha256": _canonical_hash(CALIBRATION_PROTOCOL),
        "classifier_checkpoint_sha256": checkpoint_before,
        "classifier_protocol_sha256": frozen["protocol_sha256"],
        "temperature": temperature,
        "selected_operating_point": calibrated_point,
        "raw_logit_operating_point": raw_logit_point,
        "raw_probability_operating_point": raw_probability_point,
        "validation_pair_count": PAIRED_GAP_COUNT,
        "validation_prevalence": 0.5,
        "validation_raw_nll": float(log_loss(labels, raw_probabilities, labels=[0.0, 1.0])),
        "validation_calibrated_nll": float(log_loss(labels, calibrated, labels=[0.0, 1.0])),
        "v5_test_samples_opened": 0,
        "old_confirm_used_for_calibration": False,
        "expert_data_accessed": False,
        "cracks_samples_opened": 0,
    }
    return {**core, "freeze_sha256": _canonical_hash(core), "rows": rows}


def write_validation_calibration(output_root: Path, *, project_root: Path, device: str = "cuda") -> dict[str, Any]:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    freeze_path = output / "calibration_freeze.json"
    scores_path = output / "validation_scores.csv"
    if freeze_path.exists():
        existing = json.loads(freeze_path.read_text())
        core = {key: value for key, value in existing.items() if key != "freeze_sha256"}
        if existing.get("freeze_sha256") != _canonical_hash(core):
            raise PermissionError("calibration freeze hash invalid")
        return {**existing, "action": "SKIP_ALREADY_FROZEN"}
    result = run_validation_calibration(project_root, device=device)
    rows = result.pop("rows")
    with scores_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result["validation_scores_csv"] = str(scores_path)
    # Include the score artifact hash in the immutable core before final hashing.
    result.pop("freeze_sha256")
    result["validation_scores_sha256"] = _sha256(scores_path)
    result["freeze_sha256"] = _canonical_hash(result)
    freeze_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result

