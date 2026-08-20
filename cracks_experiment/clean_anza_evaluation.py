"""Crowd-heldout R0 evaluation and section-clustered CleanANZA comparison."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
import torch

from cracks_experiment.clean_anza_r0 import (
    LEGACY_TRAINING_ROOT,
    R0_ROOT,
    R0_PROTOCOL,
    CleanR0Spec,
    clean_r0_specs,
)
from cracks_experiment.evaluation import evaluate_binary_section, verify_threshold_freeze
from cracks_experiment.matrix import CRACKSRunSpec, PROJECT_ROOT, setting_a_matrix
from cracks_experiment.training import NORMALIZATION, build_real_model, load_real_checkpoint
from cracks_experiment.validation import _sha256, tiled_probability
from datasets.cracks import CRACKSSectionDataset


PRIMARY_METRICS = (
    "dice",
    "iou",
    "auprc",
    "cldice",
    "skeleton_f1_at_2px",
    "precision",
    "recall",
    "predicted_foreground_fraction",
    "target_foreground_fraction",
    "fragmentation",
)


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_clean_threshold_freeze() -> dict[str, Any]:
    path = R0_ROOT / "clean_threshold_freeze.json"
    receipt = json.loads(path.read_text())
    digest = receipt.pop("freeze_sha256", None)
    if digest != _canonical_hash(receipt):
        raise PermissionError("invalid CleanANZA threshold freeze hash")
    if receipt.get("status") != "FROZEN" or receipt.get("expert_scores_used") is not False:
        raise PermissionError("CleanANZA expert lock failed")
    by_run = {row["run_id"]: row for row in receipt["runs"]}
    if set(by_run) != {spec.run_id for spec in clean_r0_specs()}:
        raise PermissionError("CleanANZA threshold freeze is incomplete")
    for spec in clean_r0_specs():
        row = by_run[spec.run_id]
        run_dir = R0_ROOT / f"{spec.run_id}-{spec.run_hash}"
        if (
            row["run_hash"] != spec.run_hash
            or row["checkpoint_sha256"] != _sha256(run_dir / "checkpoint-last.pt")
            or row["validation_sha256"] != _sha256(run_dir / "crowd_validation.json")
        ):
            raise PermissionError(f"CleanANZA frozen artifact changed: {spec.run_id}")
    return {**receipt, "freeze_sha256": digest}


def _r0_runs() -> list[tuple[str, CRACKSRunSpec | CleanR0Spec, Path, float, str]]:
    legacy = verify_threshold_freeze(LEGACY_TRAINING_ROOT)
    clean = verify_clean_threshold_freeze()
    legacy_threshold = {row["run_id"]: float(row["selected_threshold"]) for row in legacy["runs"]}
    clean_threshold = {row["run_id"]: float(row["selected_threshold"]) for row in clean["runs"]}
    matrix = setting_a_matrix()
    runs: list[tuple[str, CRACKSRunSpec | CleanR0Spec, Path, float, str]] = []
    for model, model_name in (("unet", "unet"), ("anza_v1", "anza_v1")):
        for seed in R0_PROTOCOL["seeds"]:
            spec = next(item for item in matrix if item.model == model_name and item.seed == seed and item.comparison_family == "main")
            runs.append((model, spec, LEGACY_TRAINING_ROOT, legacy_threshold[spec.run_id], legacy["freeze_sha256"]))
    for spec in clean_r0_specs():
        runs.append(("clean_anza", spec, R0_ROOT, clean_threshold[spec.run_id], clean["freeze_sha256"]))
    return runs


def _heldout_dataset() -> CRACKSSectionDataset:
    protocol = json.loads((PROJECT_ROOT / "results" / "anza_v2_study" / "protocol.json").read_text())
    return CRACKSSectionDataset(
        PROJECT_ROOT / "data" / "cracks" / "images",
        PROJECT_ROOT / "data" / "cracks" / "crowd_targets" / R0_PROTOCOL["policy"] / "heldout",
        protocol["setting_a"]["held_out_validation_section_ids"],
        mean=NORMALIZATION["mean"],
        std=NORMALIZATION["std"],
    )


def evaluate_r0_run(
    model_name: str,
    spec: CRACKSRunSpec | CleanR0Spec,
    training_root: Path,
    threshold: float,
    freeze_sha256: str,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    output_root = R0_ROOT / "crowd_evaluation"
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / f"{spec.run_id}-{spec.run_hash}.json"
    rows_path = output_root / f"{spec.run_id}-{spec.run_hash}.csv"
    run_dir = training_root / f"{spec.run_id}-{spec.run_hash}"
    checkpoint = run_dir / "checkpoint-last.pt"
    checkpoint_sha = _sha256(checkpoint)
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if (
            existing.get("status") == "COMPLETE"
            and existing.get("checkpoint_sha256") == checkpoint_sha
            and existing.get("threshold_freeze_sha256") == freeze_sha256
            and existing.get("expert_data_accessed") is False
        ):
            return {**existing, "action": "SKIP"}
    model = build_real_model(spec).to(torch.device(device))
    load_real_checkpoint(checkpoint, spec.run_hash, model)
    model.eval()
    dataset = _heldout_dataset()
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        batch = dataset[index]
        probability = tiled_probability(model, batch["image"]).numpy()
        height, width = batch["original_hw"]
        probability = probability[:height, :width]
        target = batch["target"][0, :height, :width].numpy() >= 0.5
        valid = batch["valid"][0, :height, :width].numpy().astype(bool)
        metrics = evaluate_binary_section(probability, target, valid, threshold)
        selected_probability = probability[valid]
        selected_target = target[valid]
        if not selected_target.any():
            auprc = 1.0 if not np.any(selected_probability > 0) else 0.0
        else:
            auprc = float(average_precision_score(selected_target, selected_probability))
        prediction = (probability >= threshold) & valid
        rows.append({
            "model": model_name,
            "run_id": spec.run_id,
            "run_hash": spec.run_hash,
            "seed": spec.seed,
            "section_id": int(batch["section_id"]),
            "threshold": float(threshold),
            "auprc": auprc,
            "predicted_foreground_fraction": float(prediction[valid].mean()),
            "target_foreground_fraction": float(target[valid].mean()),
            **{key: metrics[key] for key in ("dice", "iou", "precision", "recall", "cldice", "skeleton_f1_at_2px", "fragmentation")},
        })
        if (index + 1) % 40 == 0 or index + 1 == len(dataset):
            print(
                f"phase=clean_anza_r0_eval model={spec.run_id} section={index + 1}/{len(dataset)} "
                "expert=LOCKED status=RUNNING"
            )
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {metric: float(np.mean([float(row[metric]) for row in rows])) for metric in PRIMARY_METRICS}
    result = {
        "status": "COMPLETE",
        "action": "RUN",
        "model": model_name,
        "run_id": spec.run_id,
        "run_hash": spec.run_hash,
        "seed": spec.seed,
        "section_count": len(rows),
        "selected_threshold": float(threshold),
        "threshold_freeze_sha256": freeze_sha256,
        "checkpoint_sha256": checkpoint_sha,
        "expert_scores_used_for_selection": False,
        "expert_data_accessed": False,
        "rows_csv": str(rows_path),
        "summary": summary,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _bootstrap(values: np.ndarray, *, seed: int = 42, resamples: int = 10_000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = array[generator.integers(0, len(array), size=(int(resamples), len(array)))].mean(axis=1)
    return [float(array.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def build_r0_statistics() -> dict[str, Any]:
    evaluations = [
        evaluate_r0_run(model, spec, root, threshold, freeze, device="cuda" if torch.cuda.is_available() else "cpu")
        for model, spec, root, threshold, freeze in _r0_runs()
    ]
    all_rows: list[dict[str, Any]] = []
    for result in evaluations:
        with Path(result["rows_csv"]).open(newline="") as handle:
            all_rows.extend(csv.DictReader(handle))
    indexed = {
        (row["model"], int(row["seed"]), int(row["section_id"])): row
        for row in all_rows
    }
    section_ids = sorted({key[2] for key in indexed})
    expected = {(model, seed, section) for model in ("unet", "anza_v1", "clean_anza") for seed in R0_PROTOCOL["seeds"] for section in section_ids}
    if set(indexed) != expected or len(section_ids) != 392:
        raise ValueError("R0 section/seed alignment failed")
    model_rows: list[dict[str, Any]] = []
    section_means: dict[tuple[str, int, str], float] = {}
    for model in ("unet", "anza_v1", "clean_anza"):
        for metric in PRIMARY_METRICS:
            values = []
            for section in section_ids:
                value = float(np.mean([
                    float(indexed[(model, seed, section)][metric]) for seed in R0_PROTOCOL["seeds"]
                ]))
                section_means[(model, section, metric)] = value
                values.append(value)
            mean, low, high = _bootstrap(np.asarray(values), seed=100 + len(model_rows))
            model_rows.append({
                "model": model,
                "metric": metric,
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                "section_count": len(section_ids),
                "seed_count": 3,
                "aggregation": "seed_mean_within_section_then_section_bootstrap",
            })
    comparisons: list[dict[str, Any]] = []
    for comparator in ("anza_v1", "unet"):
        for metric in PRIMARY_METRICS:
            deltas = np.asarray([
                section_means[("clean_anza", section, metric)] - section_means[(comparator, section, metric)]
                for section in section_ids
            ])
            mean, low, high = _bootstrap(deltas, seed=500 + len(comparisons))
            comparisons.append({
                "comparison": f"clean_anza_minus_{comparator}",
                "metric": metric,
                "mean_delta": mean,
                "ci95_low": low,
                "ci95_high": high,
                "section_count": len(section_ids),
                "seed_count": 3,
                "pairing": "section+seed_delta_then_seed_mean_within_section",
            })
    v1 = {row["metric"]: row for row in comparisons if row["comparison"] == "clean_anza_minus_anza_v1"}
    structural = ("cldice", "skeleton_f1_at_2px")
    structural_success = [metric for metric in structural if v1[metric]["mean_delta"] >= 0.010 and v1[metric]["ci95_low"] > 0]
    second = next((metric for metric in structural if metric not in structural_success), None)
    checks = {
        "dice_noninferior": v1["dice"]["mean_delta"] >= -0.005,
        "auprc_noninferior": v1["auprc"]["mean_delta"] >= -0.005,
        "one_structural_gain_confirmed": bool(structural_success),
        "second_structural_noninferior": len(structural_success) == len(structural) or (
            second is not None and v1[second]["mean_delta"] >= -0.005
        ),
    }
    status = "CLEAN_ANZA_REAL_SUCCESS" if all(checks.values()) else "CLEAN_ANZA_REAL_GATE_FAIL"
    output = R0_ROOT / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("raw_per_section.csv", all_rows), ("main_metrics.csv", model_rows), ("paired_comparisons.csv", comparisons)):
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    result = {
        "status": status,
        "checks": checks,
        "confirmed_structural_metrics": structural_success,
        "models": model_rows,
        "comparisons": comparisons,
        "section_count": len(section_ids),
        "seed_count": 3,
        "expert_scores_used_for_selection": False,
        "expert_data_accessed": False,
    }
    (output / "r0_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
