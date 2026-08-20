"""Expert-blind T0/T1 evaluation on held-out per-annotator partial labels."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
import torch

from cracks_experiment.clean_anza_evaluation import _bootstrap
from cracks_experiment.evaluation import evaluate_binary_section
from cracks_experiment.matrix import PROJECT_ROOT, setting_a_matrix
from cracks_experiment.partial_label_training import (
    T1_PROTOCOL,
    T1RunSpec,
    _model,
    load_t1_checkpoint,
    t1_matrix,
    t1_protocol_hash,
)
from cracks_experiment.partial_labels import CRACKSMultiAnnotatorDataset
from cracks_experiment.training import NORMALIZATION, build_real_model, load_real_checkpoint
from cracks_experiment.validation import _sha256, tiled_probability


T1_ROOT = PROJECT_ROOT / "results" / "final_practical_cycle" / "cracks_t1"
T0_ROOT = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_a"
DETAIL_METRICS = (
    "dice",
    "iou",
    "auprc",
    "precision",
    "recall",
    "cldice",
    "skeleton_f1_at_2px",
    "fragmentation",
    "predicted_foreground_fraction",
    "explicit_predicted_foreground_fraction",
    "explicit_target_foreground_fraction",
    "brier",
    "ece",
)
T1_EVALUATION_PROTOCOL = {
    "version": "cracks_partial_label_evaluator_v1",
    "training_protocol_sha256": t1_protocol_hash(),
    "threshold_candidates": T1_PROTOCOL["threshold_candidates"],
    "threshold_selection": "maximize macro section mean of per-annotator explicit-pixel Dice; ties choose lower threshold",
    "metrics": list(DETAIL_METRICS),
    "aggregation": "annotator mean within section; seed mean within section; 10000 section bootstrap",
    "strong_gain_rule": "for both models Dice and recall delta >=0.02 with CI95 low >0; AUPRC delta >=-0.005",
    "expert": "LOCKED_NOT_ACCESSED",
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}


def t1_evaluation_protocol_hash() -> str:
    return hashlib.sha256(
        json.dumps(T1_EVALUATION_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _heldout_dataset() -> CRACKSMultiAnnotatorDataset:
    return CRACKSMultiAnnotatorDataset(
        PROJECT_ROOT / "data" / "cracks" / "images",
        PROJECT_ROOT / "data" / "cracks" / "annotations",
        T1_PROTOCOL["heldout_section_ids"],
        T1_PROTOCOL["heldout_annotators"],
        mean=NORMALIZATION["mean"],
        std=NORMALIZATION["std"],
        crop_size=None,
        annotators_per_section=None,
        seed=42,
    )


def _calibration(probability: np.ndarray, target: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    selected = weight > 0
    p = np.asarray(probability, dtype=np.float64)[selected]
    y = np.asarray(target, dtype=np.float64)[selected]
    w = np.asarray(weight, dtype=np.float64)[selected]
    brier = float(np.sum(w * (p - y) ** 2) / np.sum(w))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (p >= low) & (p < high if high < 1.0 else p <= high)
        if not np.any(mask):
            continue
        mass = float(np.sum(w[mask]))
        ece += mass / float(np.sum(w)) * abs(
            float(np.average(p[mask], weights=w[mask])) - float(np.average(y[mask], weights=w[mask]))
        )
    return brier, float(ece)


def _load_t1_model(spec: T1RunSpec, device: str) -> tuple[torch.nn.Module, Path, str]:
    run_dir = T1_ROOT / f"{spec.run_id}-{spec.run_hash}"
    checkpoint = run_dir / "checkpoint-last.pt"
    status = json.loads((run_dir / "status.json").read_text())
    if status.get("status") != "COMPLETE" or status.get("expert_data_accessed") is not False:
        raise ValueError(f"T1 training is not complete and expert-locked: {spec.run_id}")
    model = _model(spec).to(torch.device(device))
    load_t1_checkpoint(checkpoint, spec, model)
    return model, checkpoint, spec.run_hash


def _load_t0_model(model_name: str, seed: int, device: str) -> tuple[torch.nn.Module, Path, str, str]:
    spec = next(
        row for row in setting_a_matrix()
        if row.model == model_name and row.seed == seed and row.comparison_family == "main"
    )
    run_dir = T0_ROOT / f"{spec.run_id}-{spec.run_hash}"
    checkpoint = run_dir / "checkpoint-last.pt"
    status = json.loads((run_dir / "status.json").read_text())
    if status.get("status") != "COMPLETE" or status.get("expert_scores_used") is not False:
        raise ValueError(f"Historical T0 run is not complete and expert-locked: {spec.run_id}")
    model = build_real_model(spec).to(torch.device(device))
    load_real_checkpoint(checkpoint, spec.run_hash, model)
    return model, checkpoint, spec.run_hash, spec.run_id


def _select_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sweep = []
    for threshold in sorted({float(row["threshold"]) for row in rows}):
        selected = [row for row in rows if float(row["threshold"]) == float(threshold)]
        sweep.append({
            "threshold": float(threshold),
            "section_count": len(selected),
            "macro_annotator_mean_dice": float(np.mean([row["dice"] for row in selected])),
            "macro_annotator_mean_recall": float(np.mean([row["recall"] for row in selected])),
        })
    choice = max(sweep, key=lambda row: (row["macro_annotator_mean_dice"], -row["threshold"]))
    return {
        "selected_threshold": choice["threshold"],
        "selection_metric": "macro_section_mean_of_annotator_explicit_pixel_dice",
        "sweep": sweep,
    }


def _evaluate_loaded_model(
    model: torch.nn.Module,
    *,
    label: str,
    model_name: str,
    seed: int,
    run_id: str,
    run_hash: str,
    checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = _sha256(checkpoint)
    result_path = output_dir / "evaluation.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if (
            existing.get("status") == "COMPLETE"
            and existing.get("checkpoint_sha256") == checkpoint_sha
            and existing.get("t1_evaluator_protocol_sha256") == t1_evaluation_protocol_hash()
            and existing.get("expert_data_accessed") is False
        ):
            return {**existing, "action": "SKIP"}
        raise ValueError(f"Existing T1 evaluation provenance drift: {result_path}")
    model.eval()
    dataset = _heldout_dataset()
    probabilities: list[np.ndarray] = []
    sweep_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            batch = dataset[index]
            probability = tiled_probability(model, batch["image"]).numpy()[:255, :701]
            probabilities.append(probability.astype(np.float16))
            section_rows: dict[float, list[dict[str, float]]] = {
                float(value): [] for value in T1_PROTOCOL["threshold_candidates"]
            }
            for target, weight in zip(batch["targets"][:, 0, :255, :701], batch["weights"][:, 0, :255, :701]):
                target_np = target.numpy() >= 0.5
                valid = weight.numpy() > 0
                for threshold in section_rows:
                    prediction = probability >= threshold
                    pred = prediction[valid]
                    truth = target_np[valid]
                    tp = int(np.count_nonzero(pred & truth))
                    fp = int(np.count_nonzero(pred & ~truth))
                    fn = int(np.count_nonzero(~pred & truth))
                    denominator = 2 * tp + fp + fn
                    section_rows[threshold].append({
                        "dice": float(2 * tp / denominator) if denominator else 1.0,
                        "recall": float(tp / (tp + fn)) if tp + fn else 1.0,
                    })
            for threshold, values in section_rows.items():
                sweep_rows.append({
                    "section_id": int(batch["section_id"]),
                    "threshold": threshold,
                    "dice": float(np.mean([value["dice"] for value in values])),
                    "recall": float(np.mean([value["recall"] for value in values])),
                })
            if (index + 1) % 40 == 0 or index + 1 == len(dataset):
                print(
                    f"phase=cracks_t1_validation run={label} section={index + 1}/{len(dataset)} "
                    "expert=LOCKED status=RUNNING",
                    flush=True,
                )
    selection = _select_threshold(sweep_rows)
    threshold = float(selection["selected_threshold"])
    annotation_rows: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        batch = dataset[index]
        probability = probabilities[index].astype(np.float32)
        rows = []
        for annotator, target, weight in zip(
            batch["annotators"], batch["targets"][:, 0, :255, :701], batch["weights"][:, 0, :255, :701]
        ):
            target_np = target.numpy() >= 0.5
            weight_np = weight.numpy()
            valid = weight_np > 0
            metrics = evaluate_binary_section(probability, target_np, valid, threshold)
            selected_target = target_np[valid]
            selected_probability = probability[valid]
            auprc = (
                float(average_precision_score(selected_target, selected_probability))
                if selected_target.any()
                else float(not np.any(selected_probability > threshold))
            )
            brier, ece = _calibration(probability, target_np, weight_np)
            prediction = probability >= threshold
            row = {
                "protocol": label,
                "model": model_name,
                "seed": seed,
                "run_id": run_id,
                "run_hash": run_hash,
                "section_id": int(batch["section_id"]),
                "annotator": annotator,
                "threshold": threshold,
                "auprc": auprc,
                "predicted_foreground_fraction": float(prediction.mean()),
                "explicit_predicted_foreground_fraction": float(prediction[valid].mean()),
                "explicit_target_foreground_fraction": float(target_np[valid].mean()),
                "brier": brier,
                "ece": ece,
                **{key: metrics[key] for key in (
                    "dice", "iou", "precision", "recall", "cldice",
                    "skeleton_f1_at_2px", "fragmentation",
                )},
            }
            annotation_rows.append(row)
            rows.append(row)
        section_rows.append({
            "protocol": label,
            "model": model_name,
            "seed": seed,
            "run_id": run_id,
            "run_hash": run_hash,
            "section_id": int(batch["section_id"]),
            "annotator_count": len(rows),
            "threshold": threshold,
            **{metric: float(np.mean([float(row[metric]) for row in rows])) for metric in DETAIL_METRICS},
        })
    for path, rows in (
        (output_dir / "per_annotator.csv", annotation_rows),
        (output_dir / "per_section.csv", section_rows),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    result = {
        "status": "COMPLETE",
        "action": "RUN",
        "protocol": label,
        "model": model_name,
        "seed": seed,
        "run_id": run_id,
        "run_hash": run_hash,
        "checkpoint_sha256": checkpoint_sha,
        "t1_evaluator_protocol_sha256": t1_evaluation_protocol_hash(),
        "section_count": len(section_rows),
        "annotation_row_count": len(annotation_rows),
        "per_section_csv": str(output_dir / "per_section.csv"),
        "per_annotator_csv": str(output_dir / "per_annotator.csv"),
        "summary": {metric: float(np.mean([row[metric] for row in section_rows])) for metric in DETAIL_METRICS},
        "expert_scores_used": False,
        "expert_data_accessed": False,
        **selection,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def evaluate_t1_run(spec: T1RunSpec, *, device: str = "cuda") -> dict[str, Any]:
    model, checkpoint, run_hash = _load_t1_model(spec, device)
    return _evaluate_loaded_model(
        model,
        label="T1_partial_labels",
        model_name=spec.model,
        seed=spec.seed,
        run_id=spec.run_id,
        run_hash=run_hash,
        checkpoint=checkpoint,
        output_dir=T1_ROOT / "evaluation" / spec.run_id,
    )


def evaluate_t0_control(model_name: str, seed: int, *, device: str = "cuda") -> dict[str, Any]:
    model, checkpoint, run_hash, source_run_id = _load_t0_model(model_name, seed, device)
    return _evaluate_loaded_model(
        model,
        label="T0_paper_like_training_partial_evaluator",
        model_name=model_name,
        seed=seed,
        run_id=source_run_id,
        run_hash=run_hash,
        checkpoint=checkpoint,
        output_dir=T1_ROOT / "controls" / f"t0_{model_name}_s{seed}",
    )


def build_t1_statistics() -> dict[str, Any]:
    results = [
        evaluate_t0_control(model, seed, device="cuda" if torch.cuda.is_available() else "cpu")
        for model in ("unet", "anza_v1") for seed in (41, 42, 43)
    ] + [
        evaluate_t1_run(spec, device="cuda" if torch.cuda.is_available() else "cpu")
        for spec in t1_matrix()
    ]
    indexed: dict[tuple[str, str, int, int], dict[str, str]] = {}
    all_rows: list[dict[str, str]] = []
    for result in results:
        with Path(result["per_section_csv"]).open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["protocol"], row["model"], int(row["seed"]), int(row["section_id"]))
                indexed[key] = row
                all_rows.append(row)
    sections = sorted({key[3] for key in indexed})
    if len(sections) != len(T1_PROTOCOL["heldout_section_ids"]):
        raise ValueError("T1 statistics section alignment failed")
    comparisons: list[dict[str, Any]] = []
    by_model: dict[str, dict[str, dict[str, float]]] = {}
    for model in ("unet", "anza_v1"):
        by_model[model] = {}
        for metric in DETAIL_METRICS:
            deltas = np.asarray([
                np.mean([
                    float(indexed[("T1_partial_labels", model, seed, section)][metric])
                    - float(indexed[("T0_paper_like_training_partial_evaluator", model, seed, section)][metric])
                    for seed in (41, 42, 43)
                ])
                for section in sections
            ])
            mean, low, high = _bootstrap(deltas, seed=800 + len(comparisons), resamples=10_000)
            row = {
                "comparison": f"T1_minus_T0_{model}",
                "model": model,
                "metric": metric,
                "mean_delta": mean,
                "ci95_low": low,
                "ci95_high": high,
                "section_count": len(sections),
                "seed_count": 3,
                "pairing": "section+seed_delta_then_seed_mean_within_section",
            }
            comparisons.append(row)
            by_model[model][metric] = row
    checks = {}
    for model in ("unet", "anza_v1"):
        checks[f"{model}_dice_strong_gain"] = (
            by_model[model]["dice"]["mean_delta"] >= 0.02
            and by_model[model]["dice"]["ci95_low"] > 0
        )
        checks[f"{model}_recall_strong_gain"] = (
            by_model[model]["recall"]["mean_delta"] >= 0.02
            and by_model[model]["recall"]["ci95_low"] > 0
        )
        checks[f"{model}_auprc_noninferior"] = by_model[model]["auprc"]["mean_delta"] >= -0.005
    status = "CRACKS_PARTIAL_LABEL_SUCCESS" if all(checks.values()) else "CRACKS_PARTIAL_LABEL_NO_STRONG_GAIN"
    analysis = T1_ROOT / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    with (analysis / "raw_per_section.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    with (analysis / "paired_comparisons.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)
    result = {
        "status": status,
        "target_construction_was_major_bottleneck": status == "CRACKS_PARTIAL_LABEL_SUCCESS",
        "predeclared_strong_gain_rule": "for both models Dice and recall delta >=0.02 with section-bootstrap CI95 low >0; AUPRC delta >=-0.005",
        "checks": checks,
        "comparisons": comparisons,
        "section_count": len(sections),
        "seed_count": 3,
        "expert_scores_used": False,
        "expert_data_accessed": False,
    }
    (analysis / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
