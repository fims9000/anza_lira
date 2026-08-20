"""Frozen validation evaluation and quality gate for synthetic candidates."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.segmentation_v2 import build_comparable_model
from synthetic.crossing_trace_bench import generate_sample
from synthetic.experiment_matrix import SyntheticRunSpec, development_matrix
from synthetic.structural_losses import branch_transition_logits
from synthetic.structural_metrics import compute_structural_metrics
from synthetic.training import load_checkpoint
from trace_extraction.geometry import local_pca_orientation
from trace_extraction.skeleton import skeletonize_mask


EVALUATION_PROTOCOL = {
    "split": "validation",
    "indices": list(range(256)),
    "visible_threshold_candidates": [round(value, 2) for value in np.arange(0.10, 0.91, 0.05)],
    "visible_threshold_selection": "maximum mean per-sample visible Dice; lowest threshold breaks ties",
    "completion_threshold": 0.5,
    "continuation_probability_threshold": 0.5,
    "baseline_continuation_readout": "minimum axial turning angle from predicted-query branch geometry",
    "v2_continuation_readout": "mode transport aggregated over declared branch query masks",
    "ground_truth_relation": "generator lineage only",
    "test_stream": "FROZEN_UNOPENED",
}


def evaluation_protocol_hash() -> str:
    encoded = json.dumps(EVALUATION_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def continuation_probabilities(logits: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
    scores = torch.as_tensor(logits)
    mask = torch.as_tensor(eligible, device=scores.device, dtype=torch.bool)
    if scores.shape != mask.shape or scores.ndim != 2:
        raise ValueError("Continuation logits and eligibility must be matching branch matrices")
    output = torch.zeros_like(scores)
    valid_rows = mask.any(dim=-1)
    if valid_rows.any():
        masked = scores[valid_rows].masked_fill(~mask[valid_rows], torch.finfo(scores.dtype).min)
        output[valid_rows] = torch.softmax(masked, dim=-1)
    return output


def _axial_distance(first: float, second: float) -> float:
    delta = first - second
    return math.atan2(abs(math.sin(delta)), abs(math.cos(delta)))


def _branch_angle(sample: dict[str, Any], branch_id: int, point_xy: list[float], radius: int = 5) -> float:
    branch_index = sample["branch_ids"].index(branch_id)
    x, y = int(round(point_xy[0])), int(round(point_xy[1]))
    centerline = np.asarray(sample["branch_centerlines"][branch_index], dtype=bool)
    yy, xx = np.ogrid[: centerline.shape[0], : centerline.shape[1]]
    selected = centerline & ((yy - y) ** 2 + (xx - x) ** 2 <= radius**2)
    cos2 = np.asarray(sample["branch_tangent_cos2"][branch_index])[selected]
    sin2 = np.asarray(sample["branch_tangent_sin2"][branch_index])[selected]
    nonzero = np.abs(cos2) + np.abs(sin2) > 0
    if not nonzero.any():
        return 0.0
    return 0.5 * math.atan2(float(sin2[nonzero].mean()), float(cos2[nonzero].mean()))


def minimum_angle_continuation_scores(sample: dict[str, Any]) -> np.ndarray:
    """Geometry-only predicted baseline; never used to construct GT."""
    branch_ids = [int(value) for value in sample["branch_ids"]]
    branch_index = {branch_id: index for index, branch_id in enumerate(branch_ids)}
    scores = np.zeros((len(branch_ids), len(branch_ids)), dtype=np.float32)
    for junction in sample["junctions"]:
        incident = [int(value) for value in junction["incident_branch_ids"]]
        angles = {
            branch_id: _branch_angle(sample, branch_id, junction["point_xy"])
            for branch_id in incident
        }
        pair_cost = {
            tuple(sorted((first, second))): _axial_distance(angles[first], angles[second])
            for position, first in enumerate(incident)
            for second in incident[position + 1 :]
        }
        selected: list[tuple[int, int]]
        if junction["junction_type"] == "x_crossing" and len(incident) == 4:
            a, b, c, d = incident
            matchings = (
                ((a, b), (c, d)),
                ((a, c), (b, d)),
                ((a, d), (b, c)),
            )
            selected = list(min(matchings, key=lambda pairs: sum(pair_cost[tuple(sorted(pair))] for pair in pairs)))
        elif junction["junction_type"] == "y_branch":
            selected = [pair for pair, _cost in sorted(pair_cost.items(), key=lambda item: item[1])[:2]]
        else:
            selected = [min(pair_cost, key=pair_cost.get)] if pair_cost else []
        for first, second in selected:
            first_index, second_index = branch_index[first], branch_index[second]
            scores[first_index, second_index] = 1.0
            scores[second_index, first_index] = 1.0
    return scores


def _sample_visible_dice(probability: np.ndarray, target: np.ndarray, threshold: float) -> float:
    prediction = probability >= threshold
    truth = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum() + truth.sum())
    return 2.0 * int(np.logical_and(prediction, truth).sum()) / denominator if denominator else 1.0


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key not in {"index", "seed"}
    ]
    aggregate: dict[str, float] = {}
    for name in metric_names:
        if name in {"branch_pairing_accuracy", "branch_continuation_f1"}:
            count_name = "branch_pairing_count" if name == "branch_pairing_accuracy" else "continuation_relation_count"
            numerator = sum(float(row[name]) * float(row[count_name]) for row in rows)
            denominator = sum(float(row[count_name]) for row in rows)
            aggregate[name] = numerator / denominator if denominator else 1.0
        elif name == "gap_recovery_rate":
            denominator = sum(float(row["positive_gap_count"]) for row in rows)
            aggregate[name] = (
                sum(float(row[name]) * float(row["positive_gap_count"]) for row in rows) / denominator
                if denominator
                else 1.0
            )
        elif name == "false_bridge_rate":
            denominator = sum(float(row["negative_gap_count"]) for row in rows)
            aggregate[name] = (
                sum(float(row[name]) * float(row["negative_gap_count"]) for row in rows) / denominator
                if denominator
                else 0.0
            )
        else:
            aggregate[name] = float(np.mean([float(row[name]) for row in rows]))
    return aggregate


def evaluate_candidate(
    spec: SyntheticRunSpec,
    development_root: Path,
    output_root: Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    run_dir = development_root / f"{spec.candidate_id}-{spec.run_hash}"
    status = json.loads((run_dir / "status.json").read_text())
    if status.get("status") != "COMPLETE":
        raise ValueError(f"Training is not complete for {spec.candidate_id}")
    torch_device = torch.device(device)
    model = build_comparable_model(spec.model).to(torch_device)
    load_checkpoint(
        run_dir / "checkpoint-last.pt",
        expected_hash=spec.run_hash,
        model=model,
    )
    model.eval()

    cached: list[tuple[dict[str, Any], np.ndarray]] = []
    with torch.no_grad():
        for index in EVALUATION_PROTOCOL["indices"]:
            sample = generate_sample("validation", index, image_size=128)
            image = torch.as_tensor(sample["image"], device=torch_device).unsqueeze(0)
            probability = torch.sigmoid(model(image))[0, 0].cpu().numpy()
            cached.append((sample, probability))
    threshold_scores = {
        threshold: float(
            np.mean(
                [
                    _sample_visible_dice(probability, sample["visible_fault_mask"], threshold)
                    for sample, probability in cached
                ]
            )
        )
        for threshold in EVALUATION_PROTOCOL["visible_threshold_candidates"]
    }
    selected_threshold = max(threshold_scores, key=lambda value: (threshold_scores[value], -value))

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample, probability in cached:
            image = torch.as_tensor(sample["image"], device=torch_device).unsqueeze(0)
            diagnostics = model(image, return_diagnostics=True)
            visible_prediction = probability >= selected_threshold
            if "completion_logits" in diagnostics:
                completion_probability = torch.sigmoid(diagnostics["completion_logits"])[0, 0].cpu().numpy()
                completion_prediction = visible_prediction | (
                    completion_probability >= float(EVALUATION_PROTOCOL["completion_threshold"])
                )
            else:
                completion_prediction = visible_prediction
            if diagnostics["transport_diagnostics"]:
                first = diagnostics["transport_diagnostics"][0]
                logits = branch_transition_logits(
                    first["transport"],
                    torch.as_tensor(sample["branch_masks"], device=torch_device),
                    variant=first["variant"],
                )
                continuation = continuation_probabilities(
                    logits,
                    torch.as_tensor(sample["continuation_eligible_matrix"], device=torch_device),
                ).cpu().numpy()
            else:
                continuation = minimum_angle_continuation_scores(sample)
            orientation = local_pca_orientation(skeletonize_mask(completion_prediction))
            metrics = compute_structural_metrics(
                visible_prediction,
                sample,
                predicted_completion_mask=completion_prediction,
                predicted_continuation_scores=continuation,
                predicted_orientation=orientation,
            )
            rows.append(
                {
                    "candidate_id": spec.candidate_id,
                    "model": spec.model,
                    "index": sample["index"],
                    "seed": sample["seed"],
                    "case": sample["case"],
                    "strata": ";".join(sample["strata"]),
                    **metrics,
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"{spec.candidate_id}-{spec.run_hash}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "COMPLETE",
        "candidate_id": spec.candidate_id,
        "model": spec.model,
        "run_hash": spec.run_hash,
        "evaluation_protocol_hash": evaluation_protocol_hash(),
        "selected_visible_threshold": selected_threshold,
        "threshold_scores": {str(key): value for key, value in threshold_scores.items()},
        "sample_count": len(rows),
        "metrics": _aggregate(rows),
        "test_samples_opened": 0,
        "rows_csv": str(csv_path),
    }
    (output_root / f"{spec.candidate_id}-{spec.run_hash}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def evaluate_frozen_test(study_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    freeze_path = study_root / "synthetic" / "frozen_v2.json"
    if not freeze_path.exists():
        raise ValueError("Synthetic candidate must be frozen before opening test")
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("test_open_authorization") != "ONE_EVALUATION_AFTER_THIS_FREEZE":
        raise ValueError("Frozen candidate does not authorize a test evaluation")
    test_root = study_root / "synthetic" / "test"
    receipt_path = test_root / "test_open_receipt.json"
    summary_path = test_root / "summary.json"
    if receipt_path.exists() and summary_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("freeze_sha256") == freeze["freeze_sha256"]:
            return {**json.loads(summary_path.read_text()), "action": "SKIP"}

    by_id = {spec.candidate_id: spec for spec in development_matrix()}
    selected_ids = ("B0", "B1", "C0", freeze["frozen_candidate_id"])
    development_root = study_root / "synthetic" / "development"
    validation_root = study_root / "synthetic" / "validation"
    test_root.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)
    summaries = {}
    for candidate_id in selected_ids:
        spec = by_id[candidate_id]
        validation = json.loads(
            (validation_root / f"{candidate_id}-{spec.run_hash}.json").read_text()
        )
        threshold = float(validation["selected_visible_threshold"])
        model = build_comparable_model(spec.model).to(torch_device)
        load_checkpoint(
            development_root / f"{candidate_id}-{spec.run_hash}" / "checkpoint-last.pt",
            expected_hash=spec.run_hash,
            model=model,
        )
        model.eval()
        rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for index in range(2000):
                sample = generate_sample("test", index, image_size=128)
                image = torch.as_tensor(sample["image"], device=torch_device).unsqueeze(0)
                output = model(image, return_diagnostics=True)
                visible_probability = torch.sigmoid(output["visible_logits"])[0, 0].cpu().numpy()
                visible_prediction = visible_probability >= threshold
                if "completion_logits" in output:
                    completion_probability = torch.sigmoid(output["completion_logits"])[0, 0].cpu().numpy()
                    completion_prediction = visible_prediction | (
                        completion_probability >= float(EVALUATION_PROTOCOL["completion_threshold"])
                    )
                else:
                    completion_prediction = visible_prediction
                if output["transport_diagnostics"]:
                    first = output["transport_diagnostics"][0]
                    logits = branch_transition_logits(
                        first["transport"],
                        torch.as_tensor(sample["branch_masks"], device=torch_device),
                        variant=first["variant"],
                    )
                    continuation = continuation_probabilities(
                        logits,
                        torch.as_tensor(sample["continuation_eligible_matrix"], device=torch_device),
                    ).cpu().numpy()
                else:
                    continuation = minimum_angle_continuation_scores(sample)
                orientation = local_pca_orientation(skeletonize_mask(completion_prediction))
                metrics = compute_structural_metrics(
                    visible_prediction,
                    sample,
                    predicted_completion_mask=completion_prediction,
                    predicted_continuation_scores=continuation,
                    predicted_orientation=orientation,
                )
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "model": spec.model,
                        "index": index,
                        "seed": sample["seed"],
                        "case": sample["case"],
                        "strata": ";".join(sample["strata"]),
                        **metrics,
                    }
                )
        csv_path = test_root / f"{candidate_id}-{spec.run_hash}.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summaries[candidate_id] = {
            "model": spec.model,
            "run_hash": spec.run_hash,
            "validation_selected_visible_threshold": threshold,
            "sample_count": len(rows),
            "metrics": _aggregate(rows),
            "rows_csv": str(csv_path),
        }
        print(
            f"phase=synthetic_test model={candidate_id} samples=2000 "
            f"visible_dice={summaries[candidate_id]['metrics']['visible_dice']:.4f} "
            f"pairing={summaries[candidate_id]['metrics']['branch_pairing_accuracy']:.4f} status=COMPLETE"
        )
    receipt = {
        "status": "OPENED_ONCE",
        "freeze_sha256": freeze["freeze_sha256"],
        "split": "test",
        "seed_base": 30_000_000,
        "sample_count": 2000,
        "candidate_ids": list(selected_ids),
        "evaluation_protocol_hash": evaluation_protocol_hash(),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    summary = {
        "status": "COMPLETE",
        "action": "RUN",
        "scientific_result": True,
        "freeze_sha256": freeze["freeze_sha256"],
        "quality_gate": freeze["quality_gate"],
        "frozen_candidate_id": freeze["frozen_candidate_id"],
        "test_open_receipt": str(receipt_path),
        "models": summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
