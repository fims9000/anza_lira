"""Validation-only diagnostics and frozen gates for B0-B3."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from method_repair.context_matrix import CONTEXT_PROTOCOL, ContextRepairSpec, context_matrix
from method_repair.context_training import (
    build_context_candidate,
    cached_v3_sample,
    load_context_checkpoint,
)
from models.azconv_context_repaired import context_head_macs_per_pixel, context_head_parameter_count
from models.azconv_v2 import axial_distance
from synthetic.context_repair_losses import effective_mode_count, mode_cardinality_diagnostics
from synthetic.evaluation import continuation_probabilities
from synthetic.evaluation_corrected import evaluate_sample_corrected
from synthetic.mode_supervision import (
    axial_mode_set_loss,
    branch_mode_masks_from_tangents,
    mode_specific_branch_transition_logits,
)


def bootstrap_mean_ci(
    values: Iterable[float], *, resamples: int = 10_000, seed: int = 42
) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("bootstrap requires finite values")
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(int(resamples), len(array)))].mean(axis=1)
    return float(array.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _visible_dice(probability: np.ndarray, target: np.ndarray, threshold: float) -> float:
    prediction = probability >= threshold
    truth = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum() + truth.sum())
    return 2.0 * int((prediction & truth).sum()) / denominator if denominator else 1.0


def _branch_orientation(diagnostics: dict[str, torch.Tensor], sample: dict[str, Any], device: torch.device) -> tuple[np.ndarray, torch.Tensor]:
    branch_modes = branch_mode_masks_from_tangents(
        diagnostics["theta"],
        torch.as_tensor(sample["gt_branch_theta"], device=device).float().unsqueeze(0),
        torch.as_tensor(sample["gt_branch_theta_valid"], device=device).bool().unsqueeze(0),
    )[0]
    predicted = (branch_modes * diagnostics["theta"][0].unsqueeze(0)).sum(dim=1)
    return predicted.cpu().numpy(), branch_modes


def _matched_orientation_errors(
    diagnostics: dict[str, torch.Tensor],
    sample: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    _loss, details = axial_mode_set_loss(
        diagnostics["theta"],
        diagnostics["membership"],
        torch.as_tensor(sample["gt_theta_set"], device=device).float().unsqueeze(0),
        torch.as_tensor(sample["gt_theta_valid"], device=device).bool().unsqueeze(0),
    )
    assignment = details["assignment"]
    truth = torch.as_tensor(sample["gt_theta_set"], device=device).float().unsqueeze(0)
    valid = torch.as_tensor(sample["gt_theta_valid"], device=device).bool().unsqueeze(0)
    gathered = diagnostics["theta"].gather(1, assignment.clamp_min(0))
    errors = torch.rad2deg(axial_distance(gathered, truth))[valid]
    return errors.cpu().numpy(), details


def _weighted(rows: list[dict[str, Any]], metric: str, count: str, *, empty: float) -> float:
    denominator = sum(int(row[count]) for row in rows)
    if not denominator:
        return float(empty)
    return sum(float(row[metric]) * int(row[count]) for row in rows) / denominator


def evaluate_context_candidate(
    spec: ContextRepairSpec,
    development_root: Path,
    output_root: Path,
    *,
    device: str = "cuda",
    sample_count: int = 512,
) -> dict[str, Any]:
    run_dir = Path(development_root) / f"{spec.candidate_id}-{spec.run_hash}"
    status = json.loads((run_dir / "status.json").read_text())
    if status.get("status") != "COMPLETE" or status.get("run_hash") != spec.run_hash:
        raise ValueError(f"candidate training incomplete or mismatched: {spec.candidate_id}")
    for field in ("legacy_test_samples_opened", "v3_test_samples_opened", "cracks_samples_opened"):
        if status.get(field) != 0:
            raise ValueError(f"validation lock violation: {field}")
    image_size = int(status["image_size"])
    widths = tuple(int(value) for value in status["widths"])
    torch_device = torch.device(device)
    model = build_context_candidate(spec, widths=widths).to(torch_device)
    load_context_checkpoint(run_dir / "checkpoint-last.pt", expected_hash=spec.run_hash, model=model)
    model.eval()

    cached: list[tuple[dict[str, Any], np.ndarray]] = []
    with torch.inference_mode():
        for index in range(int(sample_count)):
            sample = cached_v3_sample("validation", index, image_size)
            image = torch.as_tensor(sample["image"], device=torch_device).unsqueeze(0)
            probability = torch.sigmoid(model(image))[0, 0].cpu().numpy()
            cached.append((sample, probability))
    threshold_scores = {
        float(threshold): float(np.mean([
            _visible_dice(probability, sample["visible_fault_mask"], float(threshold))
            for sample, probability in cached
        ]))
        for threshold in CONTEXT_PROTOCOL["threshold_candidates"]
    }
    threshold = max(threshold_scores, key=lambda value: (threshold_scores[value], -value))

    rows: list[dict[str, Any]] = []
    gate_truth: list[np.ndarray] = []
    gate_score: list[np.ndarray] = []
    neff_junction_values: list[np.ndarray] = []
    neff_straight_values: list[np.ndarray] = []
    gate_deltas: list[float] = []
    neff_deltas: list[float] = []
    orientation_errors: list[np.ndarray] = []
    with torch.inference_mode():
        for sample, probability in cached:
            image = torch.as_tensor(sample["image"], device=torch_device).unsqueeze(0)
            output = model(image, return_diagnostics=True)
            diagnostics = output["transport_diagnostics"][0]
            visible = probability >= threshold
            predicted_orientation, branch_modes = _branch_orientation(diagnostics, sample, torch_device)
            route_logits = mode_specific_branch_transition_logits(
                diagnostics["transport"], branch_modes, kernel_size=3
            )
            eligible = torch.as_tensor(sample["continuation_eligible_matrix"], device=torch_device).bool()
            route_probability = continuation_probabilities(route_logits, eligible).cpu().numpy()
            evaluated = evaluate_sample_corrected(
                visible,
                sample,
                predicted_completion_mask=visible,
                predicted_orientation=predicted_orientation,
                routing_probabilities=route_probability,
                has_transport_output=True,
            )
            family_a = evaluated["family_a"]
            family_b = evaluated["family_b"]

            errors, mode_details = _matched_orientation_errors(diagnostics, sample, torch_device)
            if len(errors):
                orientation_errors.append(errors)
            cardinality = mode_cardinality_diagnostics(
                diagnostics["membership"],
                torch.as_tensor(sample["gt_mode_count"], device=torch_device).unsqueeze(0),
            )
            neff = effective_mode_count(diagnostics["membership"])[0].cpu().numpy()
            gate = diagnostics["ambiguity_gate"][0].cpu().numpy()
            junction = np.asarray(sample["junction_map"], dtype=bool)
            straight = np.asarray(sample["visible_fault_mask"], dtype=bool) & (
                np.asarray(sample["gate_target"]) < 0.05
            )
            if junction.any() and straight.any():
                neff_junction_values.append(neff[junction])
                neff_straight_values.append(neff[straight])
                neff_deltas.append(float(neff[junction].mean() - neff[straight].mean()))
                gate_deltas.append(float(gate[junction].mean() - gate[straight].mean()))
            valid_gate = junction | straight | np.asarray(sample["gate_hard_negative_mask"], dtype=bool)
            if junction.any() and np.any(valid_gate & ~junction):
                gate_truth.append(junction[valid_gate].astype(np.uint8))
                gate_score.append(gate[valid_gate])

            correction = diagnostics["correction"][0].abs().mean(dim=0).cpu().numpy()
            base_abs = diagnostics["base_output"][0].abs().mean().item()
            masks = {
                "straight": straight,
                "junction": junction,
                "positive_gap": np.asarray(sample["positive_gap_mask"], dtype=bool),
                "negative_gap": np.asarray(sample["negative_gap_mask"], dtype=bool),
            }
            row: dict[str, Any] = {
                "candidate_id": spec.candidate_id,
                "index": int(sample["index"]),
                "case": sample["case"],
                **{f"family_a_{key}": value for key, value in family_a.items() if key != "family"},
                **{key: value for key, value in family_b.items() if isinstance(value, (int, float)) or value is None},
                "orientation_error_model_modes_median_deg": float(np.median(errors)) if len(errors) else None,
                "membership_set_kl": float(mode_details["membership_set_kl"]),
                "mode_count_accuracy": float(cardinality["mode_count_accuracy"]),
                "neff_mae": float(cardinality["neff_mae"]),
                "neff_junction": float(neff[junction].mean()) if junction.any() else None,
                "neff_straight": float(neff[straight].mean()) if straight.any() else None,
                "gate_junction": float(gate[junction].mean()) if junction.any() else None,
                "gate_straight": float(gate[straight].mean()) if straight.any() else None,
                "correction_abs_mean": float(correction.mean()),
                "correction_to_base_abs_mean_ratio": float(correction.mean() / max(base_abs, 1e-8)),
                **{
                    f"correction_{name}": float(correction[mask].mean()) if mask.any() else None
                    for name, mask in masks.items()
                },
            }
            rows.append(row)

    metrics: dict[str, Any] = {}
    family_a_names = [
        key for key, value in rows[0].items() if key.startswith("family_a_") and isinstance(value, (int, float))
    ]
    metrics.update({key.removeprefix("family_a_"): float(np.mean([row[key] for row in rows])) for key in family_a_names})
    metrics["gap_recovery_rate"] = _weighted(rows, "family_a_gap_recovery_rate", "family_a_positive_gap_count", empty=1.0)
    metrics["false_bridge_rate"] = _weighted(rows, "family_a_false_bridge_rate", "family_a_negative_gap_count", empty=0.0)
    metrics["positive_gap_count"] = int(sum(row["family_a_positive_gap_count"] for row in rows))
    metrics["negative_gap_count"] = int(sum(row["family_a_negative_gap_count"] for row in rows))
    route_rows = [row for row in rows if int(row.get("route_row_count") or 0) > 0]
    for name in (
        "route_top1_hit",
        "route_true_probability_mass",
        "route_mrr",
        "route_average_precision",
        "route_entropy_normalized",
        "route_excess_over_chance",
        "topology_constrained_pairing_score",
    ):
        available = [row for row in route_rows if row.get(name) is not None]
        metrics[name] = (
            sum(float(row[name]) * int(row["route_row_count"]) for row in available)
            / sum(int(row["route_row_count"]) for row in available)
            if available else None
        )
    metrics["route_row_count"] = int(sum(int(row.get("route_row_count") or 0) for row in route_rows))
    all_errors = np.concatenate(orientation_errors) if orientation_errors else np.asarray([], dtype=float)
    metrics["orientation_error_model_modes_median_deg"] = float(np.median(all_errors)) if len(all_errors) else None
    for name in ("membership_set_kl", "mode_count_accuracy", "neff_mae", "correction_abs_mean", "correction_to_base_abs_mean_ratio"):
        metrics[name] = float(np.mean([float(row[name]) for row in rows]))
    if neff_deltas:
        mean, low, high = bootstrap_mean_ci(neff_deltas)
        metrics["neff_junction_minus_straight"] = mean
        metrics["neff_junction_minus_straight_median"] = float(np.median(np.concatenate(neff_junction_values)) - np.median(np.concatenate(neff_straight_values)))
        metrics["neff_junction_minus_straight_ci95"] = [low, high]
    else:
        metrics.update({
            "neff_junction_minus_straight": None,
            "neff_junction_minus_straight_median": None,
            "neff_junction_minus_straight_ci95": [None, None],
        })
    if gate_deltas:
        mean, low, high = bootstrap_mean_ci(gate_deltas)
        metrics["gate_junction_minus_straight"] = mean
        metrics["gate_junction_minus_straight_ci95"] = [low, high]
    else:
        metrics["gate_junction_minus_straight"] = None
        metrics["gate_junction_minus_straight_ci95"] = [None, None]
    if gate_truth:
        truth = np.concatenate(gate_truth)
        score = np.concatenate(gate_score)
        metrics["gate_auroc"] = float(roc_auc_score(truth, score))
        metrics["gate_auprc"] = float(average_precision_score(truth, score))
    else:
        metrics["gate_auroc"] = None
        metrics["gate_auprc"] = None
    for region in ("straight", "junction", "positive_gap", "negative_gap"):
        values = [float(row[f"correction_{region}"]) for row in rows if row[f"correction_{region}"] is not None]
        metrics[f"correction_{region}"] = float(np.mean(values)) if values else None
    if not all(
        math.isfinite(float(value))
        for value in metrics.values()
        if isinstance(value, (float, int))
    ):
        raise ValueError("context validation produced NaN or Inf")

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    operator = model.enc1.spatial
    cost = {"total_parameters": int(total_parameters)}
    if spec.model == "context":
        cost.update({
            "context_head_parameters": context_head_parameter_count(operator),
            "context_head_macs_per_pixel": context_head_macs_per_pixel(operator),
        })
    else:
        head_parameters = sum(parameter.numel() for parameter in operator.membership_head.parameters()) + sum(
            parameter.numel() for parameter in operator.geometry_head.parameters()
        )
        cost.update({"context_head_parameters": 0, "pointwise_head_parameters": int(head_parameters)})

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"{spec.candidate_id}-{spec.run_hash}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "COMPLETE",
        "candidate_id": spec.candidate_id,
        "run_hash": spec.run_hash,
        "sample_count": len(rows),
        "selected_visible_threshold": threshold,
        "threshold_scores": {str(key): value for key, value in threshold_scores.items()},
        "metrics": metrics,
        "cost": cost,
        "rows_csv": str(csv_path),
        "expert_data_accessed": False,
        "legacy_test_samples_opened": 0,
        "v3_test_samples_opened": 0,
        "cracks_samples_opened": 0,
    }
    (output_root / f"{spec.candidate_id}-{spec.run_hash}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def decide_context_gate(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = {spec.candidate_id for spec in context_matrix()}
    if set(summaries) != expected:
        raise ValueError("context gate requires exactly B0-B3")
    baseline = summaries["B0"]["metrics"]
    gate = CONTEXT_PROTOCOL["mechanism_gate"]
    decisions: dict[str, Any] = {}
    for candidate_id in ("B1", "B2", "B3"):
        metrics = summaries[candidate_id]["metrics"]
        checks = {
            "visible_dice_safe": metrics["visible_dice"] >= baseline["visible_dice"] - gate["visible_noninferiority_margin"],
            "visible_cldice_safe": metrics["visible_cldice"] >= baseline["visible_cldice"] - gate["visible_noninferiority_margin"],
            "route_ap": metrics["route_average_precision"] >= gate["route_average_precision_minimum"],
            "route_entropy": metrics["route_entropy_normalized"] <= gate["route_entropy_normalized_maximum"],
            "orientation": metrics["orientation_error_model_modes_median_deg"] <= gate["orientation_error_median_deg_maximum"],
            "neff_ci": metrics["neff_junction_minus_straight_ci95"][0] > gate["neff_ci95_low_minimum_exclusive"],
            "neff_mean": metrics["neff_junction_minus_straight"] >= gate["neff_mean_and_median_separation_minimum"],
            "neff_median": metrics["neff_junction_minus_straight_median"] >= gate["neff_mean_and_median_separation_minimum"],
            "membership_kl": metrics["membership_set_kl"] <= gate["membership_set_kl_maximum"],
            "gate_auroc": metrics["gate_auroc"] >= gate["gate_auroc_minimum"],
            "gate_delta_ci": metrics["gate_junction_minus_straight_ci95"][0] > gate["gate_delta_ci95_low_minimum_exclusive"],
            "negative_gap_count": metrics["negative_gap_count"] >= gate["negative_gap_count_minimum"],
            "false_bridge": metrics["false_bridge_rate"] <= gate["false_bridge_rate_maximum"],
            "gap_recovery": metrics["gap_recovery_rate"] >= gate["gap_recovery_rate_minimum"],
            "false_bridge_reduction": baseline["false_bridge_rate"] - metrics["false_bridge_rate"] >= gate["false_bridge_reduction_vs_b0_minimum"],
        }
        composite = (
            metrics["route_average_precision"]
            - metrics["route_entropy_normalized"]
            + metrics["neff_junction_minus_straight"]
            + metrics["gate_auroc"]
            - metrics["false_bridge_rate"]
        )
        decisions[candidate_id] = {
            "checks": checks,
            "all_gates_pass": all(checks.values()),
            "predeclared_composite": float(composite),
        }
    eligible = [name for name, value in decisions.items() if value["all_gates_pass"]]
    selected = max(eligible, key=lambda name: (decisions[name]["predeclared_composite"], name)) if eligible else None
    return {
        "status": "CONTEXT_MECHANISM_PASS" if selected else "CONTEXT_MECHANISM_FAIL",
        "selected_candidate": selected,
        "confirm_authorized": selected is not None,
        "cracks_authorized": False,
        "expert_data_accessed": False,
        "legacy_test_samples_opened": 0,
        "v3_test_samples_opened": 0,
        "decisions": decisions,
    }


def write_context_gate(validation_root: Path, output_path: Path) -> dict[str, Any]:
    summaries = {
        spec.candidate_id: json.loads(
            (Path(validation_root) / f"{spec.candidate_id}-{spec.run_hash}.json").read_text()
        )
        for spec in context_matrix()
    }
    result = decide_context_gate(summaries)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
