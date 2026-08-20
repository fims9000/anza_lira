"""Validation-only mechanism evaluation and fail-closed A0-A4 gate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from method_repair.matrix import COMMON_PROTOCOL, MethodRepairSpec, synthetic_matrix
from method_repair.training import build_candidate_model, cached_sample, load_candidate_checkpoint
from synthetic.evaluation import continuation_probabilities
from synthetic.evaluation_corrected import evaluate_sample_corrected
from synthetic.mode_supervision import (
    axial_mode_set_loss,
    branch_mode_masks_from_tangents,
    mode_specific_branch_transition_logits,
)
from trace_extraction.geometry import local_pca_orientation
from trace_extraction.skeleton import skeletonize_mask


def _visible_dice(probability: np.ndarray, target: np.ndarray, threshold: float) -> float:
    prediction = probability >= threshold
    truth = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum() + truth.sum())
    return 2.0 * int((prediction & truth).sum()) / denominator if denominator else 1.0


def _effective_mode_count(membership: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    mu = membership / membership.sum(dim=1, keepdim=True).clamp_min(epsilon)
    return torch.exp(-(mu * mu.clamp_min(epsilon).log()).sum(dim=1))


def _mean_selected(values: np.ndarray, mask: np.ndarray) -> float | None:
    return float(values[mask].mean()) if np.asarray(mask, dtype=bool).any() else None


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap requires non-empty finite values")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(int(resamples), len(array)))
    means = array[indices].mean(axis=1)
    return float(array.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _weighted_metric(rows: list[dict[str, Any]], name: str, count_name: str) -> float:
    denominator = sum(float(row[count_name]) for row in rows)
    if denominator == 0:
        return 0.0
    return sum(float(row[name]) * float(row[count_name]) for row in rows) / denominator


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_a_names = [
        name
        for name, value in rows[0].items()
        if name.startswith("family_a_") and isinstance(value, (int, float))
    ]
    aggregate = {name.removeprefix("family_a_"): float(np.mean([row[name] for row in rows])) for name in family_a_names}
    aggregate["gap_recovery_rate"] = _weighted_metric(rows, "family_a_gap_recovery_rate", "family_a_positive_gap_count")
    aggregate["false_bridge_rate"] = _weighted_metric(rows, "family_a_false_bridge_rate", "family_a_negative_gap_count")
    aggregate["positive_gap_count"] = int(sum(row["family_a_positive_gap_count"] for row in rows))
    aggregate["negative_gap_count"] = int(sum(row["family_a_negative_gap_count"] for row in rows))
    route_rows = [row for row in rows if row.get("route_row_count", 0) > 0]
    route_names = (
        "route_top1_hit",
        "route_true_probability_mass",
        "route_mrr",
        "route_average_precision",
        "route_entropy_normalized",
        "chance_top1",
        "route_excess_over_chance",
        "topology_constrained_pairing_score",
    )
    total_route_rows = sum(int(row.get("route_row_count", 0)) for row in route_rows)
    for name in route_names:
        available = [row for row in route_rows if row.get(name) is not None]
        aggregate[name] = (
            sum(float(row[name]) * int(row["route_row_count"]) for row in available)
            / sum(int(row["route_row_count"]) for row in available)
            if available
            else None
        )
    aggregate["route_row_count"] = total_route_rows
    diagnostic_names = (
        "neff_junction",
        "neff_straight",
        "ambiguity_junction",
        "ambiguity_straight",
        "gate_junction",
        "gate_straight",
        "orientation_set_loss",
        "membership_set_kl",
    )
    for name in diagnostic_names:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        aggregate[name] = float(np.mean(values)) if values else None
    for prefix in ("neff", "ambiguity"):
        deltas = [
            float(row[f"{prefix}_junction"]) - float(row[f"{prefix}_straight"])
            for row in rows
            if row.get(f"{prefix}_junction") is not None and row.get(f"{prefix}_straight") is not None
        ]
        if deltas:
            mean, low, high = bootstrap_mean_ci(deltas)
            aggregate[f"{prefix}_junction_minus_straight"] = mean
            aggregate[f"{prefix}_junction_minus_straight_ci95"] = [low, high]
            aggregate[f"{prefix}_paired_sample_count"] = len(deltas)
        else:
            aggregate[f"{prefix}_junction_minus_straight"] = None
            aggregate[f"{prefix}_junction_minus_straight_ci95"] = [None, None]
            aggregate[f"{prefix}_paired_sample_count"] = 0
    return aggregate


def evaluate_validation_candidate(
    spec: MethodRepairSpec,
    development_root: Path,
    output_root: Path,
    *,
    device: str = "cuda",
    sample_count: int = 256,
) -> dict[str, Any]:
    run_dir = Path(development_root) / f"{spec.candidate_id}-{spec.run_hash}"
    status = json.loads((run_dir / "status.json").read_text())
    if status.get("status") != "COMPLETE" or status.get("run_hash") != spec.run_hash:
        raise ValueError(f"candidate training incomplete or mismatched: {spec.candidate_id}")
    if status.get("expert_data_accessed") is not False:
        raise ValueError("candidate selection cannot use expert data")
    image_size = int(status["image_size"])
    widths = tuple(int(value) for value in status["widths"])
    torch_device = torch.device(device)
    model = build_candidate_model(spec, widths=widths).to(torch_device)
    load_candidate_checkpoint(
        run_dir / "checkpoint-last.pt",
        expected_hash=spec.run_hash,
        model=model,
    )
    model.eval()
    cached_probabilities: list[tuple[dict[str, Any], np.ndarray]] = []
    with torch.inference_mode():
        for index in range(int(sample_count)):
            sample = cached_sample("validation", index, image_size)
            image = torch.as_tensor(sample["image"], device=torch_device).unsqueeze(0)
            probability = torch.sigmoid(model(image))[0, 0].cpu().numpy()
            cached_probabilities.append((sample, probability))
    threshold_scores = {
        float(threshold): float(np.mean([
            _visible_dice(probability, sample["visible_fault_mask"], float(threshold))
            for sample, probability in cached_probabilities
        ]))
        for threshold in COMMON_PROTOCOL["threshold_candidates"]
    }
    threshold = max(threshold_scores, key=lambda value: (threshold_scores[value], -value))

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for sample, probability in cached_probabilities:
            image = torch.as_tensor(sample["image"], device=torch_device).unsqueeze(0)
            output = model(image, return_diagnostics=True)
            visible = probability >= threshold
            orientation = local_pca_orientation(skeletonize_mask(visible))
            routing_probability = None
            mechanism: dict[str, Any] = {
                "neff_junction": None,
                "neff_straight": None,
                "ambiguity_junction": None,
                "ambiguity_straight": None,
                "gate_junction": None,
                "gate_straight": None,
                "orientation_set_loss": None,
                "membership_set_kl": None,
            }
            has_transport = bool(output["transport_diagnostics"])
            if has_transport:
                diagnostics = output["transport_diagnostics"][0]
                branch_modes = branch_mode_masks_from_tangents(
                    diagnostics["theta"],
                    torch.as_tensor(sample["gt_branch_theta"], device=torch_device).float().unsqueeze(0),
                    torch.as_tensor(sample["gt_branch_theta_valid"], device=torch_device).bool().unsqueeze(0),
                )[0]
                route_logits = mode_specific_branch_transition_logits(
                    diagnostics["transport"],
                    branch_modes,
                    kernel_size=spec.routing_kernel_size,
                )
                routing_probability = continuation_probabilities(
                    route_logits,
                    torch.as_tensor(sample["continuation_eligible_matrix"], device=torch_device).bool(),
                ).cpu().numpy()
                mode_loss, details = axial_mode_set_loss(
                    diagnostics["theta"],
                    diagnostics["membership"],
                    torch.as_tensor(sample["gt_theta_set"], device=torch_device).float().unsqueeze(0),
                    torch.as_tensor(sample["gt_theta_valid"], device=torch_device).bool().unsqueeze(0),
                )
                effective = _effective_mode_count(diagnostics["membership"])[0].cpu().numpy()
                ambiguity = diagnostics["ambiguity"][0].cpu().numpy()
                gate = diagnostics["ambiguity_gate"][0].cpu().numpy()
                junction = np.asarray(sample["junction_map"], dtype=bool)
                straight = np.asarray(sample["visible_fault_mask"], dtype=bool) & ~junction
                mechanism = {
                    "neff_junction": _mean_selected(effective, junction),
                    "neff_straight": _mean_selected(effective, straight),
                    "ambiguity_junction": _mean_selected(ambiguity, junction),
                    "ambiguity_straight": _mean_selected(ambiguity, straight),
                    "gate_junction": _mean_selected(gate, junction),
                    "gate_straight": _mean_selected(gate, straight),
                    "orientation_set_loss": float(details["orientation_set_loss"]),
                    "membership_set_kl": float(details["membership_set_kl"]),
                }
                if not np.isfinite(float(mode_loss)):
                    raise ValueError("mode diagnostics produced a non-finite loss")
            evaluated = evaluate_sample_corrected(
                visible,
                sample,
                predicted_completion_mask=visible,
                predicted_orientation=orientation,
                routing_probabilities=routing_probability,
                has_transport_output=has_transport,
            )
            family_a = evaluated["family_a"]
            family_b = evaluated["family_b"]
            rows.append({
                "candidate_id": spec.candidate_id,
                "index": int(sample["index"]),
                "case": sample["case"],
                "strata": ";".join(sample["strata"]),
                **{f"family_a_{name}": value for name, value in family_a.items() if name != "family"},
                **{name: value for name, value in family_b.items() if isinstance(value, (int, float)) or value is None},
                **mechanism,
            })

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
        "metrics": _aggregate_rows(rows),
        "rows_csv": str(csv_path),
        "expert_data_accessed": False,
        "old_test_samples_opened": 0,
        "new_test_samples_opened": 0,
    }
    result_path = output_root / f"{spec.candidate_id}-{spec.run_hash}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def decide_mechanism_gate(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = {spec.candidate_id for spec in synthetic_matrix()}
    if set(summaries) != expected:
        raise ValueError("mechanism gate requires exactly A0-A4")
    baseline = summaries["A0"]["metrics"]
    criteria = COMMON_PROTOCOL["mechanism_gate"]
    decisions: dict[str, Any] = {}
    for candidate_id in ("A1", "A2", "A3", "A4"):
        metrics = summaries[candidate_id]["metrics"]
        checks = {
            "neff_junction_gt_straight": metrics["neff_junction_minus_straight_ci95"][0] > criteria["neff_junction_minus_straight_bootstrap_ci_low_gt"],
            "ambiguity_junction_gt_straight": metrics["ambiguity_junction_minus_straight_ci95"][0] > criteria["ambiguity_junction_minus_straight_bootstrap_ci_low_gt"],
            "route_ap_above_old_c3": metrics["route_average_precision"] is not None and metrics["route_average_precision"] > criteria["old_c3_route_average_precision"] + criteria["route_average_precision_minimum_gain_over_old_c3"],
            "route_excess_over_chance": metrics["route_excess_over_chance"] is not None and metrics["route_excess_over_chance"] >= criteria["route_excess_over_chance_minimum"],
            "route_entropy_below_near_uniform": metrics["route_entropy_normalized"] is not None and metrics["route_entropy_normalized"] <= criteria["route_entropy_normalized_maximum"],
            "false_bridge_not_saturated": metrics["false_bridge_rate"] < criteria["false_bridge_rate_maximum_exclusive"],
            "visible_dice_noninferior": metrics["visible_dice"] >= baseline["visible_dice"] - criteria["visible_dice_noninferiority_margin"],
            "visible_cldice_noninferior": metrics["visible_cldice"] >= baseline["visible_cldice"] - criteria["visible_cldice_noninferiority_margin"],
        }
        composite = (
            float(metrics["route_average_precision"] or 0.0)
            + float(metrics["route_mrr"] or 0.0)
            - float(metrics["route_entropy_normalized"] or 1.0)
            + float(metrics["neff_junction_minus_straight"] or 0.0)
            + float(metrics["ambiguity_junction_minus_straight"] or 0.0)
            - float(metrics["false_bridge_rate"])
        )
        decisions[candidate_id] = {
            "checks": checks,
            "all_gates_pass": all(checks.values()),
            "predeclared_composite": composite,
        }
    eligible = [name for name, item in decisions.items() if item["all_gates_pass"]]
    selected = max(eligible, key=lambda name: (decisions[name]["predeclared_composite"], name)) if eligible else None
    return {
        "status": "SYNTHETIC_MECHANISM_PASS" if selected else "SYNTHETIC_MECHANISM_FAIL",
        "selected_candidate": selected,
        "cracks_authorized": selected is not None,
        "expert_data_accessed": False,
        "old_test_samples_opened": 0,
        "new_test_samples_opened": 0,
        "decisions": decisions,
    }


def write_mechanism_gate(
    validation_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    summaries = {
        spec.candidate_id: json.loads(
            (Path(validation_root) / f"{spec.candidate_id}-{spec.run_hash}.json").read_text()
        )
        for spec in synthetic_matrix()
    }
    gate = decide_mechanism_gate(summaries)
    gate["validation_summary_sha256"] = {
        name: hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        for name, payload in summaries.items()
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    return gate
