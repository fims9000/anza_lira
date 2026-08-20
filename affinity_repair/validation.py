"""Validation-only mechanism metrics and fixed gates for C0--C3."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from affinity_repair.matrix import AFFINITY_PROTOCOL, AffinityRepairSpec, affinity_matrix
from affinity_repair.training import build_candidate, cached_v4_sample, load_checkpoint
from models.azconv_affinity import LOCAL8_OFFSETS, RADIUS2_OFFSETS, StructuralAffinityAZConv2d
from synthetic.affinity_targets import build_affinity_targets
from synthetic.evaluation_corrected import evaluate_sample_corrected


HARD_STRATA = {
    "acute_angle_crossing": {"acute_angle_crossing"},
    "similar_tangent_crossing": {"similar_tangent_crossing"},
    "nontrivial_pairing": {"nontrivial_pairing"},
    "crossing_near_junction": {"crossing_near_junction"},
    "near_parallel_close": {"near_parallel", "close_non_intersecting"},
    "matched_negative_gap": {"negative_gap"},
}


def bootstrap_mean_ci(values: Iterable[float], *, resamples: int = 10_000, seed: int = 42) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("bootstrap requires finite values")
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(int(resamples), len(array)))].mean(axis=1)
    return [float(array.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _dice(probability: np.ndarray, target: np.ndarray, threshold: float) -> float:
    prediction = probability >= threshold
    truth = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum() + truth.sum())
    return 2.0 * int((prediction & truth).sum()) / denominator if denominator else 1.0


def _set_affinity(model: torch.nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, StructuralAffinityAZConv2d):
            module.set_affinity_enabled(enabled)


def _weighted(rows: list[dict[str, Any]], metric: str, count: str, *, empty: float) -> float:
    denominator = sum(int(row[count]) for row in rows)
    return (
        sum(float(row[metric]) * int(row[count]) for row in rows) / denominator
        if denominator else float(empty)
    )


def _affinity_metrics(edge_rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_stratum: dict[str, dict[str, Any]] = {}
    aps: list[float] = []
    for name, cases in HARD_STRATA.items():
        selected = [row for row in edge_rows if row["case"] in cases]
        truth = np.concatenate([row["truth"] for row in selected]) if selected else np.asarray([], dtype=np.uint8)
        score = np.concatenate([row["score"] for row in selected]) if selected else np.asarray([], dtype=float)
        if len(truth) and len(np.unique(truth)) == 2:
            ap = float(average_precision_score(truth, score))
            auc = float(roc_auc_score(truth, score))
            aps.append(ap)
        else:
            ap = None
            auc = None
        per_stratum[name] = {
            "edge_count": int(len(truth)),
            "positive_count": int(truth.sum()) if len(truth) else 0,
            "average_precision": ap,
            "auroc": auc,
        }
    separations = [float(row["separation"]) for row in edge_rows if row["separation"] is not None]
    true_scores = np.concatenate([row["true_scores"] for row in edge_rows if len(row["true_scores"])])
    false_scores = np.concatenate([row["false_scores"] for row in edge_rows if len(row["false_scores"])])
    return {
        "hard_affinity_macro_ap": float(np.mean(aps)) if len(aps) == len(HARD_STRATA) else None,
        "per_stratum": per_stratum,
        "matched_negative_gap_auroc": per_stratum["matched_negative_gap"]["auroc"],
        "affinity_positive_mass_true": float(true_scores.mean()) if len(true_scores) else None,
        "affinity_positive_mass_false": float(false_scores.mean()) if len(false_scores) else None,
        "true_minus_false_affinity_ci95": bootstrap_mean_ci(separations) if separations else [None, None, None],
    }


def evaluate_candidate(
    spec: AffinityRepairSpec,
    development_root: Path,
    output_root: Path,
    *,
    device: str = "cuda",
    sample_count: int = 512,
) -> dict[str, Any]:
    run_dir = Path(development_root) / f"{spec.candidate_id}-{spec.run_hash}"
    status = json.loads((run_dir / "status.json").read_text())
    if status.get("status") != "COMPLETE" or status.get("run_hash") != spec.run_hash:
        raise ValueError(f"incomplete candidate: {spec.candidate_id}")
    for field in ("expert_data_accessed", "legacy_test_samples_opened", "v4_test_samples_opened", "cracks_samples_opened"):
        expected = False if field == "expert_data_accessed" else 0
        if status.get(field) != expected:
            raise ValueError(f"validation lock violation: {field}")
    widths = tuple(int(value) for value in status["widths"])
    image_size = int(status["image_size"])
    clean_checkpoint = None
    clean_state = None
    clean_sha = status.get("clean_checkpoint_sha256")
    if spec.affinity:
        clean_spec = affinity_matrix()[1]
        clean_checkpoint = Path(development_root) / f"C1-{clean_spec.run_hash}" / "checkpoint-last.pt"
        clean_state = torch.load(clean_checkpoint, map_location="cpu", weights_only=False)["model_state"]
    model = build_candidate(spec, widths=widths, clean_state=clean_state).to(torch.device(device))
    load_checkpoint(
        run_dir / "checkpoint-last.pt", spec=spec, model=model,
        clean_checkpoint_sha256=clean_sha,
    )
    model.eval()

    cached: list[tuple[dict[str, Any], np.ndarray]] = []
    with torch.inference_mode():
        for start in range(0, sample_count, 8):
            samples = [cached_v4_sample("validation", index, image_size) for index in range(start, min(start + 8, sample_count))]
            images = torch.stack([torch.as_tensor(sample["image"]) for sample in samples]).to(device)
            probability = torch.sigmoid(model(images))[:, 0].cpu().numpy()
            cached.extend(zip(samples, probability))
    threshold_scores = {
        float(threshold): float(np.mean([
            _dice(probability, sample["visible_fault_mask"], float(threshold))
            for sample, probability in cached
        ]))
        for threshold in AFFINITY_PROTOCOL["threshold_candidates"]
    }
    threshold = max(threshold_scores, key=lambda value: (threshold_scores[value], -value))

    rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    causal_deltas: list[float] = []
    batch_size = 8
    with torch.inference_mode():
        for start in range(0, sample_count, batch_size):
            batch_cached = cached[start : start + batch_size]
            samples = [item[0] for item in batch_cached]
            images = torch.stack([torch.as_tensor(sample["image"]) for sample in samples]).to(device)
            if spec.affinity:
                _set_affinity(model, True)
                on_probability = torch.sigmoid(model(images))[:, 0].cpu().numpy()
                layer = next(module for module in model.modules() if isinstance(module, StructuralAffinityAZConv2d))
                edge = layer.edge_logits(images, include_radius2=spec.radius2)
                edge_probability = torch.sigmoid(edge["logits"]).cpu().numpy()
                _set_affinity(model, False)
                off_probability = torch.sigmoid(model(images))[:, 0].cpu().numpy()
                _set_affinity(model, True)
            else:
                on_probability = np.stack([item[1] for item in batch_cached])
                off_probability = on_probability
                edge_probability = None
            for local, sample in enumerate(samples):
                predicted = on_probability[local] >= threshold
                on_metrics = evaluate_sample_corrected(
                    predicted, sample, predicted_completion_mask=predicted
                )["family_a"]
                off_predicted = off_probability[local] >= threshold
                off_metrics = evaluate_sample_corrected(
                    off_predicted, sample, predicted_completion_mask=off_predicted
                )["family_a"]
                causal_delta = float(on_metrics["latent_skeleton_f1_2px"] - off_metrics["latent_skeleton_f1_2px"])
                causal_deltas.append(causal_delta)
                rows.append({
                    "candidate_id": spec.candidate_id,
                    "index": int(sample["index"]),
                    "case": sample["case"],
                    **{key: value for key, value in on_metrics.items() if key != "family"},
                    "beta_on_minus_off_latent_skeleton_f1": causal_delta,
                })
                if edge_probability is not None:
                    offsets = LOCAL8_OFFSETS + (RADIUS2_OFFSETS if spec.radius2 else ())
                    target = build_affinity_targets(sample, offsets)
                    positive = target["affinity_positive"]
                    negative = target["affinity_hard_negative"]
                    selected = positive | negative
                    score = edge_probability[local]
                    true_score = score[positive]
                    false_score = score[negative]
                    edge_rows.append({
                        "case": sample["case"],
                        "truth": positive[selected].astype(np.uint8),
                        "score": score[selected],
                        "true_scores": true_score,
                        "false_scores": false_score,
                        "separation": float(true_score.mean() - false_score.mean()) if len(true_score) and len(false_score) else None,
                    })

    metrics: dict[str, Any] = {}
    scalar_names = [
        "visible_dice", "visible_iou", "visible_precision", "visible_recall", "visible_cldice",
        "latent_cldice", "latent_skeleton_f1_2px", "junction_f1", "endpoint_f1",
        "false_merge_rate", "false_split_rate", "fragmentation", "symmetric_skeleton_distance",
    ]
    for name in scalar_names:
        metrics[name] = float(np.mean([float(row[name]) for row in rows]))
    metrics["gap_recovery_rate"] = _weighted(rows, "gap_recovery_rate", "positive_gap_count", empty=1.0)
    metrics["false_bridge_rate"] = _weighted(rows, "false_bridge_rate", "negative_gap_count", empty=0.0)
    metrics["positive_gap_count"] = int(sum(int(row["positive_gap_count"]) for row in rows))
    metrics["negative_gap_count"] = int(sum(int(row["negative_gap_count"]) for row in rows))
    metrics["beta_on_minus_off_latent_skeleton_f1_ci95"] = bootstrap_mean_ci(causal_deltas)
    metrics["learned_beta"] = float(next(
        (module.beta.detach().cpu() for module in model.modules() if isinstance(module, StructuralAffinityAZConv2d)),
        torch.tensor(0.0),
    ))
    metrics.update(_affinity_metrics(edge_rows) if spec.affinity else {
        "hard_affinity_macro_ap": None,
        "per_stratum": {},
        "matched_negative_gap_auroc": None,
        "affinity_positive_mass_true": None,
        "affinity_positive_mass_false": None,
        "true_minus_false_affinity_ci95": [None, None, None],
    })
    if not all(math.isfinite(float(value)) for value in metrics.values() if isinstance(value, (int, float))):
        raise ValueError("non-finite validation metric")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"{spec.candidate_id}-{spec.run_hash}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "COMPLETE", "candidate_id": spec.candidate_id, "run_hash": spec.run_hash,
        "sample_count": len(rows), "selected_visible_threshold": threshold,
        "threshold_scores": {str(key): value for key, value in threshold_scores.items()},
        "metrics": metrics, "rows_csv": str(csv_path),
        "expert_data_accessed": False, "legacy_test_samples_opened": 0,
        "v4_test_samples_opened": 0, "cracks_samples_opened": 0,
    }
    (output_root / f"{spec.candidate_id}-{spec.run_hash}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def decide_affinity_gate(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(summaries) != {"C0", "C1", "C2", "C3"}:
        raise ValueError("affinity gate requires exactly C0-C3")
    baseline = summaries["C0"]["metrics"]
    decisions: dict[str, Any] = {}
    for candidate_id in ("C2", "C3"):
        metrics = summaries[candidate_id]["metrics"]
        macro_ap = metrics["hard_affinity_macro_ap"]
        negative_auc = metrics["matched_negative_gap_auroc"]
        checks = {
            "visible_dice_safe": metrics["visible_dice"] >= baseline["visible_dice"] - 0.005,
            "visible_cldice_safe": metrics["visible_cldice"] >= baseline["visible_cldice"] - 0.005,
            "hard_macro_ap": macro_ap is not None and macro_ap >= 0.85,
            "negative_gap_auroc": negative_auc is not None and negative_auc >= 0.85,
            "affinity_separation_ci": metrics["true_minus_false_affinity_ci95"][1] is not None and metrics["true_minus_false_affinity_ci95"][1] > 0.0,
            "causal_topology_ci": metrics["beta_on_minus_off_latent_skeleton_f1_ci95"][1] > 0.0,
            "gap_recovery": metrics["gap_recovery_rate"] >= 0.88,
            "false_bridge": metrics["false_bridge_rate"] <= 0.50,
            "false_bridge_reduction": baseline["false_bridge_rate"] - metrics["false_bridge_rate"] >= 0.25,
            "topology_not_worse": metrics["latent_skeleton_f1_2px"] >= baseline["latent_skeleton_f1_2px"] - 0.005,
            "endpoint_not_worse": metrics["endpoint_f1"] >= baseline["endpoint_f1"] - 0.01,
        }
        decisions[candidate_id] = {"checks": checks, "all_gates_pass": all(checks.values())}
    eligible = [name for name, decision in decisions.items() if decision["all_gates_pass"]]
    selected = max(eligible, key=lambda name: (summaries[name]["metrics"]["hard_affinity_macro_ap"], name)) if eligible else None
    return {
        "status": "AFFINITY_MECHANISM_PASS" if selected else "AFFINITY_MECHANISM_FAIL",
        "selected_candidate": selected,
        "confirm_authorized": selected is not None,
        "cracks_authorized": False,
        "decisions": decisions,
        "expert_data_accessed": False,
        "legacy_test_samples_opened": 0,
        "v4_test_samples_opened": 0,
        "cracks_samples_opened": 0,
    }


def write_affinity_gate(validation_root: Path, output_path: Path) -> dict[str, Any]:
    summaries = {
        spec.candidate_id: json.loads((Path(validation_root) / f"{spec.candidate_id}-{spec.run_hash}.json").read_text())
        for spec in affinity_matrix()
    }
    result = decide_affinity_gate(summaries)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result

