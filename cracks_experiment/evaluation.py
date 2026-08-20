"""Expert evaluation for frozen Setting A models.

This module is deliberately guarded by the crowd-validation freeze receipt.
Importing it does not read expert annotations; the guarded runner does so only
after every model and threshold is frozen.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, label
import torch

from cracks_experiment.matrix import CRACKSRunSpec, PROJECT_ROOT, setting_a_matrix, setting_a_protocol_hash
from cracks_experiment.training import NORMALIZATION, build_real_model, load_real_checkpoint
from cracks_experiment.validation import _binary_metrics, _sha256, tiled_probability
from datasets.cracks import load_rgb_mask, map_mask_rgb
from trace_extraction.geometry import local_pca_orientation
from trace_extraction.metrics import compute_trace_metrics
from trace_extraction.skeleton import skeletonize_mask


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_threshold_freeze(training_root: Path) -> dict[str, Any]:
    training_root = Path(training_root)
    receipt_path = training_root / "threshold_freeze.json"
    if not receipt_path.exists():
        raise PermissionError("Expert evaluation locked: threshold freeze receipt missing")
    receipt = json.loads(receipt_path.read_text())
    freeze_sha = receipt.pop("freeze_sha256", None)
    if freeze_sha != _canonical_hash(receipt):
        raise PermissionError("Expert evaluation locked: invalid threshold freeze hash")
    if receipt.get("status") != "FROZEN" or receipt.get("expert_scores_used") is not False:
        raise PermissionError("Expert evaluation locked: invalid freeze provenance")
    if receipt.get("protocol_hash") != setting_a_protocol_hash():
        raise PermissionError("Expert evaluation locked: protocol changed after freeze")
    by_run = {row.get("run_id"): row for row in receipt.get("runs", [])}
    if set(by_run) != {spec.run_id for spec in setting_a_matrix()}:
        raise PermissionError("Expert evaluation locked: incomplete frozen run matrix")
    for spec in setting_a_matrix():
        frozen = by_run[spec.run_id]
        run_dir = training_root / f"{spec.run_id}-{spec.run_hash}"
        checkpoint = run_dir / "checkpoint-last.pt"
        validation = run_dir / "crowd_validation.json"
        if (
            frozen.get("run_hash") != spec.run_hash
            or frozen.get("checkpoint_sha256") != _sha256(checkpoint)
            or frozen.get("validation_sha256") != _sha256(validation)
        ):
            raise PermissionError(f"Expert evaluation locked: frozen artifact changed for {spec.run_id}")
    return {**receipt, "freeze_sha256": freeze_sha}


def hard_cldice(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    pred_skeleton = skeletonize_mask(pred)
    target_skeleton = skeletonize_mask(truth)
    if not pred.any() and not truth.any():
        return 1.0
    topology_precision = float((pred_skeleton & truth).sum() / pred_skeleton.sum()) if pred_skeleton.any() else 0.0
    topology_recall = float((target_skeleton & pred).sum() / target_skeleton.sum()) if target_skeleton.any() else 0.0
    denominator = topology_precision + topology_recall
    return 2.0 * topology_precision * topology_recall / denominator if denominator else 0.0


def fragmentation(predicted_skeleton: np.ndarray, target_skeleton: np.ndarray, tolerance: int = 2) -> float:
    """Mean excess predicted components intersecting each dilated GT component."""
    pred_labels, _ = label(np.asarray(predicted_skeleton, dtype=bool), structure=np.ones((3, 3)))
    target_labels, target_count = label(np.asarray(target_skeleton, dtype=bool), structure=np.ones((3, 3)))
    if target_count == 0:
        return 0.0 if not np.asarray(predicted_skeleton).any() else 1.0
    structure = np.ones((3, 3), dtype=bool)
    values = []
    for component_id in range(1, target_count + 1):
        region = target_labels == component_id
        for _ in range(int(tolerance)):
            region = binary_dilation(region, structure=structure)
        intersecting = np.unique(pred_labels[region])
        count = int(np.count_nonzero(intersecting))
        values.append(max(0, count - 1))
    return float(np.mean(values))


def evaluate_binary_section(
    probability: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    *,
    orientation_radius: int = 5,
    orientation_sensitivity_radii: tuple[int, ...] = (),
) -> dict[str, float | int]:
    valid_bool = np.asarray(valid, dtype=bool)
    prediction = (np.asarray(probability) >= float(threshold)) & valid_bool
    truth = np.asarray(target, dtype=bool) & valid_bool
    pixel = _binary_metrics(prediction, truth, valid_bool)
    pred_skeleton = skeletonize_mask(prediction)
    target_skeleton = skeletonize_mask(truth)
    pred_orientation = local_pca_orientation(pred_skeleton, radius=orientation_radius)
    trace = compute_trace_metrics(
        pred_skeleton,
        target_skeleton,
        pred_orientation=pred_orientation,
        tolerance=2.0,
        pca_radius=orientation_radius,
    )
    result: dict[str, float | int] = {
        **pixel,
        "cldice": hard_cldice(prediction, truth),
        "skeleton_f1_at_2px": float(trace["trace_f1"]),
        "fragmentation": fragmentation(pred_skeleton, target_skeleton, tolerance=2),
        **{f"trace_{key}": value for key, value in trace.items()},
    }
    result[f"orientation_error_median_deg_r{orientation_radius}"] = float(
        trace["orientation_error_median_deg"]
    )
    for radius in orientation_sensitivity_radii:
        if int(radius) == int(orientation_radius):
            continue
        sensitivity_orientation = local_pca_orientation(pred_skeleton, radius=int(radius))
        sensitivity = compute_trace_metrics(
            pred_skeleton,
            target_skeleton,
            pred_orientation=sensitivity_orientation,
            tolerance=2.0,
            pca_radius=int(radius),
        )
        result[f"orientation_error_median_deg_r{int(radius)}"] = float(
            sensitivity["orientation_error_median_deg"]
        )
    if not all(math.isfinite(float(value)) for value in result.values()):
        raise ValueError("Expert evaluator produced NaN or Inf")
    return result


def _normalized_section(section_id: int) -> torch.Tensor:
    from datasets.cracks import load_section_image

    image = load_section_image(PROJECT_ROOT / "data" / "cracks" / "images" / f"section_{section_id:03d}.png")
    tensor = torch.from_numpy(image.transpose(2, 0, 1))
    mean = torch.tensor(NORMALIZATION["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(NORMALIZATION["std"], dtype=torch.float32).view(3, 1, 1).clamp_min(1e-6)
    tensor = (tensor - mean) / std
    return torch.nn.functional.pad(tensor, (0, 3, 0, 1))


def run_setting_a_expert_evaluation(
    spec: CRACKSRunSpec,
    training_root: Path,
    output_root: Path,
    *,
    device: str = "cuda",
    max_sections: int | None = None,
) -> dict[str, Any]:
    receipt = verify_threshold_freeze(training_root)
    frozen = next(row for row in receipt["runs"] if row["run_id"] == spec.run_id)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / f"{spec.run_id}-{spec.run_hash}.json"
    rows_path = output_root / f"{spec.run_id}-{spec.run_hash}.csv"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if (
            existing.get("status") == "COMPLETE"
            and existing.get("freeze_sha256") == receipt["freeze_sha256"]
            and existing.get("section_limit") == max_sections
        ):
            return {**existing, "action": "SKIP"}

    run_dir = Path(training_root) / f"{spec.run_id}-{spec.run_hash}"
    model = build_real_model(spec).to(torch.device(device))
    load_real_checkpoint(run_dir / "checkpoint-last.pt", spec.run_hash, model)
    model.eval()
    protocol = json.loads((PROJECT_ROOT / "results" / "anza_v2_study" / "protocol.json").read_text())
    section_ids = list(protocol["setting_a"]["expert_evaluation_sections"])
    if max_sections is not None:
        section_ids = section_ids[: int(max_sections)]
    rows: list[dict[str, Any]] = []
    for index, section_id in enumerate(section_ids):
        probability = tiled_probability(model, _normalized_section(section_id)).numpy()[:255, :701]
        rgb = load_rgb_mask(
            PROJECT_ROOT / "data" / "cracks" / "annotations" / "expert" / f"section_{section_id:03d}.png"
        )
        for policy in ("paper_like", "conservative"):
            target, valid, _confidence = map_mask_rgb(rgb, policy)
            metrics = evaluate_binary_section(
                probability,
                target >= 0.5,
                valid,
                frozen["selected_threshold"],
                orientation_sensitivity_radii=(3, 7),
            )
            rows.append(
                {
                    "run_id": spec.run_id,
                    "run_hash": spec.run_hash,
                    "model": spec.model,
                    "seed": spec.seed,
                    "section_id": section_id,
                    "policy": policy,
                    "threshold": frozen["selected_threshold"],
                    **metrics,
                }
            )
        print(
            f"phase=cracks_setting_a_expert model={spec.run_id} section={index + 1}/{len(section_ids)} "
            "status=RUNNING"
        )
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    primary = [row for row in rows if row["policy"] == "paper_like"]
    summary_keys = ("dice", "iou", "cldice", "skeleton_f1_at_2px", "fragmentation", "trace_orientation_error_median_deg")
    summary = {key: float(np.mean([float(row[key]) for row in primary])) for key in summary_keys}
    payload = {
        "status": "COMPLETE",
        "action": "RUN",
        "run_id": spec.run_id,
        "run_hash": spec.run_hash,
        "freeze_sha256": receipt["freeze_sha256"],
        "section_limit": max_sections,
        "section_count": len(section_ids),
        "primary_policy": "paper_like",
        "sensitivity_policy": "conservative",
        "expert_scores_used_after_freeze": True,
        "summary": summary,
    }
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def finalize_setting_a_expert_evaluation(
    training_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Record that the complete frozen Setting A matrix was evaluated once."""
    receipt = verify_threshold_freeze(training_root)
    output_root = Path(output_root)
    runs = []
    for spec in setting_a_matrix():
        result_path = output_root / f"{spec.run_id}-{spec.run_hash}.json"
        rows_path = output_root / f"{spec.run_id}-{spec.run_hash}.csv"
        if not result_path.exists() or not rows_path.exists():
            raise FileNotFoundError(f"Expert result missing for {spec.run_id}")
        result = json.loads(result_path.read_text())
        if (
            result.get("status") != "COMPLETE"
            or result.get("freeze_sha256") != receipt["freeze_sha256"]
            or result.get("section_limit") is not None
            or result.get("section_count") != 40
        ):
            raise ValueError(f"Incomplete expert result for {spec.run_id}")
        runs.append(
            {
                "run_id": spec.run_id,
                "run_hash": spec.run_hash,
                "result_sha256": _sha256(result_path),
                "rows_sha256": _sha256(rows_path),
            }
        )
    core = {
        "status": "COMPLETE",
        "threshold_freeze_sha256": receipt["freeze_sha256"],
        "run_count": len(runs),
        "expert_section_count": 40,
        "runs": runs,
    }
    payload = {**core, "sha256": _canonical_hash(core)}
    path = output_root / "complete.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Existing Setting A expert completion receipt differs")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
