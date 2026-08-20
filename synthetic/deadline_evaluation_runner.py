"""Batch inference for the audited synthetic evaluator; never retrains models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch

from models.segmentation_v2 import build_comparable_model
from synthetic.crossing_trace_bench import generate_sample
from synthetic.evaluation import continuation_probabilities
from synthetic.evaluation_corrected import evaluate_sample_corrected
from synthetic.evaluator_audit import (
    audit_gap_predictions,
    corrected_evaluator_hash,
    false_bridge_verdict,
    snapshot_legacy_outputs,
)
from synthetic.experiment_matrix import SyntheticRunSpec, development_matrix
from synthetic.geometry_generator import GEOMETRY_TYPES, generate_geometry, scale_geometry
from synthetic.instance_targets import rasterize_targets
from synthetic.seismic_background import render_seismic
from synthetic.structural_losses import branch_transition_logits
from synthetic.training import load_checkpoint
from trace_extraction.geometry import local_pca_orientation
from trace_extraction.skeleton import skeletonize_mask


CANDIDATE_IDS = ("B0", "B1", "C0", "C3")
VALIDATION_RANGE = (0, 256)
LEGACY_TEST_RANGE = (0, 2000)
REPLACEMENT_TEST_RANGE = (2000, 4000)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extended_test_sample(index: int, *, image_size: int = 128) -> dict[str, Any]:
    """Extend the frozen deterministic test RNG stream without changing legacy code."""

    if not 0 <= int(index) < REPLACEMENT_TEST_RANGE[1]:
        raise IndexError(index)
    if index < LEGACY_TEST_RANGE[1]:
        return generate_sample("test", index, image_size=image_size)
    seed = 30_000_000 + int(index)
    rng = np.random.default_rng(seed)
    selected_case = GEOMETRY_TYPES[int(rng.integers(0, len(GEOMETRY_TYPES)))]
    geometry = scale_geometry(generate_geometry(selected_case, rng), image_size)
    return {
        "image": render_seismic(geometry, image_size, rng),
        **rasterize_targets(geometry, image_size),
        "case": selected_case,
        "split": "test",
        "index": int(index),
        "seed": seed,
        "image_size": int(image_size),
        "scientific_scope": "controlled structural benchmark; not an F3 physical simulator",
    }


def _specs() -> dict[str, SyntheticRunSpec]:
    return {spec.candidate_id: spec for spec in development_matrix() if spec.candidate_id in CANDIDATE_IDS}


def _frozen_context(study_root: Path) -> dict[str, Any]:
    root = Path(study_root) / "synthetic"
    specs = _specs()
    context: dict[str, Any] = {"specs": specs, "thresholds": {}, "checkpoints": {}, "checkpoint_hashes": {}}
    for candidate_id in CANDIDATE_IDS:
        spec = specs[candidate_id]
        checkpoint = root / "development" / f"{candidate_id}-{spec.run_hash}" / "checkpoint-last.pt"
        status_path = checkpoint.with_name("status.json")
        validation_path = root / "validation" / f"{candidate_id}-{spec.run_hash}.json"
        if not checkpoint.exists() or not status_path.exists() or not validation_path.exists():
            raise FileNotFoundError(f"Frozen artifacts missing for {candidate_id}")
        status = json.loads(status_path.read_text())
        validation = json.loads(validation_path.read_text())
        if status.get("status") != "COMPLETE" or status.get("run_hash", spec.run_hash) != spec.run_hash:
            raise RuntimeError(f"Frozen training status mismatch for {candidate_id}")
        if "selected_visible_threshold" not in validation:
            raise RuntimeError(f"Validation-selected threshold missing for {candidate_id}")
        context["thresholds"][candidate_id] = float(validation["selected_visible_threshold"])
        context["checkpoints"][candidate_id] = checkpoint
        context["checkpoint_hashes"][candidate_id] = _sha256(checkpoint)
    return context


def _load_frozen_model(spec: SyntheticRunSpec, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = build_comparable_model(spec.model).to(device)
    load_checkpoint(checkpoint, expected_hash=spec.run_hash, model=model)
    model.eval()
    return model


def _mean_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for row in rows for key in row})
    result: dict[str, Any] = {}
    for key in keys:
        values = [row.get(key) for row in rows]
        numeric = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if numeric:
            result[key] = float(np.mean(numeric))
        elif any(value is not None for value in values):
            result[key] = next(value for value in values if value is not None)
        else:
            result[key] = None
    return result


def _bootstrap_mean_ci(values: list[float], *, seed: int = 20260817, resamples: int = 10_000) -> tuple[float, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    for start in range(0, resamples, 500):
        count = min(500, resamples - start)
        means.append(array[rng.integers(0, len(array), size=(count, len(array)))].mean(axis=1))
    samples = np.concatenate(means)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _summarize_gap_audits(audits: list[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_count": len(audits),
        "primary_coverage_threshold": 0.5,
        "threshold_selection_permitted": False,
    }
    for gap_type in ("positive", "negative"):
        rows = [row for audit in audits for row in audit[gap_type]["rows"]]
        result[gap_type] = {"count": len(rows), "metrics": _mean_metrics(rows) if rows else {}}
    negative_rows = [row for audit in audits for row in audit["negative"]["rows"]]
    bridge_count = sum(
        bool(row["coverage_at_0.50"] >= 0.5 and row["connected_at_0.50"])
        for row in negative_rows
    )
    result["false_bridge_count_at_fixed_0_5"] = int(bridge_count)
    result["false_bridge_rate_at_fixed_0_5"] = (
        float(bridge_count / len(negative_rows)) if negative_rows else 0.0
    )
    return result


def _evaluate_indices(
    study_root: Path,
    indices: Iterable[int],
    sample_factory: Callable[[int], dict[str, Any]],
    *,
    split_label: str,
    device: str,
) -> dict[str, Any]:
    context = _frozen_context(study_root)
    torch_device = torch.device(device)
    indexed_samples = list(indices)
    summaries: dict[str, Any] = {}
    geometry_rows: list[Mapping[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        spec = context["specs"][candidate_id]
        model = _load_frozen_model(spec, context["checkpoints"][candidate_id], torch_device)
        metric_rows: list[dict[str, Any]] = []
        gap_rows: list[Mapping[str, Any]] = []
        with torch.no_grad():
            for index in indexed_samples:
                sample = sample_factory(index)
                image = torch.as_tensor(sample["image"], device=torch_device).unsqueeze(0)
                output = model(image, return_diagnostics=True)
                visible_probability = torch.sigmoid(output["visible_logits"])[0, 0].cpu().numpy()
                visible_prediction = visible_probability >= context["thresholds"][candidate_id]
                completion_probability = visible_probability
                if "completion_logits" in output:
                    completion_head = torch.sigmoid(output["completion_logits"])[0, 0].cpu().numpy()
                    completion_probability = np.maximum(visible_probability, completion_head)
                    completion_prediction = visible_prediction | (completion_head >= 0.5)
                else:
                    completion_prediction = visible_prediction
                routing = None
                diagnostics = output.get("transport_diagnostics", [])
                if diagnostics:
                    first = diagnostics[0]
                    logits = branch_transition_logits(
                        first["transport"],
                        torch.as_tensor(sample["branch_masks"], device=torch_device),
                        variant=first["variant"],
                    )
                    routing = continuation_probabilities(
                        logits,
                        torch.as_tensor(sample["continuation_eligible_matrix"], device=torch_device),
                    ).cpu().numpy()
                orientation = local_pca_orientation(skeletonize_mask(completion_prediction))
                corrected = evaluate_sample_corrected(
                    visible_prediction,
                    sample,
                    predicted_completion_mask=completion_prediction,
                    predicted_orientation=orientation,
                    routing_probabilities=routing,
                    has_transport_output=routing is not None,
                    include_geometry_diagnostic=candidate_id == CANDIDATE_IDS[0],
                )
                metric_rows.append(
                    {
                        "index": int(index),
                        "seed": int(sample["seed"]),
                        "case": sample["case"],
                        "strata": ";".join(sample["strata"]),
                        **corrected["family_a"],
                        "route_available": corrected["family_b"]["available"],
                        **{
                            key: value
                            for key, value in corrected["family_b"].items()
                            if key.startswith("route_")
                            or key.startswith("topology_")
                            or key.startswith("x_")
                            or key.startswith("t_")
                            or key.startswith("y_")
                            or key.startswith("legacy_")
                        },
                    }
                )
                gap_rows.append(audit_gap_predictions(sample, completion_probability))
                if corrected["family_c"] is not None:
                    geometry_rows.append(corrected["family_c"]["metrics"])
        metrics = _mean_metrics(metric_rows)
        route_ci = _bootstrap_mean_ci(
            [float(row["route_excess_over_chance"]) for row in metric_rows if row.get("route_excess_over_chance") is not None]
        )
        metrics["route_excess_over_chance_ci95_low"] = route_ci[0] if route_ci else None
        metrics["route_excess_over_chance_ci95_high"] = route_ci[1] if route_ci else None
        summaries[candidate_id] = {
            "model": spec.model,
            "run_hash": spec.run_hash,
            "checkpoint_sha256": context["checkpoint_hashes"][candidate_id],
            "validation_selected_visible_threshold": context["thresholds"][candidate_id],
            "sample_count": len(metric_rows),
            "metrics": metrics,
            "gap_audit": _summarize_gap_audits(gap_rows),
            "rows": metric_rows,
        }
        del model
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
    method_gap = {candidate_id: summary["gap_audit"] for candidate_id, summary in summaries.items()}
    geometry_summary = {
        "uses_generator_branch_geometry": True,
        "is_model_specific": False,
        "metrics": _mean_metrics(geometry_rows),
    }
    table_rows = [
        {
            "candidate_id": candidate_id,
            "model": summary["model"],
            "uses_generator_branch_geometry": False,
            **summary["metrics"],
        }
        for candidate_id, summary in summaries.items()
    ]
    table_rows.append(
        {
            "candidate_id": "GEOMETRY_ONLY",
            "model": "geometry_only_minimum_angle_heuristic",
            "uses_generator_branch_geometry": True,
            **geometry_summary["metrics"],
        }
    )
    return {
        "status": "COMPLETE",
        "split": split_label,
        "indices": [indexed_samples[0], indexed_samples[-1] + 1] if indexed_samples else [0, 0],
        "sample_count": len(indexed_samples),
        "corrected_evaluator_sha256": corrected_evaluator_hash(),
        "checkpoint_hashes": context["checkpoint_hashes"],
        "visible_thresholds": context["thresholds"],
        "models": summaries,
        "geometry_only_minimum_angle_heuristic": geometry_summary,
        "synthetic_corrected_rows": table_rows,
        "false_bridge_verdict": false_bridge_verdict(method_gap),
    }


def _limited_range(bounds: tuple[int, int], max_samples: int | None) -> range:
    start, stop = bounds
    if max_samples is None:
        return range(start, stop)
    if int(max_samples) <= 0:
        raise ValueError("max_samples must be positive")
    return range(start, min(stop, start + int(max_samples)))


def compute_validation_audit(study_root: Path, *, device: str = "cuda", max_samples: int | None = None) -> dict[str, Any]:
    indices = _limited_range(VALIDATION_RANGE, max_samples)
    result = _evaluate_indices(
        study_root,
        indices,
        lambda index: generate_sample("validation", index, image_size=128),
        split_label="validation",
        device=device,
    )
    result["test_samples_opened"] = 0
    result["scientific_scope"] = "VALIDATION_ONLY_EVALUATOR_AND_FALSE_BRIDGE_AUDIT"
    return result


def compute_legacy_reanalysis(study_root: Path, *, device: str = "cuda", max_samples: int | None = None) -> dict[str, Any]:
    study_root = Path(study_root)
    legacy_root = study_root / "synthetic" / "test"
    before = snapshot_legacy_outputs(legacy_root)
    result = _evaluate_indices(
        study_root,
        _limited_range(LEGACY_TEST_RANGE, max_samples),
        _extended_test_sample,
        split_label="test[0:2000]",
        device=device,
    )
    if snapshot_legacy_outputs(legacy_root) != before:
        raise RuntimeError("Legacy test outputs changed during post-hoc reanalysis")
    legacy_summary_path = legacy_root / "summary.json"
    legacy_summary = json.loads(legacy_summary_path.read_text()) if legacy_summary_path.exists() else None
    differences: dict[str, dict[str, float]] = {}
    if legacy_summary is not None:
        for candidate_id, corrected in result["models"].items():
            legacy_metrics = legacy_summary.get("models", {}).get(candidate_id, {}).get("metrics", {})
            differences[candidate_id] = {
                key: float(value) - float(legacy_metrics[key])
                for key, value in corrected["metrics"].items()
                if isinstance(value, (int, float))
                and key in legacy_metrics
                and isinstance(legacy_metrics[key], (int, float))
            }
    result.update(
        {
            "status": "POSTHOC_REANALYSIS_NOT_CONFIRMATORY",
            "legacy_output_hashes": before,
            "legacy_summary": legacy_summary,
            "corrected_minus_legacy": differences,
            "legacy_originals_modified": False,
            "diff_scope": "corrected families versus immutable legacy summary",
        }
    )
    return result


def compute_replacement_confirmation(study_root: Path, *, device: str = "cuda", max_samples: int | None = None) -> dict[str, Any]:
    study_root = Path(study_root)
    root = study_root / "synthetic" / "replacement_confirmation"
    freeze_path = root / "freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("Corrected evaluator freeze must exist before replacement opening")
    freeze = json.loads(freeze_path.read_text())
    context = _frozen_context(study_root)
    expected = {
        "corrected_evaluator_sha256": corrected_evaluator_hash(),
        "model_checkpoint_hashes": context["checkpoint_hashes"],
        "visible_thresholds": context["thresholds"],
        "replacement_test_indices": list(REPLACEMENT_TEST_RANGE),
    }
    for key, value in expected.items():
        if freeze.get(key) != value:
            raise RuntimeError(f"Replacement freeze mismatch: {key}")
    receipt_path = root / "open_receipt.json"
    if receipt_path.exists():
        raise RuntimeError("Replacement stream was already opened; rerun refused")
    indices = _limited_range(REPLACEMENT_TEST_RANGE, max_samples)
    receipt = {
        "status": "OPENED_FOR_SINGLE_EVALUATION",
        "scientific_status": "CONFIRMATORY" if max_samples is None else "SMOKE_ONLY",
        "frozen_range": list(REPLACEMENT_TEST_RANGE),
        "executed_range": [indices.start, indices.stop],
        "freeze_sha256": _sha256(freeze_path),
        **expected,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    result = _evaluate_indices(
        study_root,
        indices,
        _extended_test_sample,
        split_label="test[2000:4000]",
        device=device,
    )
    result.update(
        {
            "status": "REPLACEMENT_CONFIRMATION_AFTER_EVALUATOR_AUDIT",
            "scientific_result": max_samples is None,
            "open_receipt": str(receipt_path),
            "freeze_sha256": receipt["freeze_sha256"],
            "no_retraining": True,
        }
    )
    return result
