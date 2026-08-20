"""Predicted-endpoint v6 development gate for generic and ANZA-guided paths."""

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
from path_completion.anza_guided import (
    anza_guided_conductance,
    filtered_endpoint_pairs,
    image_conductance,
    widest_path_tiebroken,
)
from path_completion.calibration import _canonical_hash
from path_completion.learned_confirm import load_frozen_pair_classifier, score_pair
from path_completion.widest_path import EndpointPair, rasterize_path
from synthetic.crossing_trace_bench_v3 import PAIRED_GAP_COUNT
from synthetic.crossing_trace_bench_v6 import benchmark_v6_config, freeze_benchmark_v6_config, generate_sample_v6
from synthetic.evaluation_corrected import evaluate_sample_corrected
from synthetic.experiment_matrix import development_matrix
from synthetic.training import load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATH_THRESHOLDS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
REALISTIC_PROTOCOL = {
    "version": "anza_predicted_endpoint_v6_development_v1",
    "benchmark_v6_sha256": benchmark_v6_config()["sha256"],
    "models": {
        "unet_generic": "frozen synthetic B0 U-Net plus image conductance",
        "legacy_anza_guided": "frozen synthetic C0 legacy ANZA plus image-times-ANZA conductance",
    },
    "visible_thresholds": {"unet_generic": 0.25, "legacy_anza_guided": 0.15},
    "pair_classifier": "unchanged frozen v5 classifier with frozen validation calibration",
    "endpoint_source": "predicted binary mask only",
    "d_min_px": 3.0,
    "d_max_px": 18.0,
    "min_branch_length_px": 8.0,
    "border_margin_px": 5,
    "endpoint_evaluator_tolerance_px": 5.0,
    "image_contrast_scale": 0.15,
    "corridor_margin_px": 12,
    "path_threshold_candidates": list(PATH_THRESHOLDS),
    "path_selection": "maximize latent clDice among all-gate development cells; ties prefer higher threshold",
    "gates": {
        "recovery_min": 0.70,
        "false_bridge_max": 0.05,
        "visible_dice_margin": -0.005,
        "latent_cldice_non_decrease": True,
        "endpoint_f1_improvement_min": 0.03,
    },
    "v6_test": "LOCKED_UNOPENED",
    "expert": "FORBIDDEN",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_base_models(device: torch.device) -> dict[str, tuple[torch.nn.Module, float, str]]:
    specs = {spec.candidate_id: spec for spec in development_matrix()}
    selected = {
        "unet_generic": (specs["B0"], 0.25),
        "legacy_anza_guided": (specs["C0"], 0.15),
    }
    output = {}
    for name, (spec, threshold) in selected.items():
        run_dir = PROJECT_ROOT / "results/anza_v2_study/synthetic/development" / f"{spec.candidate_id}-{spec.run_hash}"
        status = json.loads((run_dir / "status.json").read_text())
        if status.get("status") != "COMPLETE" or status.get("run_hash") != spec.run_hash:
            raise ValueError(f"frozen synthetic base is incomplete: {name}")
        model = build_comparable_model(spec.model).to(device)
        checkpoint = run_dir / "checkpoint-last.pt"
        load_checkpoint(checkpoint, expected_hash=spec.run_hash, model=model)
        output[name] = (model.eval(), float(threshold), _sha256(checkpoint))
    return output


def _pair_matches_truth(pair: EndpointPair, sample: dict[str, Any], tolerance: float) -> bool:
    gap = np.asarray(sample["gaps"][0]["endpoint_xy"], dtype=float)[:, ::-1]
    direct = math.dist(pair.first, gap[0]) <= tolerance and math.dist(pair.second, gap[1]) <= tolerance
    reverse = math.dist(pair.first, gap[1]) <= tolerance and math.dist(pair.second, gap[0]) <= tolerance
    return bool(direct or reverse)


def _candidate_paths(
    sample: dict[str, Any],
    prediction: np.ndarray,
    relation: np.ndarray,
    pair_model: torch.nn.Module,
    pair_frozen: dict[str, Any],
    calibration: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = filtered_endpoint_pairs(
        prediction,
        d_min=float(REALISTIC_PROTOCOL["d_min_px"]),
        d_max=float(REALISTIC_PROTOCOL["d_max_px"]),
        min_branch_length=float(REALISTIC_PROTOCOL["min_branch_length_px"]),
        border_margin=int(REALISTIC_PROTOCOL["border_margin_px"]),
    )
    inference_sample = {**sample, "visible_fault_mask": prediction}
    temperature = float(calibration["temperature"])
    pair_threshold = float(calibration["selected_operating_point"]["threshold"])
    scored = []
    for pair in pairs:
        raw_probability = score_pair(pair_model, inference_sample, pair, device)
        logit = float(np.log(np.clip(raw_probability, 1e-12, 1 - 1e-12) / np.clip(1 - raw_probability, 1e-12, 1)))
        probability = float(1 / (1 + np.exp(-logit / temperature)))
        scored.append((probability, pair))
    accepted = []
    used_endpoints: set[tuple[int, int]] = set()
    for probability, pair in sorted(scored, key=lambda item: (-item[0], item[1].distance, item[1].first, item[1].second)):
        if probability < pair_threshold or pair.first in used_endpoints or pair.second in used_endpoints:
            continue
        bottleneck, path, costs = widest_path_tiebroken(relation, pair)
        if path:
            accepted.append({"pair": pair, "pair_probability": probability, "bottleneck": bottleneck, "path": path, **costs})
            used_endpoints.update((pair.first, pair.second))
    diagnostic = {
        "candidate_count": len(pairs),
        "accepted_pair_count": len(accepted),
        "truth_pair_candidate_present": bool(sample["case"] == "fault_with_gap" and any(
            _pair_matches_truth(pair, sample, float(REALISTIC_PROTOCOL["endpoint_evaluator_tolerance_px"])) for pair in pairs
        )),
    }
    return accepted, diagnostic


def _completion(prediction: np.ndarray, paths: list[dict[str, Any]], threshold: float) -> np.ndarray:
    result = np.asarray(prediction, dtype=bool).copy()
    for item in paths:
        if float(item["bottleneck"]) >= float(threshold):
            result |= rasterize_path(item["path"], result.shape, radius=3)
    return result


def _evaluate_cell(samples: list[dict[str, Any]], records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    rows = []
    for sample, record in zip(samples, records):
        prediction = record["prediction"]
        completion = _completion(prediction, record["paths"], threshold)
        base = evaluate_sample_corrected(prediction, sample, predicted_completion_mask=prediction)["family_a"]
        completed = evaluate_sample_corrected(prediction, sample, predicted_completion_mask=completion)["family_a"]
        rows.append({
            "case": sample["case"],
            "base_visible_dice": base["visible_dice"],
            "completion_visible_dice": completed["visible_dice"],
            "base_latent_cldice": base["latent_cldice"],
            "completion_latent_cldice": completed["latent_cldice"],
            "base_endpoint_f1": base["endpoint_f1"],
            "completion_endpoint_f1": completed["endpoint_f1"],
            "gap_recovery_rate": completed["gap_recovery_rate"],
            "false_bridge_rate": completed["false_bridge_rate"],
        })
    def mean(key: str, case: str | None = None) -> float:
        values = [row[key] for row in rows if case is None or row["case"] == case]
        return float(np.mean(values))
    result = {
        "path_threshold": float(threshold),
        "positive_gap_recovery": mean("gap_recovery_rate", "fault_with_gap"),
        "false_bridge_rate": mean("false_bridge_rate", "negative_gap"),
        "base_visible_dice": mean("base_visible_dice"),
        "completion_visible_dice": mean("completion_visible_dice"),
        "base_latent_cldice": mean("base_latent_cldice"),
        "completion_latent_cldice": mean("completion_latent_cldice"),
        "base_endpoint_f1": mean("base_endpoint_f1"),
        "completion_endpoint_f1": mean("completion_endpoint_f1"),
    }
    gates = REALISTIC_PROTOCOL["gates"]
    checks = {
        "recovery": result["positive_gap_recovery"] >= float(gates["recovery_min"]),
        "false_bridge": result["false_bridge_rate"] <= float(gates["false_bridge_max"]),
        "visible_dice": result["completion_visible_dice"] >= result["base_visible_dice"] + float(gates["visible_dice_margin"]),
        "latent_cldice": result["completion_latent_cldice"] >= result["base_latent_cldice"],
        "endpoint_f1": result["completion_endpoint_f1"] >= result["base_endpoint_f1"] + float(gates["endpoint_f1_improvement_min"]),
    }
    return {**result, "checks": checks, "all_gates_pass": all(checks.values())}


def run_v6_development(project_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    root = Path(project_root)
    torch_device = torch.device(device)
    samples = [generate_sample_v6("development", index) for index in range(2 * PAIRED_GAP_COUNT)]
    bases = _load_base_models(torch_device)
    pair_model, pair_frozen = load_frozen_pair_classifier(root, torch_device)
    calibration = json.loads((root / "results/final_practical_cycle/path_calibration/calibration_freeze.json").read_text())
    model_results = {}
    for name, (model, visible_threshold, checkpoint_sha) in bases.items():
        records = []
        diagnostics = []
        for start in range(0, len(samples), 16):
            batch = torch.stack([torch.as_tensor(sample["image"]) for sample in samples[start : start + 16]]).to(torch_device)
            with torch.inference_mode():
                probabilities = torch.sigmoid(model(batch))[:, 0]
                if name == "legacy_anza_guided":
                    relations = anza_guided_conductance(
                        model.enc1.spatial,
                        batch,
                        contrast_scale=float(REALISTIC_PROTOCOL["image_contrast_scale"]),
                    )
                else:
                    relations = image_conductance(batch, contrast_scale=float(REALISTIC_PROTOCOL["image_contrast_scale"]))
            for local, sample in enumerate(samples[start : start + 16]):
                prediction = probabilities[local].cpu().numpy() >= visible_threshold
                paths, diagnostic = _candidate_paths(
                    sample,
                    prediction,
                    relations[local].cpu().numpy(),
                    pair_model,
                    pair_frozen,
                    calibration,
                    torch_device,
                )
                records.append({"prediction": prediction, "paths": paths})
                diagnostics.append(diagnostic)
        cells = [_evaluate_cell(samples, records, threshold) for threshold in PATH_THRESHOLDS]
        passing = [cell for cell in cells if cell["all_gates_pass"]]
        selected = max(passing, key=lambda cell: (cell["completion_latent_cldice"], cell["path_threshold"])) if passing else None
        positives = diagnostics[:PAIRED_GAP_COUNT]
        model_results[name] = {
            "status": "V6_DEVELOPMENT_ELIGIBLE" if selected else "V6_DEVELOPMENT_GATE_FAIL",
            "base_checkpoint_sha256": checkpoint_sha,
            "visible_threshold": visible_threshold,
            "mean_candidate_count": float(np.mean([row["candidate_count"] for row in diagnostics])),
            "mean_accepted_pair_count": float(np.mean([row["accepted_pair_count"] for row in diagnostics])),
            "positive_truth_pair_candidate_recall": float(np.mean([row["truth_pair_candidate_present"] for row in positives])),
            "selected": selected,
            "cells": cells,
        }
    eligible = [name for name, result in model_results.items() if result["status"] == "V6_DEVELOPMENT_ELIGIBLE"]
    return {
        "status": "V6_DEVELOPMENT_ELIGIBLE" if eligible else "V6_PREDICTED_ENDPOINT_NEGATIVE",
        "protocol": REALISTIC_PROTOCOL,
        "protocol_sha256": _canonical_hash(REALISTIC_PROTOCOL),
        "models": model_results,
        "eligible_models": eligible,
        "development_samples_opened": 2 * PAIRED_GAP_COUNT,
        "v6_test_samples_opened": 0,
        "expert_data_accessed": False,
        "cracks_samples_opened": 0,
    }


def write_v6_development(output_root: Path, *, project_root: Path, device: str = "cuda") -> dict[str, Any]:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    freeze_benchmark_v6_config(output / "benchmark_v6_config.json")
    result_path = output / "development_result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("protocol_sha256") != _canonical_hash(REALISTIC_PROTOCOL):
            raise PermissionError("v6 development protocol drift")
        return {**existing, "action": "SKIP_ALREADY_FROZEN"}
    result = run_v6_development(project_root, device=device)
    rows = []
    for model, model_result in result["models"].items():
        for cell in model_result["cells"]:
            rows.append({"model": model, **{key: value for key, value in cell.items() if key != "checks"}, **{f"check_{key}": value for key, value in cell["checks"].items()}})
    with (output / "development_cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result

