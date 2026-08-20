"""Human-to-expert agreement and section-level disagreement analyses."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
import torch

from cracks_experiment.evaluation import evaluate_binary_section
from cracks_experiment.finetuning import verify_setting_a_complete
from cracks_experiment.matrix import PROJECT_ROOT, setting_a_matrix
from cracks_experiment.training import NORMALIZATION, build_real_model, load_real_checkpoint
from cracks_experiment.validation import _blend_window, _tile_starts
from datasets.cracks import fuse_crowd_masks, load_rgb_mask, load_section_image, map_mask_rgb


def annotator_role(name: str) -> str:
    if name.startswith("novice"):
        return "novice"
    if name.startswith("practitioner"):
        return "practitioner"
    if name == "expert":
        return "expert"
    raise ValueError(f"Unknown CRACKS annotator role: {name}")


def run_human_baseline(
    setting_a_root: Path,
    setting_a_expert_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    setting_a_receipt = verify_setting_a_complete(setting_a_root, setting_a_expert_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "summary.json"
    rows_path = output_root / "annotator_sections.csv"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("status") == "COMPLETE" and existing.get("setting_a_receipt_sha256") == setting_a_receipt["sha256"]:
            return {**existing, "action": "SKIP"}

    protocol = json.loads((PROJECT_ROOT / "results" / "anza_v2_study" / "protocol.json").read_text())
    expert_ids = list(protocol["setting_a"]["expert_evaluation_sections"])
    annotation_root = PROJECT_ROOT / "data" / "cracks" / "annotations"
    annotators = sorted(
        path.name for path in annotation_root.iterdir()
        if path.is_dir() and path.name != "expert"
    )
    rows: list[dict[str, Any]] = []
    for index, section_id in enumerate(expert_ids):
        name = f"section_{section_id:03d}.png"
        expert_rgb = load_rgb_mask(annotation_root / "expert" / name)
        for annotator in annotators:
            candidate_path = annotation_root / annotator / name
            if not candidate_path.exists():
                continue
            candidate_rgb = load_rgb_mask(candidate_path)
            for policy in ("paper_like", "conservative"):
                expert_target, expert_valid, _ = map_mask_rgb(expert_rgb, policy)
                human_target, human_valid, _ = map_mask_rgb(candidate_rgb, policy)
                common_valid = expert_valid & human_valid
                metrics = evaluate_binary_section(
                    human_target,
                    expert_target >= 0.5,
                    common_valid,
                    0.5,
                )
                rows.append(
                    {
                        "section_id": section_id,
                        "annotator": annotator,
                        "role": annotator_role(annotator),
                        "policy": policy,
                        "common_valid_pixel_count": int(common_valid.sum()),
                        **metrics,
                    }
                )
        print(
            f"phase=cracks_human_baseline section={index + 1}/{len(expert_ids)} "
            f"annotations={sum(row['section_id'] == section_id for row in rows)} status=RUNNING"
        )
    if not rows:
        raise ValueError("No crowd annotations overlap the available expert sections")
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, dict[str, float | int]] = {}
    primary = [row for row in rows if row["policy"] == "paper_like"]
    keys = ("dice", "cldice", "skeleton_f1_at_2px", "trace_orientation_error_median_deg")
    for role in ("novice", "practitioner"):
        selected = [row for row in primary if row["role"] == role]
        summary[role] = {
            "annotation_section_count": len(selected),
            **{f"median_{key}": float(np.median([float(row[key]) for row in selected])) for key in keys},
        }
    payload = {
        "status": "COMPLETE",
        "action": "RUN",
        "claim_boundary": "agreement with the available expert annotation; not human superiority",
        "setting_a_receipt_sha256": setting_a_receipt["sha256"],
        "expert_section_count": len(expert_ids),
        "annotator_count": len(annotators),
        "row_count": len(rows),
        "primary_policy": "paper_like",
        "sensitivity_policy": "conservative",
        "summary": summary,
    }
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


@torch.no_grad()
def tiled_v2_uncertainty(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    tile_size: int = 256,
    overlap: int = 64,
) -> dict[str, np.ndarray]:
    """Blend first-layer V2 geometry and incoming transport entropy."""
    if image.ndim != 3:
        raise ValueError("Expected normalized CHW image")
    device = next(model.parameters()).device
    height, width = image.shape[-2:]
    window = _blend_window(tile_size, overlap, device)[0, 0]
    names = (
        "probability", "routing_entropy", "rho", "one_minus_rho",
        "junction_score", "anisotropy", "orientation_cos2", "orientation_sin2",
    )
    sums = {name: torch.zeros((height, width), device=device) for name in names}
    weights = torch.zeros((height, width), device=device)
    for top in _tile_starts(height, tile_size, overlap):
        for left in _tile_starts(width, tile_size, overlap):
            tile = image[:, top : top + tile_size, left : left + tile_size].unsqueeze(0).to(device)
            output = model(tile, return_diagnostics=True)
            diagnostics = output["transport_diagnostics"]
            if not diagnostics:
                raise ValueError("V2 uncertainty requires model-native transport diagnostics")
            first = diagnostics[0]
            membership = first["membership"]
            theta = first["theta"]
            sigma_u = first["sigma_u"]
            sigma_s = first["sigma_s"]
            c_value = (membership * torch.cos(2.0 * theta)).sum(dim=1)
            s_value = (membership * torch.sin(2.0 * theta)).sum(dim=1)
            rho = torch.sqrt(c_value.square() + s_value.square()).clamp(0.0, 1.0)
            per_mode_anisotropy = torch.tanh(torch.abs(torch.log(sigma_u / sigma_s)))
            anisotropy = (rho * (membership * per_mode_anisotropy).sum(dim=1)).clamp(0.0, 1.0)
            transition = first["transport"].float().reshape(1, -1, tile_size * tile_size)
            incoming = transition / transition.sum(dim=1, keepdim=True).clamp_min(1e-8)
            routing_entropy = -(
                incoming * torch.log(incoming.clamp_min(1e-8))
            ).sum(dim=1) / np.log(incoming.shape[1])
            tile_maps = {
                "probability": torch.sigmoid(output["visible_logits"])[0, 0],
                "routing_entropy": routing_entropy.reshape(tile_size, tile_size).clamp(0.0, 1.0),
                "rho": rho[0],
                "one_minus_rho": 1.0 - rho[0],
                "junction_score": first["junction_score"][0].clamp(0.0, 1.0),
                "anisotropy": anisotropy[0],
                "orientation_cos2": c_value[0],
                "orientation_sin2": s_value[0],
            }
            region = (slice(top, top + tile_size), slice(left, left + tile_size))
            for name, value in tile_maps.items():
                sums[name][region] += value * window
            weights[region] += window
    if torch.any(weights <= 0):
        raise AssertionError("V2 diagnostic tiling left uncovered pixels")
    result = {name: (value / weights).cpu().numpy() for name, value in sums.items()}
    result["orientation"] = 0.5 * np.arctan2(
        result.pop("orientation_sin2"), result.pop("orientation_cos2")
    )
    bounded = {key: value for key, value in result.items() if key != "orientation"}
    if not all(np.isfinite(value).all() and np.all((0.0 <= value) & (value <= 1.0)) for value in bounded.values()):
        raise ValueError("V2 uncertainty maps must be finite and bounded")
    if not np.isfinite(result["orientation"]).all():
        raise ValueError("V2 orientation map must be finite")
    return result


def section_bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    resamples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Section bootstrap requires matching finite one-dimensional arrays")
    observed = float(spearmanr(x, y).statistic)
    if not np.isfinite(observed):
        return {"status": "NOT_ESTIMABLE", "spearman_r": None, "ci95_low": None, "ci95_high": None, "n_sections": len(x)}
    generator = np.random.default_rng(seed)
    samples = []
    for _ in range(int(resamples)):
        indices = generator.integers(0, len(x), size=len(x))
        value = float(spearmanr(x[indices], y[indices]).statistic)
        if np.isfinite(value):
            samples.append(value)
    if not samples:
        return {"status": "NOT_ESTIMABLE", "spearman_r": observed, "ci95_low": None, "ci95_high": None, "n_sections": len(x)}
    return {
        "status": "COMPLETE",
        "spearman_r": observed,
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
        "n_sections": len(x),
        "bootstrap_resamples": int(resamples),
        "bootstrap_unit": "seismic_section",
    }


def _normalized_image(section_id: int) -> torch.Tensor:
    image = load_section_image(
        PROJECT_ROOT / "data" / "cracks" / "images" / f"section_{section_id:03d}.png"
    )
    tensor = torch.from_numpy(image.transpose(2, 0, 1))
    mean = torch.tensor(NORMALIZATION["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(NORMALIZATION["std"], dtype=torch.float32).view(3, 1, 1).clamp_min(1e-6)
    return torch.nn.functional.pad((tensor - mean) / std, (0, 3, 0, 1))


def run_disagreement_analysis(
    setting_a_root: Path,
    setting_a_expert_root: Path,
    output_root: Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    setting_a_receipt = verify_setting_a_complete(setting_a_root, setting_a_expert_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "summary.json"
    sections_path = output_root / "section_metrics.csv"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("status") == "COMPLETE" and existing.get("setting_a_receipt_sha256") == setting_a_receipt["sha256"]:
            return {**existing, "action": "SKIP"}

    spec = next(
        candidate for candidate in setting_a_matrix()
        if candidate.run_id == "v2_s42" and candidate.comparison_family == "main"
    )
    run_dir = Path(setting_a_root) / f"{spec.run_id}-{spec.run_hash}"
    validation = json.loads((run_dir / "crowd_validation.json").read_text())
    threshold = float(validation["selected_threshold"])
    model = build_real_model(spec).to(torch.device(device))
    load_real_checkpoint(run_dir / "checkpoint-last.pt", spec.run_hash, model)
    model.eval()
    protocol = json.loads((PROJECT_ROOT / "results" / "anza_v2_study" / "protocol.json").read_text())
    section_ids = list(protocol["setting_a"]["expert_evaluation_sections"])
    annotation_root = PROJECT_ROOT / "data" / "cracks" / "annotations"
    annotators = sorted(
        path.name for path in annotation_root.iterdir()
        if path.is_dir() and path.name != "expert"
    )
    rows = []
    for index, section_id in enumerate(section_ids):
        name = f"section_{section_id:03d}.png"
        available = [annotator for annotator in annotators if (annotation_root / annotator / name).exists()]
        masks = [load_rgb_mask(annotation_root / annotator / name) for annotator in available]
        crowd = fuse_crowd_masks(masks, available, "paper_like", minimum_disagreement_support=5)
        expert_target, expert_valid, _ = map_mask_rgb(load_rgb_mask(annotation_root / "expert" / name), "paper_like")
        maps = tiled_v2_uncertainty(model, _normalized_image(section_id))
        maps = {key: value[:255, :701] for key, value in maps.items()}
        analysis_mask = crowd["human_entropy_valid"] & expert_valid
        if not analysis_mask.any():
            raise ValueError(f"No supported disagreement pixels for expert section {section_id}")
        prediction = maps["probability"] >= threshold
        row = {
            "section_id": section_id,
            "crowd_annotator_count": len(available),
            "analysis_pixel_count": int(analysis_mask.sum()),
            "mean_human_entropy": float(crowd["human_entropy"][analysis_mask].mean()),
            "mean_routing_entropy": float(maps["routing_entropy"][analysis_mask].mean()),
            "mean_one_minus_rho": float(maps["one_minus_rho"][analysis_mask].mean()),
            "mean_junction_score": float(maps["junction_score"][analysis_mask].mean()),
            "mean_anisotropy": float(maps["anisotropy"][analysis_mask].mean()),
            "model_error_rate": float((prediction[analysis_mask] != (expert_target[analysis_mask] >= 0.5)).mean()),
        }
        if not np.isfinite(list(row.values())).all():
            raise ValueError("Disagreement analysis produced NaN or Inf")
        rows.append(row)
        print(f"phase=cracks_disagreement section={index + 1}/{len(section_ids)} status=RUNNING")
    with sections_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    human = np.asarray([row["mean_human_entropy"] for row in rows])
    correlations = {}
    for offset, metric in enumerate(
        ("mean_routing_entropy", "mean_one_minus_rho", "model_error_rate", "mean_junction_score", "mean_anisotropy")
    ):
        correlations[metric] = section_bootstrap_spearman(
            human,
            np.asarray([row[metric] for row in rows]),
            resamples=2000,
            seed=42 + offset,
        )
    payload = {
        "status": "COMPLETE",
        "action": "RUN",
        "setting_a_receipt_sha256": setting_a_receipt["sha256"],
        "model_run_id": spec.run_id,
        "model_run_hash": spec.run_hash,
        "threshold": threshold,
        "section_count": len(rows),
        "statistical_unit": "seismic_section",
        "routing_entropy_definition": "normalized entropy of incoming first-layer V2B transport weights per destination pixel",
        "correlations": correlations,
    }
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
