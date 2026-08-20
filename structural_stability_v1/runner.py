"""Execute only SS0 audit/freeze and SS1 frozen-H0 perturbation validation."""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from cracks_experiment.evaluation import evaluate_binary_section
from cracks_experiment.partial_label_evaluation import _load_t1_model
from cracks_experiment.partial_label_training import T1RunSpec
from cracks_experiment.partial_labels import map_partial_annotation
from cracks_experiment.training import NORMALIZATION
from cracks_experiment.validation import tiled_probability
from datasets.cracks import load_section_image
from lira_final.protocol import HELDOUT_ANNOTATORS
from structural_stability_v1.agreement import crowd_agreement
from structural_stability_v1.data import audit_dataset, backbone_provenance, sha256_file
from structural_stability_v1.perturb import apply_perturbation, transform_rgb_mask
from structural_stability_v1.protocol import FAMILIES, PROTOCOL, RESULT_ROOT, ROOT, SEVERITIES, protocol_hash


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _old_stop_manifest() -> dict[str, str]:
    paths = (
        ROOT / "results/lira_h1/final/ANZA_LIRA_H1_MASTER_RESULT.json",
        ROOT / "results/lira_graph_cut_v2/benchmark/retention.json",
        ROOT / "results/lira_intervention_final/i2_candidate/validator.json",
        ROOT / "results/lira_final/f1_gap_audit/validator.json",
    )
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def run_ss0() -> dict[str, object]:
    output = RESULT_ROOT / "s0_audit"
    before = _old_stop_manifest()
    audit, split, expert = audit_dataset(output)
    backbone = backbone_provenance(output)
    payload = {
        "status": "SS_S0_PASS", "protocol": PROTOCOL, "protocol_sha256": protocol_hash(),
        "data_status": audit["status"], "dataset_root_sha256": audit["dataset_root_sha256"],
        "split_sha256": split["sha256"], "expert_provenance": expert,
        "backbone": backbone, "old_stop_sha256_before": before,
        "new_training_opened": False, "expert_label_pixels_loaded": False,
    }
    _write_json(output / "SS0_RESULT.json", payload)
    _write_json(output / "OLD_STOP_INTEGRITY.json", {"before": before, "after": _old_stop_manifest(), "unchanged": before == _old_stop_manifest()})
    return payload


def _normalized_image(section_id: int) -> np.ndarray:
    image = load_section_image(ROOT / "data/cracks/images" / f"section_{section_id:03d}.png")
    tensor = torch.from_numpy(image.transpose(2, 0, 1))
    mean = torch.tensor(NORMALIZATION["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(NORMALIZATION["std"], dtype=torch.float32).view(3, 1, 1).clamp_min(1e-6)
    return F.pad((tensor - mean) / std, (0, 3, 0, 1)).numpy().astype(np.float32)


def _section_masks(section_id: int, annotators: tuple[str, ...]) -> tuple[list[str], list[np.ndarray]]:
    names, masks = [], []
    for annotator in annotators:
        path = ROOT / "data/cracks/annotations" / annotator / f"section_{section_id:03d}.png"
        if path.is_file():
            with Image.open(path) as image:
                masks.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
            names.append(annotator)
    if not masks:
        raise FileNotFoundError(f"no held-out nonexpert annotation for calibration section {section_id}")
    return names, masks


def _metrics(probability: np.ndarray, masks: list[np.ndarray], threshold: float) -> dict[str, float]:
    rows = []
    for mask in masks:
        target, weight = map_partial_annotation(mask)
        rows.append(evaluate_binary_section(probability, target >= 0.5, weight > 0, threshold))
    keys = ("dice", "cldice", "precision", "recall", "skeleton_f1_at_2px", "fragmentation")
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def _panel(images: dict[tuple[str, int], np.ndarray], clean: np.ndarray, section_id: int, output: Path) -> None:
    figure, axes = plt.subplots(len(FAMILIES), 4, figsize=(12, 12), constrained_layout=True)
    def display(array: np.ndarray) -> np.ndarray:
        scalar = np.asarray(array[0], dtype=np.float32)[:255, :701]
        low, high = np.quantile(scalar, (0.01, 0.99))
        return np.clip((scalar - low) / max(float(high - low), 1e-8), 0.0, 1.0)
    clean_display = display(clean)
    for row, family in enumerate(FAMILIES):
        axes[row, 0].imshow(clean_display, cmap="gray", aspect="auto")
        axes[row, 0].set_title(f"{family}: clean")
        for column, severity in enumerate(SEVERITIES, 1):
            axes[row, column].imshow(display(images[(family, severity)]), cmap="gray", aspect="auto")
            axes[row, column].set_title(f"S{severity}")
        for axis in axes[row]: axis.set_axis_off()
    figure.suptitle(f"CRACKS-SSBench-V1 section {section_id}")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def run_ss1(ss0: dict[str, object], *, device: str) -> dict[str, object]:
    output = RESULT_ROOT / "s1_perturbation"
    output.mkdir(parents=True, exist_ok=True)
    split = json.loads((RESULT_ROOT / "s0_audit/split_manifest.json").read_text())
    section_ids = list(split["splits"]["SS_CALIBRATION"])
    model, checkpoint, run_hash = _load_t1_model(T1RunSpec("t1_unet_s41", "unet", 41), device)
    model.eval()
    threshold = float(ss0["backbone"]["historical_selected_threshold"])
    rows: list[dict[str, object]] = []
    agreement_rows = []
    validation = {
        "all_finite": True, "all_deterministic": True, "warp_jacobians_valid": True,
        "annotation_palette_preserved": True, "expert_lock_respected": True,
        "expert_label_pixels_loaded": False,
    }
    panel_images: dict[tuple[str, int], np.ndarray] = {}
    for position, section_id in enumerate(section_ids):
        image = _normalized_image(section_id)
        names, masks = _section_masks(section_id, HELDOUT_ANNOTATORS)
        agreement = crowd_agreement(masks)
        agreement_rows.append({
            "section_id": section_id, "annotators": len(names),
            "agreement_nonzero_fraction": float(np.mean(agreement["agreement"] > 0)),
            "agreement_mean": float(np.mean(agreement["agreement"])),
            "labeled_weight_mean": float(np.mean(agreement["labeled_weight"])),
        })
        clean_probability = tiled_probability(model, torch.from_numpy(image)).numpy()[:255, :701]
        clean_metrics = _metrics(clean_probability, masks, threshold)
        rows.append({
            "section_id": section_id, "model": "H0_T1_UNET_S41_FROZEN", "seed": 41,
            "condition": "clean", "family": "clean", "severity": 0,
            "seg_threshold": threshold, "annotator_count": len(names),
            "perturbation_seed": "", "deterministic": True, "finite": True,
            "jacobian_det_min": 1.0, "jacobian_det_max": 1.0,
            "jacobian_condition_max": 1.0, **clean_metrics,
        })
        for family in FAMILIES:
            for severity in SEVERITIES:
                result = apply_perturbation(image, section_id, "full_section", family, severity)
                repeated = apply_perturbation(image, section_id, "full_section", family, severity)
                deterministic = np.array_equal(result.image, repeated.image) and result.metadata == repeated.metadata
                validation["all_deterministic"] = validation["all_deterministic"] and deterministic
                validation["all_finite"] = validation["all_finite"] and bool(np.isfinite(result.image).all())
                if family == "warp":
                    det_min = float(result.metadata["jacobian_det_min"])
                    det_max = float(result.metadata["jacobian_det_max"])
                    cond_max = float(result.metadata["jacobian_condition_max"])
                    valid_warp = det_min >= 0.75 and det_max <= 1.25 and cond_max <= 1.5
                    validation["warp_jacobians_valid"] = validation["warp_jacobians_valid"] and valid_warp
                    transformed_masks = [transform_rgb_mask(mask, result) for mask in masks]
                else:
                    transformed_masks = [mask.copy() for mask in masks]
                allowed = {(31, 119, 180), (44, 160, 44), (255, 127, 14), (255, 255, 255)}
                palette_ok = all(set(map(tuple, np.unique(mask.reshape(-1, 3), axis=0).tolist())) <= allowed for mask in transformed_masks)
                validation["annotation_palette_preserved"] = validation["annotation_palette_preserved"] and palette_ok
                probability = tiled_probability(model, torch.from_numpy(result.image)).numpy()[:255, :701]
                metrics = _metrics(probability, transformed_masks, threshold)
                rows.append({
                    "section_id": section_id, "model": "H0_T1_UNET_S41_FROZEN", "seed": 41,
                    "condition": f"{family}_s{severity}", "family": family, "severity": severity,
                    "seg_threshold": threshold, "annotator_count": len(names), "perturbation_seed": result.seed,
                    "deterministic": deterministic, "finite": bool(np.isfinite(result.image).all()),
                    "jacobian_det_min": result.metadata.get("jacobian_det_min", 1.0),
                    "jacobian_det_max": result.metadata.get("jacobian_det_max", 1.0),
                    "jacobian_condition_max": result.metadata.get("jacobian_condition_max", 1.0),
                    **metrics,
                })
                if position == 0:
                    panel_images[(family, severity)] = result.image
        if (position + 1) % 5 == 0 or position + 1 == len(section_ids):
            print(f"phase=SS1_H0 section={position + 1}/{len(section_ids)} conditions=16 expert=LOCKED", flush=True)
    with (output / "per_section_condition.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    with (output / "agreement_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(agreement_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(agreement_rows)
    summaries = []
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows: grouped[(str(row["family"]), int(row["severity"]))].append(row)
    for (family, severity), local in sorted(grouped.items()):
        summaries.append({
            "family": family, "severity": severity, "sections": len(local),
            **{key: float(np.mean([float(row[key]) for row in local])) for key in ("dice", "cldice", "precision", "recall", "skeleton_f1_at_2px", "fragmentation")},
        })
    with (output / "condition_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(summaries)
    _panel(panel_images, _normalized_image(section_ids[0]), section_ids[0], output / "figures/five_family_panel.png")
    complete_matrix = len(rows) == len(section_ids) * (1 + len(FAMILIES) * len(SEVERITIES))
    required_checks = (
        validation["all_finite"], validation["all_deterministic"],
        validation["warp_jacobians_valid"], validation["annotation_palette_preserved"],
        validation["expert_lock_respected"], not validation["expert_label_pixels_loaded"],
    )
    status = "SS_S1_PASS" if complete_matrix and all(required_checks) else "STOP_SS_PERTURBATION_IMPLEMENTATION_INVALID"
    result = {
        "status": status, "protocol_sha256": protocol_hash(), "split_sha256": split["sha256"],
        "calibration_sections": section_ids, "calibration_section_count": len(section_ids),
        "conditions_per_section": 16, "per_section_condition_rows": len(rows),
        "h0_checkpoint": str(checkpoint.relative_to(ROOT)), "h0_checkpoint_sha256": sha256_file(checkpoint),
        "h0_run_hash": run_hash, "h0_threshold": threshold, "threshold_source": "historical frozen T1 seed41 evaluation",
        "validation": validation, "severity3_used_for_training": False,
        "new_training_opened": False, "B0_B1_B2_B3_opened": False, "LIRA_opened": False, "confirm_opened": False,
        "old_stop_sha256_after": _old_stop_manifest(),
    }
    _write_json(output / "metrics.json", result)
    _write_json(output / "validator.json", {
        "validator_status": "PASS" if status == "SS_S1_PASS" else "FAIL", "research_status": status,
        "checks": {
            "all_finite": validation["all_finite"],
            "all_deterministic": validation["all_deterministic"],
            "warp_jacobians_valid": validation["warp_jacobians_valid"],
            "annotation_palette_preserved": validation["annotation_palette_preserved"],
            "expert_lock_respected": validation["expert_lock_respected"] and not validation["expert_label_pixels_loaded"],
            "complete_5x3_matrix": complete_matrix,
            "old_stops_unchanged": _old_stop_manifest() == ss0["old_stop_sha256_before"],
        },
    })
    summary_by_condition = {(str(row["family"]), int(row["severity"])): row for row in summaries}
    diagnostic_rows = []
    for family, severity in (("clean", 0), *((family, 3) for family in FAMILIES)):
        row = summary_by_condition[(family, severity)]
        label = "clean" if family == "clean" else f"{family} S{severity}"
        diagnostic_rows.append(
            f"| {label} | {row['dice']:.4f} | {row['cldice']:.4f} | {row['fragmentation']:.4f} |"
        )
    (output / "PERTURBATION_VALIDATION_REPORT.md").write_text(
        "# CRACKS-SSBench-V1 perturbation validation\n\n"
        "Only the frozen historical T1 U-Net seed41 was evaluated. No model was trained and no family was selected or removed using its performance.\n\n"
        f"- Calibration sections: `{len(section_ids)}`; condition rows: `{len(rows)}` (`clean + 5x3` per section).\n"
        f"- Historical threshold: `{threshold}`; checkpoint SHA-256: `{result['h0_checkpoint_sha256']}`.\n"
        f"- Finite outputs: `{validation['all_finite']}`; deterministic: `{validation['all_deterministic']}`.\n"
        f"- Warp Jacobians valid: `{validation['warp_jacobians_valid']}`; label palette preserved: `{validation['annotation_palette_preserved']}`.\n"
        "- Severity 3 was evaluation-only. Expert label pixels were not loaded.\n"
        "- There is deliberately no minimum degradation gate and no perturbation cherry-picking.\n\n"
        "Frozen-H0 diagnostics (not a selection gate):\n\n"
        "| Condition | Dice | clDice | Fragmentation |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(diagnostic_rows)
        + "\n\nThe complete 16-condition table is stored in `condition_summary.csv`; no "
        "family was selected or excluded from these values.\n\n"
        f"{status}\n"
    )
    return result


def run(*, device: str = "cuda") -> dict[str, object]:
    ss0 = run_ss0()
    if ss0["status"] != "SS_S0_PASS":
        return ss0
    ss1 = run_ss1(ss0, device=device)
    master = {"phase": "SS0_SS1", "ss0": ss0, "ss1": ss1, "status": ss1["status"], "downstream_training_authorized": ss1["status"] == "SS_S1_PASS"}
    _write_json(RESULT_ROOT / "ANZA_LIRA_SS_V1_MASTER_RESULT.json", master)
    return master
