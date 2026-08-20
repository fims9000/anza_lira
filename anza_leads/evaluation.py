"""Calibration-only threshold freeze and one-shot A1 development evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F

from cracks_experiment.evaluation import evaluate_binary_section, hard_cldice
from cracks_experiment.partial_label_evaluation import _calibration
from cracks_experiment.partial_label_training import NORMALIZATION
from cracks_experiment.partial_labels import CRACKSMultiAnnotatorDataset
from cracks_experiment.validation import tiled_probability
from trace_extraction.skeleton import skeletonize_mask

from .model import LEADS_VARIANTS, build_leads_model
from .orientation import crowd_orientation_targets
from .protocol import A0_ROOT, A1_ROOT, PROTOCOL, active_manifests, canonical_hash, protocol_hash, write_json
from .training import load_checkpoint, run_hash


METRICS = (
    "dice", "precision", "recall", "auprc", "cldice", "skeleton_f1_at_2px", "fragmentation",
    "predicted_foreground_fraction", "explicit_predicted_foreground_fraction",
    "explicit_target_foreground_fraction", "unknown_white_foreground_fraction", "brier", "ece",
    "green_recall", "green_auprc", "green_skeleton_recall_at_2px",
)


def _dataset(section_ids: list[int], *, seed: int = 41) -> CRACKSMultiAnnotatorDataset:
    root = Path(__file__).resolve().parents[1]
    return CRACKSMultiAnnotatorDataset(
        image_root=root / "data" / "cracks" / "images",
        annotation_root=root / "data" / "cracks" / "annotations",
        section_ids=section_ids, annotators=PROTOCOL["data"]["heldout_annotators"],
        mean=NORMALIZATION["mean"], std=NORMALIZATION["std"], crop_size=None,
        annotators_per_section=None, seed=seed,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _safe_auprc(target: np.ndarray, probability: np.ndarray, threshold: float) -> float:
    truth = np.asarray(target, dtype=bool)
    score = np.asarray(probability, dtype=np.float64)
    if truth.any() and (~truth).any():
        return float(average_precision_score(truth, score))
    if truth.any():
        return 1.0
    return float(not np.any(score >= threshold))


def _green_metrics(probability: np.ndarray, target: np.ndarray, weight: np.ndarray, threshold: float) -> dict[str, float]:
    explicit = weight > 0
    green = (target >= 0.5) & np.isclose(weight, 0.5)
    prediction = probability >= threshold
    recall = float(np.count_nonzero(prediction & green) / np.count_nonzero(green)) if green.any() else 1.0
    auprc = _safe_auprc(green[explicit], probability[explicit], threshold)
    skeleton = skeletonize_mask(green)
    support = prediction.copy()
    for _ in range(2):
        support = binary_dilation(support, structure=np.ones((3, 3), dtype=bool))
    skeleton_recall = float(np.count_nonzero(skeleton & support) / np.count_nonzero(skeleton)) if skeleton.any() else 1.0
    return {"green_recall": recall, "green_auprc": auprc, "green_skeleton_recall_at_2px": skeleton_recall}


def annotation_metrics(probability: np.ndarray, target: np.ndarray, weight: np.ndarray, threshold: float) -> dict[str, float]:
    valid = weight > 0
    truth = target >= 0.5
    prediction = probability >= threshold
    metrics = evaluate_binary_section(probability, truth, valid, threshold)
    brier, ece = _calibration(probability, truth, weight)
    unknown = ~valid
    return {
        "dice": float(metrics["dice"]), "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]), "auprc": _safe_auprc(truth[valid], probability[valid], threshold),
        "cldice": float(metrics["cldice"]), "skeleton_f1_at_2px": float(metrics["skeleton_f1_at_2px"]),
        "fragmentation": float(metrics["fragmentation"]),
        "predicted_foreground_fraction": float(prediction.mean()),
        "explicit_predicted_foreground_fraction": float(prediction[valid].mean()),
        "explicit_target_foreground_fraction": float(truth[valid].mean()),
        "unknown_white_foreground_fraction": float(prediction[unknown].mean()) if unknown.any() else 0.0,
        "brier": float(brier), "ece": float(ece),
        **_green_metrics(probability, truth, weight, threshold),
    }


def _load_model(variant: str, device: str) -> tuple[torch.nn.Module, Path, dict[str, Any]]:
    status = json.loads((A1_ROOT / "runs" / variant / "status.json").read_text())
    if status.get("status") != "COMPLETE" or status.get("expert_data_accessed") is not False:
        raise ValueError(f"LEADS training is not complete and expert-blind: {variant}")
    checkpoint = Path(status["checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != status["checkpoint_sha256"]:
        raise ValueError(f"LEADS checkpoint hash mismatch: {variant}")
    model = build_leads_model(variant).to(torch.device(device))
    load_checkpoint(checkpoint, variant, model)
    model.eval()
    return model, checkpoint, status


def predict_split(variant: str, split_name: str, section_ids: list[int], *, device: str) -> dict[int, np.ndarray]:
    cache = A1_ROOT / "probabilities" / f"{variant}_{split_name}.npz"
    model, checkpoint, _status = _load_model(variant, device)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest_path = cache.with_suffix(".json")
    expected_manifest = {
        "checkpoint_sha256": checkpoint_sha, "section_ids": section_ids, "split": split_name,
        "probability_dtype": "float32", "expert_data_accessed": False,
    }
    if cache.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest == expected_manifest:
            loaded = np.load(cache)
            return {int(key): loaded[key].astype(np.float32) for key in loaded.files}
        legacy = dict(expected_manifest)
        legacy.pop("probability_dtype")
        if manifest != legacy:
            raise ValueError(f"LEADS probability cache drift: {cache}")
    dataset = _dataset(section_ids)
    probabilities: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for index in range(len(dataset)):
            batch = dataset[index]
            section_id = int(batch["section_id"])
            probabilities[section_id] = tiled_probability(model, batch["image"]).numpy()[:255, :701].astype(np.float32)
            if (index + 1) % 16 == 0 or index + 1 == len(dataset):
                print(f"phase=ANZA-LEADS-{split_name.upper()} variant={variant} section={index + 1}/{len(dataset)} expert=LOCKED", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{str(key): value.astype(np.float32) for key, value in probabilities.items()})
    write_json(manifest_path, expected_manifest)
    return probabilities


def evaluate_thresholds(variant: str, section_ids: list[int], probabilities: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    dataset = _dataset(section_ids)
    rows = []
    for threshold in PROTOCOL["calibration"]["threshold_grid"]:
        section_rows = []
        for index in range(len(dataset)):
            batch = dataset[index]
            probability = probabilities[int(batch["section_id"])]
            annotations = []
            for target, weight in zip(batch["targets"][:, 0, :255, :701], batch["weights"][:, 0, :255, :701], strict=True):
                truth = target.numpy() >= 0.5
                valid = weight.numpy() > 0
                prediction = (probability >= float(threshold)) & valid
                local_truth = truth & valid
                tp = int(np.count_nonzero(prediction & local_truth))
                fp = int(np.count_nonzero(prediction & ~local_truth & valid))
                fn = int(np.count_nonzero(~prediction & local_truth))
                annotations.append({
                    "precision": float(tp / (tp + fp)) if tp + fp else (1.0 if not local_truth.any() else 0.0),
                    "dice": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 1.0,
                    "cldice": hard_cldice(prediction, local_truth),
                })
            section_rows.append({key: float(np.mean([row[key] for row in annotations])) for key in ("precision", "dice", "cldice")})
        rows.append({
            "variant": variant, "threshold": float(threshold), "section_count": len(section_rows),
            **{key: float(np.mean([row[key] for row in section_rows])) for key in ("precision", "dice", "cldice")},
        })
    return rows


def select_threshold(curve: list[dict[str, Any]]) -> dict[str, Any]:
    target = float(PROTOCOL["calibration"]["precision_target"])
    feasible = [row for row in curve if float(row["precision"]) >= target]
    if feasible:
        selected = max(feasible, key=lambda row: (float(row["cldice"]), float(row["dice"]), -float(row["threshold"])))
        reason = "max_cldice_subject_to_precision"
    else:
        selected = max(curve, key=lambda row: (float(row["precision"]), float(row["cldice"]), -float(row["threshold"])))
        reason = "precision_constraint_infeasible_highest_precision"
    return {
        "selected_threshold": float(selected["threshold"]), "precision_target": target,
        "constraint_feasible": bool(feasible), "selection_reason": reason,
        "selected_calibration_metrics": selected,
    }


def calibrate_all(*, device: str) -> dict[str, Any]:
    split, _subsets = active_manifests()
    if (A1_ROOT / "development_open_receipt.json").exists():
        raise PermissionError("cannot recalibrate after development was opened")
    curves = {}
    selections = {}
    for variant in LEADS_VARIANTS:
        probabilities = predict_split(variant, "calibration", split["calibration"], device=device)
        curves[variant] = evaluate_thresholds(variant, split["calibration"], probabilities)
        selections[variant] = select_threshold(curves[variant])
        write_json(A1_ROOT / "calibration" / f"{variant}.json", {"curve": curves[variant], **selections[variant]})
    receipt = {
        "status": "FROZEN", "protocol_sha256": protocol_hash(), "split_sha256": split["sha256"],
        "precision_target": PROTOCOL["calibration"]["precision_target"], "selections": selections,
        "development_data_accessed": False, "expert_data_accessed": False,
    }
    receipt["freeze_sha256"] = canonical_hash(receipt)
    write_json(A1_ROOT / "threshold_freeze.json", receipt)
    return receipt


def _verify_threshold_freeze() -> dict[str, Any]:
    receipt = json.loads((A1_ROOT / "threshold_freeze.json").read_text())
    freeze_sha = receipt.pop("freeze_sha256", None)
    if freeze_sha != canonical_hash(receipt) or receipt.get("status") != "FROZEN":
        raise PermissionError("invalid LEADS threshold freeze")
    if set(receipt["selections"]) != set(LEADS_VARIANTS) or receipt.get("development_data_accessed") is not False:
        raise PermissionError("incomplete pre-development threshold freeze")
    return {**receipt, "freeze_sha256": freeze_sha}


def _section_rows(
    variant: str, section_ids: list[int], probabilities: dict[int, np.ndarray], threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = _dataset(section_ids)
    subset_manifest = json.loads((A0_ROOT / "label_subset_manifest.json").read_text())
    quintile_boundaries = np.asarray(subset_manifest["quintile_boundaries"], dtype=np.float64)
    annotation_rows = []
    section_rows = []
    for index in range(len(dataset)):
        batch = dataset[index]
        section_id = int(batch["section_id"])
        probability = probabilities[section_id]
        local = []
        for annotator, target, weight in zip(
            batch["annotators"], batch["targets"][:, 0, :255, :701], batch["weights"][:, 0, :255, :701], strict=True,
        ):
            row = {"variant": variant, "section_id": section_id, "annotator": annotator, "threshold": threshold,
                   **annotation_metrics(probability, target.numpy() >= 0.5, weight.numpy(), threshold)}
            annotation_rows.append(row)
            local.append(row)
        explicit_count = 0
        positive_count = 0
        for target, weight in zip(batch["targets"][:, 0, :255, :701], batch["weights"][:, 0, :255, :701], strict=True):
            valid = weight.numpy() > 0
            explicit_count += int(np.count_nonzero(valid))
            positive_count += int(np.count_nonzero((target.numpy() >= 0.5) & valid))
        positive_fraction = float(positive_count / explicit_count) if explicit_count else 0.0
        section_rows.append({
            "variant": variant, "section_id": section_id, "annotator_count": len(local), "threshold": threshold,
            "explicit_positive_fraction": positive_fraction,
            "positive_quintile": int(np.searchsorted(quintile_boundaries, positive_fraction, side="right")),
            **{key: float(np.mean([row[key] for row in local])) for key in METRICS},
        })
    return annotation_rows, section_rows


def operator_diagnostics(variant: str, *, device: str) -> dict[str, Any]:
    model, _checkpoint, status = _load_model(variant, device)
    split, _ = active_manifests()
    dataset = _dataset(split["calibration"][:8])
    entropy = []
    activation = []
    agreement = []
    residual_ratios = []
    hooks = []
    for bank in (model.bank_quarter, model.bank_half):
        if bank is not None:
            def hook(_module, inputs, output, store=residual_ratios):
                store.append(float((output[0] - inputs[0]).norm().detach().cpu() / inputs[0].norm().detach().cpu().clamp_min(1e-8)))
            hooks.append(bank.register_forward_hook(hook))
    with torch.inference_mode():
        for index in range(len(dataset)):
            batch = dataset[index]
            image = batch["image"][:, :256, :256].unsqueeze(0).to(device)
            targets = batch["targets"][:, :, :256, :256].to(device)
            weights = batch["weights"][:, :, :256, :256].to(device)
            bank_target, confidence = crowd_orientation_targets(targets, weights)
            output = model(image, return_aux=True)
            for logits in output["orientation_logits"]:
                evidence = torch.sigmoid(logits)
                entropy.append(float((-(evidence * torch.log(evidence.clamp_min(1e-7)) + (1-evidence) * torch.log((1-evidence).clamp_min(1e-7)))).mean().cpu()))
                activation.append(float((evidence > 0.5).float().mean().cpu()))
                local_target = F.interpolate(bank_target, size=logits.shape[-2:], mode="bilinear", align_corners=False)
                local_valid = F.interpolate(confidence, size=logits.shape[-2:], mode="nearest") > 0
                truth_index = local_target.argmax(dim=1)
                pred_index = logits.argmax(dim=1).expand(targets.shape[0], -1, -1)
                valid = local_valid[:, 0]
                if valid.any():
                    delta = (truth_index - pred_index).abs()
                    delta = torch.minimum(delta, 8 - delta)
                    agreement.append(float((delta[valid] <= 1).float().mean().cpu()))
    for handle in hooks:
        handle.remove()
    result: dict[str, Any] = {
        "variant": variant, "gamma": {}, "evidence_entropy": float(np.mean(entropy)),
        "evidence_activation_fraction": float(np.mean(activation)),
        "orientation_within_one_bin": float(np.mean(agreement)) if agreement else 0.0,
        "residual_correction_norm_ratio": float(np.mean(residual_ratios)) if residual_ratios else 0.0,
        "gradient_norm_last_step": float(status["history"][-1]["gradient_norm_last_step"]),
    }
    for stage, bank in (("quarter", model.bank_quarter), ("half", model.bank_half)):
        if bank is None:
            continue
        result["gamma"][stage] = float(bank.gamma.detach().cpu())
        sigma_u, sigma_s = (value.detach().cpu().numpy() for value in bank.scales())
        ell = np.sqrt(sigma_u * sigma_s)
        lam = 0.5 * np.log(sigma_u / sigma_s)
        result[f"{stage}_geometry"] = {
            "sigma_u": sigma_u.tolist(), "sigma_s": sigma_s.tolist(),
            "ell": ell.tolist(), "lambda": lam.tolist(),
        }
    return result


def _bootstrap_comparison(indexed: dict[tuple[str, int], dict[str, Any]], sections: list[int]) -> dict[str, Any]:
    generator = np.random.default_rng(4101)
    metrics = ("dice", "cldice", "fragmentation", "unknown_white_foreground_fraction")
    output = {}
    for metric in metrics:
        deltas = np.asarray([
            float(indexed[("L3_anza_hs", section)][metric]) - float(indexed[("L2_generic_aniso", section)][metric])
            for section in sections
        ])
        samples = np.empty(10_000, dtype=np.float64)
        for index in range(len(samples)):
            samples[index] = float(np.mean(generator.choice(deltas, size=len(deltas), replace=True)))
        output[metric] = {"mean_delta": float(deltas.mean()), "ci95_low": float(np.quantile(samples, 0.025)), "ci95_high": float(np.quantile(samples, 0.975))}
    return output


def development_once(*, device: str) -> dict[str, Any]:
    freeze = _verify_threshold_freeze()
    split, _ = active_manifests()
    receipt_path = A1_ROOT / "development_open_receipt.json"
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text())
        if existing.get("threshold_freeze_sha256") != freeze["freeze_sha256"]:
            raise ValueError("development was opened under a different threshold freeze")
    else:
        write_json(receipt_path, {
            "status": "OPENED_ONCE", "threshold_freeze_sha256": freeze["freeze_sha256"],
            "development_sections_sha256": canonical_hash(split["development"]),
            "expert_data_accessed": False,
        })
    all_annotations = []
    all_sections = []
    summaries = {}
    diagnostics = {}
    for variant in LEADS_VARIANTS:
        probability = predict_split(variant, "development", split["development"], device=device)
        threshold = float(freeze["selections"][variant]["selected_threshold"])
        annotation_rows, section_rows = _section_rows(variant, split["development"], probability, threshold)
        all_annotations.extend(annotation_rows)
        all_sections.extend(section_rows)
        summaries[variant] = {key: float(np.mean([row[key] for row in section_rows])) for key in METRICS}
        diagnostics[variant] = operator_diagnostics(variant, device=device)
    indexed = {(row["variant"], int(row["section_id"])): row for row in all_sections}
    l2 = summaries["L2_generic_aniso"]
    l3 = summaries["L3_anza_hs"]
    dice_delta = l3["dice"] - l2["dice"]
    cldice_delta = l3["cldice"] - l2["cldice"]
    fragmentation_ratio = l3["fragmentation"] / l2["fragmentation"] if l2["fragmentation"] > 0 else (0.0 if l3["fragmentation"] == 0 else math.inf)
    white_ratio = l3["unknown_white_foreground_fraction"] / l2["unknown_white_foreground_fraction"] if l2["unknown_white_foreground_fraction"] > 0 else (0.0 if l3["unknown_white_foreground_fraction"] == 0 else math.inf)
    checks = {
        "dice_noninferiority": dice_delta >= float(PROTOCOL["gate"]["dice_delta_min"]),
        "cldice_gain": cldice_delta >= float(PROTOCOL["gate"]["cldice_delta_min"]),
        "fragmentation_gain": fragmentation_ratio <= float(PROTOCOL["gate"]["fragmentation_ratio_max"]),
        "unknown_white_safety": white_ratio <= float(PROTOCOL["gate"]["unknown_white_foreground_ratio_max"]),
    }
    gate_pass = bool(checks["dice_noninferiority"] and checks["unknown_white_safety"] and (checks["cldice_gain"] or checks["fragmentation_gain"]))
    scale_diagnostic = {"opened": False, "authorized": False}
    if not gate_pass and checks["dice_noninferiority"] and checks["unknown_white_safety"]:
        near_equal = (
            cldice_delta >= float(PROTOCOL["gate"]["near_equal_repair_diagnostic"]["cldice_delta_min"])
            and fragmentation_ratio <= float(PROTOCOL["gate"]["near_equal_repair_diagnostic"]["fragmentation_ratio_max"])
        )
        if near_equal:
            generic = diagnostics["L2_generic_aniso"]
            stage_ells = [np.median(generic[f"{stage}_geometry"]["ell"]) for stage in ("quarter", "half")]
            lambdas = np.concatenate([generic[f"{stage}_geometry"]["lambda"] for stage in ("quarter", "half")])
            scale_difference = abs(float(stage_ells[0]) - float(stage_ells[1])) / float(PROTOCOL["operator"]["base_scale"])
            activation = float(diagnostics["L3_anza_hs"]["evidence_activation_fraction"])
            repair = PROTOCOL["gate"]["scale_repair"]
            conditions = {
                "median_lambda_retained": float(repair["median_lambda_min"]) <= float(np.median(lambdas)) <= float(repair["median_lambda_max"]),
                "stage_scale_difference_ge_15pct": scale_difference >= float(repair["systematic_scale_difference_min"]),
                "anza_evidence_not_collapsed": float(repair["evidence_activation_min"]) <= activation <= float(repair["evidence_activation_max"]),
            }
            scale_diagnostic = {
                "opened": True, "near_equal_precondition": True, "median_lambda": float(np.median(lambdas)),
                "stage_median_ell": stage_ells, "relative_stage_scale_difference": scale_difference,
                "anza_evidence_activation_fraction": activation, "conditions": conditions,
                "authorized": bool(all(conditions.values())),
            }
    status = "ANZA_LOW_LABEL_MECHANISM_PASS" if gate_pass else (
        "ANZA_LOW_LABEL_SCALE_REPAIR_AUTHORIZED" if scale_diagnostic["authorized"] else "STOP_ANZA_LABEL_EFFICIENCY_NO_SIGNAL"
    )
    _write_csv(A1_ROOT / "per_annotator.csv", all_annotations)
    _write_csv(A1_ROOT / "per_section.csv", all_sections)
    stratum_rows = []
    for variant in LEADS_VARIANTS:
        rows = [row for row in all_sections if row["variant"] == variant]
        for stratum in sorted({int(row["positive_quintile"]) for row in rows}):
            selected = [row for row in rows if int(row["positive_quintile"]) == stratum]
            stratum_rows.append({"variant": variant, "positive_quintile": stratum, "section_count": len(selected), **{key: float(np.mean([row[key] for row in selected])) for key in METRICS}})
    _write_csv(A1_ROOT / "per_stratum.csv", stratum_rows)
    write_json(A1_ROOT / "operator_diagnostics.json", diagnostics)
    bootstrap = _bootstrap_comparison(indexed, split["development"])
    write_json(A1_ROOT / "bootstrap.json", bootstrap)
    result = {
        "status": status, "gate_pass": gate_pass, "protocol_sha256": protocol_hash(),
        "threshold_freeze_sha256": freeze["freeze_sha256"], "seed": 41, "label_fraction": 0.10,
        "summaries": summaries,
        "comparison": {
            "dice_delta_L3_minus_L2": dice_delta, "cldice_delta_L3_minus_L2": cldice_delta,
            "fragmentation_ratio_L3_over_L2": fragmentation_ratio,
            "unknown_white_foreground_ratio_L3_over_L2": white_ratio, "checks": checks,
        },
        "scale_mismatch_diagnostic": scale_diagnostic, "bootstrap": bootstrap,
        "development_opened_once": True, "expert_data_accessed": False,
        "seeds_42_43_opened": False, "anza_ms_implemented": False, "ssl_opened": False,
        "domain_shift_opened": False, "oof_opened": False, "lira_opened": False,
    }
    write_json(A1_ROOT / "metrics.json", result)
    return result
