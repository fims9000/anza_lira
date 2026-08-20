"""Uncapped calibration and one-shot development evaluation for LEADS RC1."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from sklearn.metrics import average_precision_score
import torch

from cracks_experiment.evaluation import hard_cldice
from cracks_experiment.partial_label_training import NORMALIZATION
from cracks_experiment.partial_labels import CRACKSMultiAnnotatorDataset
from cracks_experiment.validation import tiled_probability

from .evaluation import METRICS, annotation_metrics
from .model import build_leads_model
from .protocol import PROTOCOL as PARENT_PROTOCOL, canonical_hash, write_json
from .rc1_protocol import ROOT, VARIANTS, load_frozen, verify_parent_immutable
from .rc1_training import load_checkpoint


WHITE_METRICS = (
    "white_foreground_fraction", "unsupported_white_foreground_fraction",
    "white_pred_distance_0_2_fraction", "white_pred_distance_2_5_fraction",
    "white_pred_distance_gt5_fraction", "white_connected_foreground_fraction",
    "white_isolated_foreground_fraction",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _dataset(section_ids: list[int]) -> CRACKSMultiAnnotatorDataset:
    project = Path(__file__).resolve().parents[1]
    return CRACKSMultiAnnotatorDataset(
        image_root=project / "data" / "cracks" / "images",
        annotation_root=project / "data" / "cracks" / "annotations",
        section_ids=section_ids, annotators=PARENT_PROTOCOL["data"]["heldout_annotators"],
        mean=NORMALIZATION["mean"], std=NORMALIZATION["std"], crop_size=None,
        annotators_per_section=None, seed=41,
    )


def _load_model(variant: str, device: str) -> tuple[torch.nn.Module, Path]:
    status = json.loads((ROOT / "training" / variant / "status.json").read_text())
    if status.get("status") != "COMPLETE" or status.get("expert_data_accessed") is not False:
        raise ValueError(f"RC1 training is incomplete: {variant}")
    checkpoint = Path(status["checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != status["checkpoint_sha256"]:
        raise ValueError(f"RC1 checkpoint hash mismatch: {variant}")
    model = build_leads_model(variant).to(device); load_checkpoint(checkpoint, variant, model); model.eval()
    return model, checkpoint


def predict_split(variant: str, split_name: str, section_ids: list[int], *, device: str) -> dict[int, np.ndarray]:
    cache = ROOT / "probabilities" / f"{variant}_{split_name}.npz"
    manifest_path = cache.with_suffix(".json")
    model, checkpoint = _load_model(variant, device)
    expected = {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "section_ids": section_ids, "split": split_name, "probability_dtype": "float32",
        "expert_data_accessed": False,
    }
    if cache.exists() and manifest_path.exists() and json.loads(manifest_path.read_text()) == expected:
        loaded = np.load(cache)
        return {int(key): loaded[key].astype(np.float32) for key in loaded.files}
    data = _dataset(section_ids); probabilities = {}
    with torch.inference_mode():
        for index in range(len(data)):
            batch = data[index]; section = int(batch["section_id"])
            probabilities[section] = tiled_probability(model, batch["image"]).numpy()[:255, :701].astype(np.float32)
            if (index + 1) % 8 == 0 or index + 1 == len(data):
                print(f"phase=LEADS-RC1-{split_name.upper()} variant={variant} section={index + 1}/{len(data)} expert=LOCKED", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{str(key): value for key, value in probabilities.items()})
    write_json(manifest_path, expected)
    return probabilities


def threshold_candidates(scores: np.ndarray, *, count: int = 4097, explicit: list[float] | None = None) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("no finite calibration scores")
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, min(int(count), values.size)), method="nearest")
    fixed = np.asarray(explicit or [0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999])
    return np.unique(np.clip(np.concatenate(([0.0, 1.0], quantiles, fixed)), 0.0, 1.0))


def _observations(section_ids: list[int], probabilities: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    data = _dataset(section_ids); rows = []
    for index in range(len(data)):
        batch = data[index]; section = int(batch["section_id"]); probability = probabilities[section]
        for annotator, target, weight in zip(batch["annotators"], batch["targets"][:, 0, :255, :701], batch["weights"][:, 0, :255, :701], strict=True):
            truth = target.numpy() >= 0.5; valid = weight.numpy() > 0
            rows.append({"section_id": section, "annotator": annotator, "probability": probability, "truth": truth, "valid": valid, "weight": weight.numpy()})
    return rows


def _pixel_frontier(observations: list[dict[str, Any]], thresholds: np.ndarray) -> list[dict[str, Any]]:
    by_section: dict[int, list[dict[str, np.ndarray]]] = {}
    for row in observations:
        score = row["probability"][row["valid"]]
        truth = row["truth"][row["valid"]]
        positives = np.sort(score[truth]); negatives = np.sort(score[~truth])
        tp = positives.size - np.searchsorted(positives, thresholds, side="left")
        fp = negatives.size - np.searchsorted(negatives, thresholds, side="left")
        fn = positives.size - tp
        precision = np.divide(
            tp, tp + fp,
            out=np.full(tp.shape, 1.0 if positives.size == 0 else 0.0, dtype=float),
            where=(tp + fp) > 0,
        )
        recall = np.divide(tp, positives.size, out=np.ones_like(tp, dtype=float), where=positives.size > 0)
        dice = np.divide(2 * tp, 2 * tp + fp + fn, out=np.ones_like(tp, dtype=float), where=(2 * tp + fp + fn) > 0)
        by_section.setdefault(int(row["section_id"]), []).append({"precision": precision, "recall": recall, "dice": dice})
    output = []
    for index, threshold in enumerate(thresholds):
        section_metrics = []
        for local in by_section.values():
            section_metrics.append({key: float(np.mean([item[key][index] for item in local])) for key in ("precision", "recall", "dice")})
        output.append({"threshold": float(threshold), "section_count": len(section_metrics), **{
            key: float(np.mean([item[key] for item in section_metrics])) for key in ("precision", "recall", "dice")
        }})
    return output


def _topology_thresholds(frontier: list[dict[str, Any]], target: float, maximum: int = 65) -> list[float]:
    feasible = [row for row in frontier if row["precision"] >= target and row["recall"] > 0]
    source = feasible if feasible else frontier
    if len(source) <= maximum:
        return [float(row["threshold"]) for row in source]
    indices = np.unique(np.rint(np.linspace(0, len(source) - 1, maximum)).astype(int))
    return [float(source[index]["threshold"]) for index in indices]


def _cldice_frontier(observations: list[dict[str, Any]], thresholds: list[float]) -> dict[float, float]:
    by_section: dict[int, list[dict[str, Any]]] = {}
    for row in observations:
        by_section.setdefault(int(row["section_id"]), []).append(row)
    output = {}
    for threshold in thresholds:
        section_values = []
        for local in by_section.values():
            section_values.append(float(np.mean([
                hard_cldice((row["probability"] >= threshold) & row["valid"], row["truth"] & row["valid"])
                for row in local
            ])))
        output[float(threshold)] = float(np.mean(section_values))
    return output


def select_threshold(curve: list[dict[str, Any]], *, precision_target: float = 0.90) -> dict[str, Any]:
    feasible = [row for row in curve if row.get("cldice") is not None and row["precision"] >= precision_target and row["recall"] > 0]
    if not feasible:
        return {"constraint_feasible": False, "precision_target": precision_target, "selected_threshold": None,
                "selection_reason": "precision_0.90_with_nonzero_recall_infeasible"}
    selected = max(feasible, key=lambda row: (row["cldice"], row["dice"], row["recall"], row["threshold"]))
    return {"constraint_feasible": True, "precision_target": precision_target, "selected_threshold": float(selected["threshold"]),
            "selection_reason": "max_cldice_subject_precision_and_nonzero_recall", "selected_calibration_metrics": selected}


def calibrate_all(*, device: str) -> dict[str, Any]:
    protocol, split = load_frozen()
    if (ROOT / "development_open_receipt.json").exists():
        raise PermissionError("RC1 recalibration forbidden after development open")
    selections = {}
    for variant in VARIANTS:
        probabilities = predict_split(variant, "calibration", split["calibration"], device=device)
        observations = _observations(split["calibration"], probabilities)
        all_scores = np.concatenate([row["probability"][row["valid"]] for row in observations])
        thresholds = threshold_candidates(all_scores, count=int(protocol["calibration"]["quantile_candidates"]), explicit=protocol["calibration"]["explicit_thresholds"])
        frontier = _pixel_frontier(observations, thresholds)
        topology = _cldice_frontier(observations, _topology_thresholds(frontier, float(protocol["calibration"]["precision_target"])))
        for row in frontier:
            row["cldice"] = topology.get(float(row["threshold"]))
            row["topology_evaluated"] = row["cldice"] is not None
            row["variant"] = variant
        selection = select_threshold(frontier, precision_target=float(protocol["calibration"]["precision_target"]))
        selections[variant] = selection
        _write_csv(ROOT / "calibration" / f"{variant}_frontier.csv", frontier)
        write_json(ROOT / "calibration" / f"{variant}.json", {"selection": selection, "candidate_count": len(frontier), "topology_candidate_count": len(topology)})
    receipt = {
        "status": "FROZEN" if selections["L2_generic_aniso"]["constraint_feasible"] and selections["L3_anza_hs"]["constraint_feasible"] else "HIGH_PRECISION_INFEASIBLE",
        "protocol_sha256": canonical_hash(protocol), "split_sha256": split["sha256"],
        "precision_target": protocol["calibration"]["precision_target"], "selections": selections,
        "development_data_accessed": False, "expert_data_accessed": False,
    }
    receipt["freeze_sha256"] = canonical_hash(receipt)
    write_json(ROOT / "threshold_freeze.json", receipt)
    if receipt["status"] == "HIGH_PRECISION_INFEASIBLE":
        write_json(ROOT / "metrics.json", {
            "status": "STOP_ANZA_RC1_HIGH_PRECISION_INFEASIBLE", "gate_pass": False,
            "threshold_freeze_sha256": receipt["freeze_sha256"], "development_opened": False,
            "expert_data_accessed": False, **{f"{key}_opened": False for key in ("seeds_42_43", "anza_ms", "ssl", "domain_shift", "lira")},
        })
    return receipt


def _verify_freeze() -> dict[str, Any]:
    receipt = json.loads((ROOT / "threshold_freeze.json").read_text()); checksum = receipt.pop("freeze_sha256", None)
    if checksum != canonical_hash(receipt) or receipt.get("development_data_accessed") is not False:
        raise PermissionError("invalid RC1 threshold freeze")
    return {**receipt, "freeze_sha256": checksum}


def unsupported_white_metrics(probability: np.ndarray, target: np.ndarray, weight: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = np.asarray(probability) >= float(threshold); white = np.asarray(weight) <= 0
    positive = (np.asarray(target) >= 0.5) & ~white
    if positive.any():
        distance = distance_transform_edt(~positive)
        support = binary_dilation(positive, structure=np.ones((3, 3), dtype=bool), iterations=5)
    else:
        distance = np.full(white.shape, np.inf); support = np.zeros_like(white)
    components, _ = label(prediction, structure=np.ones((3, 3), dtype=bool))
    supported_ids = np.unique(components[support]); supported_ids = supported_ids[supported_ids > 0]
    connected = np.isin(components, supported_ids) & prediction
    isolated = prediction & ~connected
    denominator = max(1, int(np.count_nonzero(white)))
    selected = prediction & white
    return {
        "white_foreground_fraction": float(np.count_nonzero(selected) / denominator),
        "unsupported_white_foreground_fraction": float(np.count_nonzero(selected & (distance > 5) & isolated) / denominator),
        "white_pred_distance_0_2_fraction": float(np.count_nonzero(selected & (distance <= 2)) / denominator),
        "white_pred_distance_2_5_fraction": float(np.count_nonzero(selected & (distance > 2) & (distance <= 5)) / denominator),
        "white_pred_distance_gt5_fraction": float(np.count_nonzero(selected & (distance > 5)) / denominator),
        "white_connected_foreground_fraction": float(np.count_nonzero(selected & connected) / denominator),
        "white_isolated_foreground_fraction": float(np.count_nonzero(selected & isolated) / denominator),
    }


def _section_rows(variant: str, observations: list[dict[str, Any]], threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotations = []
    for row in observations:
        metrics = annotation_metrics(row["probability"], row["truth"], row["weight"], threshold)
        white = unsupported_white_metrics(row["probability"], row["truth"], row["weight"], threshold)
        annotations.append({"variant": variant, "section_id": row["section_id"], "annotator": row["annotator"], "threshold": threshold, **metrics, **white})
    sections = []
    for section in sorted({int(row["section_id"]) for row in annotations}):
        local = [row for row in annotations if int(row["section_id"]) == section]
        sections.append({"variant": variant, "section_id": section, "annotator_count": len(local), "threshold": threshold,
                         **{key: float(np.mean([row[key] for row in local])) for key in METRICS + WHITE_METRICS}})
    return annotations, sections


def _bootstrap(indexed: dict[tuple[str, int], dict[str, Any]], sections: list[int], *, resamples: int = 10_000) -> dict[str, Any]:
    generator = np.random.default_rng(4119); output = {}
    for metric in ("dice", "cldice", "auprc", "unsupported_white_foreground_fraction"):
        delta = np.asarray([indexed[("L3_anza_hs", section)][metric] - indexed[("L2_generic_aniso", section)][metric] for section in sections], dtype=float)
        samples = np.asarray([np.mean(generator.choice(delta, size=len(delta), replace=True)) for _ in range(resamples)])
        output[metric] = {"mean_delta": float(delta.mean()), "ci95_low": float(np.quantile(samples, .025)), "ci95_high": float(np.quantile(samples, .975))}
    return output


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else (0.0 if numerator == 0 else math.inf)


def _frontier_diagnostic(variant: str, observations: list[dict[str, Any]], calibration_path: Path) -> list[dict[str, Any]]:
    with calibration_path.open(newline="") as handle:
        calibration = [row for row in csv.DictReader(handle) if row["topology_evaluated"] == "True"]
    thresholds = [float(row["threshold"]) for row in calibration]
    pixel = _pixel_frontier(observations, np.asarray(thresholds)); topology = _cldice_frontier(observations, thresholds)
    return [{"variant": variant, **row, "cldice": topology[row["threshold"]]} for row in pixel]


def interpolate_frontier(curve: list[dict[str, Any]], precision_points: list[float]) -> list[dict[str, float]]:
    pairs: dict[float, float] = {}
    for row in curve:
        p = float(row["precision"]); pairs[p] = max(float(row["cldice"]), pairs.get(p, -math.inf))
    xs = np.asarray(sorted(pairs), dtype=float); ys = np.asarray([pairs[value] for value in xs], dtype=float)
    return [{"precision": float(point), "cldice": float(np.interp(point, xs, ys)) if xs[0] <= point <= xs[-1] else math.nan} for point in precision_points]


def development_once(*, device: str) -> dict[str, Any]:
    protocol, split = load_frozen(); freeze = _verify_freeze()
    if freeze["status"] != "FROZEN":
        return json.loads((ROOT / "metrics.json").read_text())
    if not verify_parent_immutable():
        raise ValueError("parent A1 artifacts changed after RC1 freeze")
    receipt_path = ROOT / "development_open_receipt.json"
    expected_receipt = {"status": "OPENED_ONCE", "threshold_freeze_sha256": freeze["freeze_sha256"],
                        "development_sections_sha256": canonical_hash(split["development"]), "expert_data_accessed": False}
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != expected_receipt:
        raise ValueError("RC1 development receipt drift")
    write_json(receipt_path, expected_receipt)
    all_annotations = []; all_sections = []; summaries = {}; frontier_rows = []; frontier_summary = {}
    for variant in VARIANTS:
        probabilities = predict_split(variant, "development", split["development"], device=device)
        observations = _observations(split["development"], probabilities)
        threshold = float(freeze["selections"][variant]["selected_threshold"])
        annotations, sections = _section_rows(variant, observations, threshold)
        all_annotations.extend(annotations); all_sections.extend(sections)
        summaries[variant] = {key: float(np.mean([row[key] for row in sections])) for key in METRICS + WHITE_METRICS}
        curve = _frontier_diagnostic(variant, observations, ROOT / "calibration" / f"{variant}_frontier.csv")
        frontier_rows.extend(curve)
        interpolation = interpolate_frontier(curve, protocol["frontier_precision_points"])
        finite = [row for row in interpolation if math.isfinite(row["cldice"])]
        frontier_summary[variant] = {
            "points": interpolation,
            "partial_area_0.80_0.90": float(np.trapezoid([row["cldice"] for row in finite], [row["precision"] for row in finite])) if len(finite) > 1 else None,
            "diagnostic_only_cannot_rescue_primary_gate": True,
        }
    _write_csv(ROOT / "development_per_annotator.csv", all_annotations); _write_csv(ROOT / "development_per_section.csv", all_sections)
    _write_csv(ROOT / "development_frontiers.csv", frontier_rows); write_json(ROOT / "frontier_summary.json", frontier_summary)
    indexed = {(row["variant"], int(row["section_id"])): row for row in all_sections}
    bootstrap = _bootstrap(indexed, split["development"], resamples=int(protocol["bootstrap_resamples"])); write_json(ROOT / "bootstrap.json", bootstrap)
    l0, l2, l3 = (summaries[key] for key in ("L0_backbone", "L2_generic_aniso", "L3_anza_hs"))
    deltas = {
        "cldice_L3_minus_L2": l3["cldice"] - l2["cldice"], "dice_L3_minus_L2": l3["dice"] - l2["dice"],
        "cldice_L3_minus_L0": l3["cldice"] - l0["cldice"], "dice_L3_minus_L0": l3["dice"] - l0["dice"],
        "auprc_L3_minus_L2": l3["auprc"] - l2["auprc"],
        "unsupported_white_ratio_L3_L2": _safe_ratio(l3["unsupported_white_foreground_fraction"], l2["unsupported_white_foreground_fraction"]),
        "unsupported_white_ratio_L3_L0": _safe_ratio(l3["unsupported_white_foreground_fraction"], l0["unsupported_white_foreground_fraction"]),
    }
    gate = protocol["gate"]
    checks = {
        "development_precision_L2": l2["precision"] >= protocol["development_precision_min"],
        "development_precision_L3": l3["precision"] >= protocol["development_precision_min"],
        "cldice_gain": deltas["cldice_L3_minus_L2"] >= gate["cldice_delta_min"],
        "cldice_ci_positive": bootstrap["cldice"]["ci95_low"] > 0,
        "dice_noninferior_L2": deltas["dice_L3_minus_L2"] >= gate["dice_delta_min"],
        "cldice_noninferior_backbone": deltas["cldice_L3_minus_L0"] >= gate["backbone_cldice_delta_min"],
        "dice_noninferior_backbone": deltas["dice_L3_minus_L0"] >= gate["backbone_dice_delta_min"],
        "auprc_noninferior": deltas["auprc_L3_minus_L2"] >= gate["auprc_delta_min"],
        "unsupported_white_vs_L2": deltas["unsupported_white_ratio_L3_L2"] <= gate["unsupported_white_ratio_max"],
        "unsupported_white_vs_L0": deltas["unsupported_white_ratio_L3_L0"] <= gate["unsupported_white_ratio_max"],
    }
    safety_keys = {"unsupported_white_vs_L2", "unsupported_white_vs_L0"}
    gate_pass = all(checks.values())
    if gate_pass:
        status = "ANZA_RC1_HIGH_PRECISION_LOW_LABEL_PASS"
    elif all(value for key, value in checks.items() if key not in safety_keys) and not all(checks[key] for key in safety_keys):
        status = "ANZA_RC1_STRUCTURAL_SIGNAL_SAFETY_FAIL"
    elif checks["cldice_gain"] and checks["cldice_ci_positive"] and (not checks["cldice_noninferior_backbone"] or not checks["dice_noninferior_backbone"]):
        status = "ANZA_PRIOR_RELATIVE_ONLY_NOT_PRACTICAL"
    else:
        status = "STOP_ANZA_LOW_LABEL_GAIN_WAS_OPERATING_POINT_SPECIFIC"
    result = {
        "status": status, "gate_pass": gate_pass, "summaries": summaries, "deltas": deltas, "checks": checks,
        "bootstrap": bootstrap, "threshold_freeze_sha256": freeze["freeze_sha256"], "development_opened": True,
        "expert_data_accessed": False, **{f"{key}_opened": False for key in ("seeds_42_43", "anza_ms", "ssl", "domain_shift", "lira")},
    }
    write_json(ROOT / "metrics.json", result)
    return result
