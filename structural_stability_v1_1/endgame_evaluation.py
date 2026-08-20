"""Frozen calibration and one-shot development evaluation for V1.1."""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F

from cracks_experiment.evaluation import evaluate_binary_section
from cracks_experiment.partial_labels import map_partial_annotation
from cracks_experiment.validation import tiled_probability
from datasets.cracks import WHITE, load_section_image
from lira_final.protocol import HELDOUT_ANNOTATORS
from structural_stability_v1.perturb import apply_perturbation, transform_rgb_mask
from structural_stability_v1.protocol import FAMILIES, SEVERITIES
from structural_stability_v1_1.amendment import sha256_file
from structural_stability_v1_1.geometry_targets import geometry_target
from structural_stability_v1_1.protocol import PROTOCOL, RESULT_ROOT, ROOT, SEEDS, VARIANTS, canonical_hash, protocol_hash
from structural_stability_v1_1.train_variants import build_fresh_variant


THRESHOLDS = tuple(round(0.05 + 0.025 * index, 3) for index in range(37))
BOOTSTRAP_SEED = 20260819


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _split() -> dict[str, Any]:
    return json.loads((RESULT_ROOT.parent / "anza_lira_ss_v1/s0_audit/split_manifest.json").read_text())


def _normalization() -> dict[str, Any]:
    return json.loads((RESULT_ROOT / "pretrain_freeze/TRAIN_ONLY_NORMALIZATION.json").read_text())


def normalized_section(section_id: int) -> torch.Tensor:
    normalization = _normalization()
    image = load_section_image(ROOT / f"data/cracks/images/section_{section_id:03d}.png")
    tensor = torch.from_numpy(image.transpose(2, 0, 1))
    mean = torch.tensor(normalization["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(normalization["std"], dtype=torch.float32).view(3, 1, 1).clamp_min(1e-6)
    return F.pad((tensor - mean) / std, (0, 3, 0, 1))


def section_masks(section_id: int, *, expert: bool = False) -> tuple[list[str], list[np.ndarray]]:
    names: Iterable[str] = ("expert",) if expert else HELDOUT_ANNOTATORS
    found_names, masks = [], []
    for name in names:
        path = ROOT / f"data/cracks/annotations/{name}/section_{section_id:03d}.png"
        if path.is_file():
            with Image.open(path) as handle:
                masks.append(np.asarray(handle.convert("RGB"), dtype=np.uint8))
            found_names.append(name)
    if not masks:
        raise FileNotFoundError(f"no {'expert' if expert else 'heldout nonexpert'} masks for section {section_id}")
    return found_names, masks


def load_model(variant: str, seed: int, device: torch.device) -> torch.nn.Module:
    validation_path = RESULT_ROOT / f"training/{variant}/s{seed}/RUN_FINAL_VALIDATION.json"
    validation = json.loads(validation_path.read_text())
    checkpoint_path = RESULT_ROOT / f"training/{variant}/s{seed}/checkpoint_final.pt"
    if validation.get("status") != "RUN_FINAL_VALIDATION_PASS" or validation.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError(f"invalid frozen checkpoint {variant} seed {seed}")
    model = build_fresh_variant(variant, seed, RESULT_ROOT / "pretrain_freeze/initialization")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("optimizer_step") != 1980 or payload.get("protocol_sha256") != protocol_hash():
        raise ValueError("final checkpoint provenance mismatch")
    model.load_state_dict(payload["model_state"])
    return model.to(device).eval()


def _all_white(masks: list[np.ndarray]) -> np.ndarray:
    return np.logical_and.reduce([np.all(mask == np.asarray(WHITE, dtype=np.uint8), axis=-1) for mask in masks])


def section_metrics(probability: np.ndarray, masks: list[np.ndarray], threshold: float) -> dict[str, float]:
    per_annotator = []
    auprc = []
    for mask in masks:
        target, weight = map_partial_annotation(mask)
        valid = weight > 0
        per_annotator.append(evaluate_binary_section(probability, target >= 0.5, valid, threshold))
        labels = (target[valid] >= 0.5).astype(np.uint8)
        auprc.append(float(average_precision_score(labels, probability[valid])) if len(np.unique(labels)) > 1 else float(labels[0]))
    keys = ("dice", "cldice", "precision", "recall", "skeleton_f1_at_2px", "fragmentation")
    prediction = probability >= threshold
    white = _all_white(masks)
    return {
        **{key: float(np.mean([float(row[key]) for row in per_annotator])) for key in keys},
        "auprc": float(np.mean(auprc)),
        "predicted_foreground_fraction": float(np.mean(prediction)),
        "white_unknown_foreground": float(np.mean(prediction[white])) if white.any() else 0.0,
        "annotator_count": len(masks),
    }


def _select_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    curve = []
    for threshold in THRESHOLDS:
        selected = [row for row in rows if float(row["threshold"]) == threshold]
        curve.append({
            "threshold": threshold,
            "cldice": float(np.mean([float(row["cldice"]) for row in selected])),
            "precision": float(np.mean([float(row["precision"]) for row in selected])),
            "recall": float(np.mean([float(row["recall"]) for row in selected])),
            "dice": float(np.mean([float(row["dice"]) for row in selected])),
        })
    feasible = [row for row in curve if row["precision"] >= 0.75]
    pool = feasible if feasible else curve
    chosen = max(pool, key=lambda row: (row["cldice"], row["precision"], row["recall"], row["threshold"]))
    return {"selected_threshold": chosen["threshold"], "precision_constraint_feasible": bool(feasible), "selected": chosen, "curve": curve}


@torch.no_grad()
def run_calibration(*, device: str = "cuda") -> dict[str, Any]:
    completion = json.loads((RESULT_ROOT / "TWELVE_RUN_COMPLETION_MANIFEST.json").read_text())
    if completion.get("status") != "SS2_SS3_TRAINING_COMPLETE":
        raise PermissionError("calibration locked until 12-run completion")
    output = RESULT_ROOT / "calibration"
    checkpoint_hashes = {f"{row['variant']}_s{row['seed']}": row["checkpoint_sha256"] for row in completion["records"]}
    authorization = {
        "status": "SS_CALIBRATION_AUTHORIZED",
        "checkpoint_hashes": checkpoint_hashes,
        "protocol_sha256": protocol_hash(),
        "split_sha256": _split()["sha256"],
        "normalization_sha256": _normalization()["sha256"],
        "threshold_grid": list(THRESHOLDS),
        "threshold_grid_sha256": canonical_hash(list(THRESHOLDS)),
        "development_opened": False,
        "confirm_opened": False,
        "expert_opened": False,
    }
    _write_json(RESULT_ROOT / "SS_CALIBRATION_AUTHORIZATION.json", authorization)
    section_ids = list(_split()["splits"]["SS_CALIBRATION"])
    torch_device = torch.device(device)
    per_seed_rows, per_seed_thresholds, ensemble_thresholds = [], [], []
    probability_root = output / "probabilities"
    for variant in VARIANTS:
        seed_probabilities: dict[int, dict[int, np.ndarray]] = {}
        for seed in SEEDS:
            model = load_model(variant, seed, torch_device)
            seed_probabilities[seed] = {}
            local_rows = []
            for index, section_id in enumerate(section_ids):
                probability = tiled_probability(model, normalized_section(section_id)).numpy()[:255, :701]
                seed_probabilities[seed][section_id] = probability
                _names, masks = section_masks(section_id)
                for threshold in THRESHOLDS:
                    metrics = section_metrics(probability, masks, threshold)
                    row = {"variant": variant, "seed": seed, "section_id": section_id, "threshold": threshold, **metrics}
                    local_rows.append(row); per_seed_rows.append(row)
                print(f"phase=SS_CALIBRATION variant={variant} seed={seed} section={index + 1}/{len(section_ids)} dev=LOCKED confirm=LOCKED expert=LOCKED", flush=True)
            selection = _select_threshold(local_rows)
            per_seed_thresholds.append({"variant": variant, "seed": seed, **{key: value for key, value in selection.items() if key != "curve"}})
            del model
            if torch_device.type == "cuda": torch.cuda.empty_cache()
        ensemble_rows = []
        for section_id in section_ids:
            probability = np.mean([seed_probabilities[seed][section_id] for seed in SEEDS], axis=0)
            np.save(probability_root / variant / f"section_{section_id:03d}.npy", probability.astype(np.float32)) if False else None
            _names, masks = section_masks(section_id)
            for threshold in THRESHOLDS:
                ensemble_rows.append({"variant": variant, "seed": "ensemble", "section_id": section_id, "threshold": threshold, **section_metrics(probability, masks, threshold)})
        selection = _select_threshold(ensemble_rows)
        ensemble_thresholds.append({"variant": variant, "seed": "ensemble", **{key: value for key, value in selection.items() if key != "curve"}})
        _write_csv(output / f"threshold_curves/{variant}_ensemble.csv", selection["curve"])
    _write_csv(output / "per_seed_thresholds.csv", per_seed_thresholds)
    _write_csv(output / "ensemble_thresholds.csv", ensemble_thresholds)
    _write_csv(output / "threshold_curves/per_seed_raw.csv", per_seed_rows)
    freeze = {
        "status": "SS_CALIBRATION_FROZEN",
        "authorization_sha256": canonical_hash(authorization),
        "per_seed_thresholds": per_seed_thresholds,
        "ensemble_thresholds": ensemble_thresholds,
        "development_opened": False,
        "confirm_opened": False,
        "expert_opened": False,
    }
    _write_json(output / "CALIBRATION_FREEZE.json", freeze)
    (output / "CALIBRATION_REPORT.md").write_text("# V1.1 calibration\n\nAll 12 final checkpoints were calibrated on clean SS_CALIBRATION only. Development, confirm, and expert remained locked.\n\nSS_CALIBRATION_FROZEN\n")
    return freeze


def _threshold_map(freeze: dict[str, Any], *, ensemble: bool) -> dict[Any, float]:
    rows = freeze["ensemble_thresholds" if ensemble else "per_seed_thresholds"]
    if ensemble:
        return {row["variant"]: float(row["selected_threshold"]) for row in rows}
    return {(row["variant"], int(row["seed"])): float(row["selected_threshold"]) for row in rows}


def _condition_image_and_masks(section_id: int, clean: torch.Tensor, masks: list[np.ndarray], family: str, severity: int) -> tuple[torch.Tensor, list[np.ndarray], int]:
    if family == "clean":
        return clean, masks, 0
    result = apply_perturbation(clean.numpy(), section_id, "full_section", family, severity)
    transformed = [transform_rgb_mask(mask, result) for mask in masks]
    return torch.from_numpy(result.image), transformed, result.seed


@torch.no_grad()
def _tiled_geometry(model: torch.nn.Module, image: torch.Tensor) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    height, width = image.shape[-2:]
    tile, overlap, stride = 256, 64, 192
    tops = list(range(0, height - tile + 1, stride)); lefts = list(range(0, width - tile + 1, stride))
    if tops[-1] != height - tile: tops.append(height - tile)
    if lefts[-1] != width - tile: lefts.append(width - tile)
    sums = {key: np.zeros((height, width), dtype=np.float64) for key in ("c2", "s2", "d", "m", "det")}
    weights = np.zeros((height, width), dtype=np.float64)
    for top in tops:
        for left in lefts:
            output = model(image[:, top : top + tile, left : left + tile].unsqueeze(0).to(device), return_geometry=True)
            scales = []
            for geometry in output["geometry"]:
                values = torch.stack((geometry.c2, geometry.s2, geometry.d, geometry.m, torch.linalg.det(geometry.metric.permute(0, 3, 4, 1, 2))), dim=1)
                scales.append(F.interpolate(values, size=(tile, tile), mode="bilinear", align_corners=False)[0].cpu().numpy())
            value = np.mean(scales, axis=0)
            for channel, key in enumerate(("c2", "s2", "d", "m", "det")):
                sums[key][top : top + tile, left : left + tile] += value[channel]
            weights[top : top + tile, left : left + tile] += 1
    result = {key: (value / weights)[:255, :701].astype(np.float32) for key, value in sums.items()}
    magnitude = np.sqrt(result["c2"] ** 2 + result["s2"] ** 2).clip(1e-8)
    result["c2"] /= magnitude; result["s2"] /= magnitude
    return result


def _geometry_audit(variant: str, models: dict[int, torch.nn.Module], section_ids: list[int]) -> dict[str, Any]:
    angle_values, d_errors, d_values, m_values, det_residuals = [], [], [], [], []
    for index, section_id in enumerate(section_ids):
        image = normalized_section(section_id)
        predictions = [_tiled_geometry(models[seed], image) for seed in SEEDS]
        ensemble = {key: np.mean([row[key] for row in predictions], axis=0) for key in predictions[0]}
        _names, masks = section_masks(section_id)
        target = geometry_target(masks)
        selected = target["supervision"]
        dot = np.clip(ensemble["c2"] * target["target_c2"] + ensemble["s2"] * target["target_s2"], -1, 1)
        angle_values.extend(np.rad2deg(0.5 * np.arccos(dot[selected])).tolist())
        d_errors.extend(np.abs(ensemble["d"][selected] - target["target_d"][selected]).tolist())
        d_values.extend(ensemble["d"][selected].tolist())
        m_values.extend(ensemble["m"][selected].tolist())
        det_residuals.extend(np.abs(ensemble["det"][selected] - 1.0).tolist())
        print(f"phase=SS_DEV_GEOMETRY variant={variant} section={index + 1}/{len(section_ids)} confirm=LOCKED expert=LOCKED", flush=True)
    return {
        "variant": variant,
        "axis_mae_deg": float(np.mean(angle_values)),
        "d_mae": float(np.mean(d_errors)),
        "d_quantiles": dict(zip(("q05", "q25", "q50", "q75", "q95"), map(float, np.quantile(d_values, (0.05, 0.25, 0.5, 0.75, 0.95))))),
        "m_quantiles": dict(zip(("q05", "q25", "q50", "q75", "q95"), map(float, np.quantile(m_values, (0.05, 0.25, 0.5, 0.75, 0.95))))),
        "determinant_residual_mean": float(np.mean(det_residuals)),
        "pixels": len(angle_values),
    }


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["variant"], row["seed"], row["family"], row["severity"])].append(row)
    keys = ("dice", "cldice", "precision", "recall", "auprc", "skeleton_f1_at_2px", "fragmentation", "predicted_foreground_fraction", "white_unknown_foreground")
    return [{"variant": group[0], "seed": group[1], "family": group[2], "severity": group[3], "sections": len(local), **{key: float(np.mean([float(row[key]) for row in local])) for key in keys}} for group, local in sorted(groups.items(), key=str)]


def _per_section_robustness(rows: list[dict[str, Any]], variant: str, seed: Any) -> dict[int, dict[str, float]]:
    selected = [row for row in rows if row["variant"] == variant and str(row["seed"]) == str(seed)]
    by_section: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected: by_section[int(row["section_id"])].append(row)
    result = {}
    for section_id, local in by_section.items():
        clean = next(row for row in local if row["family"] == "clean")
        stress = [row for row in local if row["family"] != "clean"]
        result[section_id] = {
            "clean_dice": float(clean["dice"]), "clean_cldice": float(clean["cldice"]), "white": float(clean["white_unknown_foreground"]),
            "shift_cldice": float(np.mean([float(row["cldice"]) for row in stress])),
            "topo_drop": float(np.mean([max(0.0, float(clean["cldice"]) - float(row["cldice"])) for row in stress])),
        }
    return result


def _paired_bootstrap(a: dict[int, dict[str, float]], b: dict[int, dict[str, float]], key: str, *, ratio: bool = False) -> dict[str, float]:
    ids = sorted(set(a) & set(b))
    av = np.asarray([a[value][key] for value in ids]); bv = np.asarray([b[value][key] for value in ids])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(ids), size=(10000, len(ids)))
    if ratio:
        samples = av[indices].mean(1) / np.maximum(bv[indices].mean(1), 1e-12)
        estimate = float(av.mean() / max(bv.mean(), 1e-12))
    else:
        samples = (av[indices] - bv[indices]).mean(1)
        estimate = float((av - bv).mean())
    return {"estimate": estimate, "ci_lower": float(np.quantile(samples, 0.025)), "ci_upper": float(np.quantile(samples, 0.975)), "bootstrap_seed": BOOTSTRAP_SEED, "resamples": 10000}


def decide_development(rows: list[dict[str, Any]], geometry: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    ensemble = {variant: _per_section_robustness(rows, variant, "ensemble") for variant in VARIANTS}
    clean_dice = _paired_bootstrap(ensemble["B3"], ensemble["B2"], "clean_dice")
    clean_cldice = _paired_bootstrap(ensemble["B3"], ensemble["B2"], "clean_cldice")
    shift_b2 = _paired_bootstrap(ensemble["B3"], ensemble["B2"], "shift_cldice")
    drop_b2 = _paired_bootstrap(ensemble["B3"], ensemble["B2"], "topo_drop", ratio=True)
    shift_b1 = _paired_bootstrap(ensemble["B3"], ensemble["B1"], "shift_cldice")
    drop_b1 = _paired_bootstrap(ensemble["B3"], ensemble["B1"], "topo_drop", ratio=True)
    white_ratio = float(np.mean([row["white"] for row in ensemble["B3"].values()]) / max(np.mean([row["white"] for row in ensemble["B2"].values()]), 1e-12))
    triggering = "shift" if shift_b2["estimate"] >= 0.010 and shift_b2["ci_lower"] > 0 else "drop" if drop_b2["estimate"] <= 0.85 and drop_b2["ci_upper"] < 1 else None
    seed_directions = []
    for seed in SEEDS:
        b3 = _per_section_robustness(rows, "B3", seed); b2 = _per_section_robustness(rows, "B2", seed)
        if triggering == "shift": value = float(np.mean([b3[k]["shift_cldice"] - b2[k]["shift_cldice"] for k in b3]))
        elif triggering == "drop": value = float(np.mean([b3[k]["topo_drop"] for k in b3]) / max(np.mean([b2[k]["topo_drop"] for k in b2]), 1e-12)); value = 1.0 - value
        else: value = 0.0
        seed_directions.append({"seed": seed, "effect": value, "positive": value > 0})
    checks = {
        "geometry": geometry["B3"]["axis_mae_deg"] <= 30 and geometry["B3"]["d_mae"] <= 0.12,
        "clean": clean_dice["estimate"] >= -0.005 and clean_cldice["estimate"] >= -0.005,
        "white": white_ratio <= 1.10,
        "B3_vs_B2": triggering is not None,
        "B3_vs_B1": shift_b1["estimate"] >= 0.005 or drop_b1["estimate"] <= 0.90,
        "seed_consistency": sum(row["positive"] for row in seed_directions) >= 2,
    }
    if not checks["geometry"]: status = "STOP_ANZA_GEOMETRY_NOT_LEARNED"
    elif not checks["clean"]: status = "STOP_ANZA_STABILITY_CLEAN_COST"
    elif not checks["white"]: status = "STOP_ANZA_STABILITY_WHITE_SAFETY"
    elif not checks["B3_vs_B2"]: status = "STOP_ANZA_STABILITY_NO_INCREMENTAL_VALUE"
    elif not checks["B3_vs_B1"]: status = "STRUCTURAL_STABILITY_PASS_ANOSOV_NOT_SPECIFIC"
    elif not checks["seed_consistency"]: status = "STOP_ANZA_STABILITY_SEED_UNSTABLE"
    else: status = "ANZA_STABILITY_MULTISEED_DEV_PASS"
    details = {"checks": checks, "clean_dice_B3_B2": clean_dice, "clean_cldice_B3_B2": clean_cldice, "shift_cldice_B3_B2": shift_b2, "topo_drop_ratio_B3_B2": drop_b2, "shift_cldice_B3_B1": shift_b1, "topo_drop_ratio_B3_B1": drop_b1, "white_ratio_B3_B2": white_ratio, "triggering_metric": triggering, "seed_directions": seed_directions}
    return status, details


@torch.no_grad()
def run_development(*, device: str = "cuda") -> dict[str, Any]:
    freeze = json.loads((RESULT_ROOT / "calibration/CALIBRATION_FREEZE.json").read_text())
    if freeze.get("status") != "SS_CALIBRATION_FROZEN":
        raise PermissionError("development locked until calibration freeze")
    authorization_path = RESULT_ROOT / "SS_DEVELOPMENT_AUTHORIZATION.json"
    authorization = {
        "status": "SS_DEVELOPMENT_AUTHORIZED_ONCE",
        "thresholds": freeze["ensemble_thresholds"],
        "checkpoint_manifest_sha256": sha256_file(RESULT_ROOT / "TWELVE_RUN_COMPLETION_MANIFEST.json"),
        "protocol_sha256": protocol_hash(),
        "perturbation_code_sha256": sha256_file(ROOT / "structural_stability_v1/perturb/suite.py"),
        "metric_code_sha256": sha256_file(Path(__file__)),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "LIRA_locked": True, "confirm_locked": True, "expert_locked": True,
    }
    if authorization_path.exists():
        previous = json.loads(authorization_path.read_text())
        if previous != authorization:
            raise PermissionError("development authorization drift")
    else:
        _write_json(authorization_path, authorization)
    section_ids = list(_split()["splits"]["SS_DEVELOPMENT"])
    torch_device = torch.device(device)
    per_seed_threshold = _threshold_map(freeze, ensemble=False); ensemble_threshold = _threshold_map(freeze, ensemble=True)
    rows: list[dict[str, Any]] = []
    models_by_variant: dict[str, dict[int, torch.nn.Module]] = {}
    for variant in VARIANTS:
        models = {seed: load_model(variant, seed, torch_device) for seed in SEEDS}
        models_by_variant[variant] = models if variant in {"B2", "B3"} else {}
        for section_position, section_id in enumerate(section_ids):
            clean = normalized_section(section_id)
            _names, base_masks = section_masks(section_id)
            conditions = [("clean", 0)] + [(family, severity) for family in FAMILIES for severity in SEVERITIES]
            for family, severity in conditions:
                image, masks, perturb_seed = _condition_image_and_masks(section_id, clean, base_masks, family, severity)
                probabilities = {seed: tiled_probability(model, image).numpy()[:255, :701] for seed, model in models.items()}
                for seed, probability in probabilities.items():
                    rows.append({"variant": variant, "seed": seed, "section_id": section_id, "condition": "clean" if family == "clean" else f"{family}_s{severity}", "family": family, "severity": severity, "perturbation_seed": perturb_seed, "threshold": per_seed_threshold[(variant, seed)], **section_metrics(probability, masks, per_seed_threshold[(variant, seed)])})
                probability = np.mean(list(probabilities.values()), axis=0)
                rows.append({"variant": variant, "seed": "ensemble", "section_id": section_id, "condition": "clean" if family == "clean" else f"{family}_s{severity}", "family": family, "severity": severity, "perturbation_seed": perturb_seed, "threshold": ensemble_threshold[variant], **section_metrics(probability, masks, ensemble_threshold[variant])})
            print(f"phase=SS_DEVELOPMENT variant={variant} section={section_position + 1}/{len(section_ids)} confirm=LOCKED expert=LOCKED", flush=True)
        if variant not in {"B2", "B3"}:
            del models
            if torch_device.type == "cuda": torch.cuda.empty_cache()
    geometry = {variant: _geometry_audit(variant, models_by_variant[variant], section_ids) for variant in ("B2", "B3")}
    status, decision = decide_development(rows, geometry)
    output = RESULT_ROOT / "development"
    _write_csv(output / "per_section_clean.csv", [row for row in rows if row["family"] == "clean"])
    _write_csv(output / "per_section_stress.csv", [row for row in rows if row["family"] != "clean"])
    summaries = _summaries(rows)
    _write_csv(output / "per_seed_summary.csv", [row for row in summaries if str(row["seed"]) != "ensemble"])
    _write_csv(output / "ensemble_summary.csv", [row for row in summaries if str(row["seed"]) == "ensemble"])
    _write_json(output / "bootstrap_results.json", decision)
    _write_json(output / "geometry_metrics.json", geometry)
    result = {"status": status, "sections": len(section_ids), "condition_rows": len(rows), "decision": decision, "geometry": geometry, "development_opened_once": True, "confirm_opened": False, "expert_opened": False}
    _write_json(output / "DEVELOPMENT_MASTER_RESULT.json", result)
    (output / "DEVELOPMENT_REPORT.md").write_text(f"# V1.1 development\n\nAll 50 section-disjoint development sections, four variants, three seeds plus ensemble, clean and all 15 stress cells were evaluated once.\n\n- B3-B2 clean Dice: `{decision['clean_dice_B3_B2']}`\n- B3-B2 clean clDice: `{decision['clean_cldice_B3_B2']}`\n- B3-B2 shifted clDice: `{decision['shift_cldice_B3_B2']}`\n- B3/B2 topology-drop ratio: `{decision['topo_drop_ratio_B3_B2']}`\n- White ratio: `{decision['white_ratio_B3_B2']}`\n- Geometry: `{geometry['B3']}`\n\n{status}\n")
    return result
