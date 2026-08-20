"""Fail-closed K2 source freeze, seed-41 matrix, evaluation, and STOP."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from anza_ks.benchmark.static_signature import static_signature
from anza_ks.features import dynamic_feature_vector

from .benchmark import SPLIT_SIZES, generate_sample, split_hash
from .dense_features import features_from_patches
from .evaluation import apply_gates, calibration_curve, paired_bootstrap_improvement, pixel_summary, recall95_threshold, select_threshold, structural_summary, structure_scores
from .features import shear_ks_feature_vector
from .model import VARIANTS
from .protocol import protocol, protocol_hash
from .training import one_batch_smoke, predict_split, train_variant


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/anza_ks/k2"
FREEZE = RESULT / "freeze"
CHECKPOINTS = ROOT.parent / "_wip_backups/anza_lira/anza_ks_k2_checkpoints"


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict[str, Any]:
    paths = sorted((ROOT / "anza_ks_k2").glob("*.py")) + sorted((ROOT / "scripts").glob("*anza_ks_k2*.py")) + sorted((ROOT / "tests").glob("test_anza_ks_k2*.py"))
    files = [{"path": str(path.relative_to(ROOT)), "sha256": _sha(path)} for path in paths]
    digest = hashlib.sha256()
    for row in files: digest.update(row["path"].encode()); digest.update(row["sha256"].encode())
    return {"files": files, "sha256": digest.hexdigest()}


def dense_equivalence_receipt() -> dict[str, Any]:
    rng = np.random.default_rng(2_019_460); patch = rng.normal(size=(17, 17)); tensor = torch.tensor(patch, dtype=torch.float64)
    references = {"static": static_signature(patch), "cat_raw": dynamic_feature_vector(patch, "K1_C_cat_raw"), "cat_ks": dynamic_feature_vector(patch, "K1_D_anza_ks"), "shear_ks": shear_ks_feature_vector(patch)}
    errors = {method: float(np.max(np.abs(features_from_patches(tensor, method).numpy() - reference))) for method, reference in references.items()}
    return {"errors": errors, "tolerance": 1e-6, "pass": max(errors.values()) <= 1e-6, "fixture": "orientation-0 pre-rotation float64 patch"}


def freeze_pretraining() -> dict[str, Any]:
    FREEZE.mkdir(parents=True, exist_ok=True)
    value = protocol(); _json(FREEZE / "protocol.json", value)
    benchmark = json.loads((FREEZE / "benchmark_manifest.json").read_text())
    # Recompute immutable content hashes after the final generator implementation.
    recomputed = {split: split_hash(split) for split in SPLIT_SIZES}
    if recomputed != benchmark["hashes"]: raise ValueError("K2 benchmark drift after hash freeze")
    equivalence = dense_equivalence_receipt()
    if not equivalence["pass"]: raise ValueError("K2 dense equivalence failed")
    sources = source_manifest(); _json(FREEZE / "source_manifest.json", sources); _json(FREEZE / "dense_equivalence.json", equivalence)
    receipt = {
        "status": "ANZA_KS_K2_PRETRAINING_PASS",
        "protocol_sha256": protocol_hash(value),
        "benchmark_manifest_sha256": benchmark["manifest_sha256"],
        "source_sha256": sources["sha256"],
        "feature_norm_sha256": _sha(FREEZE / "feature_norm.json"),
        "k1_5_status": json.loads((ROOT / "results/anza_ks/k1_5/metrics.json").read_text())["status"],
        "confirm_evaluated": False,
        "training_authorized": True,
    }
    _json(FREEZE / "pretraining_receipt.json", receipt); return receipt


def _matched_result(probabilities: list[np.ndarray], samples: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    return pixel_summary(probabilities, samples, threshold)[0]


def _report(metrics: dict[str, Any]) -> str:
    lines = [
        "# ANZA-KS K2 Seed-41 Report", "", f"Status: `{metrics['status']}`", "",
        "This is a frozen seed-41 synthetic segmentation-transfer result. K2 confirm, seeds 42/43, CRACKS, and expert data remained closed.", "",
        f"K1.5 attribution: `{metrics['k1_5_status']}`. Therefore an Anosov-specific claim is restricted unless M4 causally beats M2.", "",
        "| Variant | Params | Peak MB | Dice | clDice | Fragmentation | Mechanism TPR | Mechanism FPR | False/Total |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        value = metrics["variants"][variant]; natural = value["natural_primary"]; mechanism = value["mechanism"]
        lines.append(f"| {variant} | {value['run']['parameter_count']} | {value['resources']['peak_memory_mb']:.1f} | {natural['dice']:.4f} | {natural['cldice']:.4f} | {natural['fragmentation']:.4f} | {mechanism['target_recall']:.4f} | {mechanism['false_positive_rate']:.4f} | {mechanism['false_positive_count']}/{mechanism['scene_count']} |")
    lines += ["", "## Frozen gates", "", f"`{json.dumps(metrics['gates'], sort_keys=True)}`", "", "## Claim boundary", "", "Synthetic feature and segmentation results do not establish real seismic improvement. No threshold, architecture, split, or feature family was changed after development results.", ""]
    return "\n".join(lines)


def run(device: str = "cuda") -> dict[str, Any]:
    receipt = json.loads((FREEZE / "pretraining_receipt.json").read_text()); value = json.loads((FREEZE / "protocol.json").read_text())
    if receipt["status"] != "ANZA_KS_K2_PRETRAINING_PASS" or receipt["source_sha256"] != source_manifest()["sha256"] or receipt["protocol_sha256"] != protocol_hash(value):
        raise ValueError("K2 pretraining freeze is absent or source/protocol drifted")
    smoke = one_batch_smoke(value, FREEZE / "feature_norm.json", device=device); _json(RESULT / "cuda_smoke.json", smoke)
    if not all(row["finite_gradients"] for row in smoke.values()): raise ValueError("K2 CUDA smoke failed")
    runs = {}
    for variant in VARIANTS:
        runs[variant] = train_variant(variant, protocol=value, protocol_hash=receipt["protocol_sha256"], feature_norm_path=FREEZE / "feature_norm.json", result_root=RESULT, checkpoint_root=CHECKPOINTS, device=device)

    calibration_natural = list(range(3072, 4096, 2)); calibration_mechanism = list(range(3073, 4096, 2))
    calibration_probabilities: dict[str, dict[str, list[np.ndarray]]] = {}; curves = {}; structural_thresholds = {}
    for variant in VARIANTS:
        natural_prob, natural_samples = predict_split(variant, Path(runs[variant]["checkpoint"]), "train", calibration_natural, FREEZE / "feature_norm.json", device=device)
        mechanism_prob, mechanism_samples = predict_split(variant, Path(runs[variant]["checkpoint"]), "train", calibration_mechanism, FREEZE / "feature_norm.json", device=device)
        calibration_probabilities[variant] = {"natural": natural_prob, "mechanism": mechanism_prob}
        curves[variant] = calibration_curve(natural_prob, natural_samples)
        target_scores, _ = structure_scores(mechanism_prob, mechanism_samples); structural_thresholds[variant] = recall95_threshold(target_scores)
    m1_primary = select_threshold(curves["M1_static"], "dice"); m1_row = next(row for row in curves["M1_static"] if row["threshold"] == m1_primary)
    thresholds = {}
    for variant in VARIANTS:
        thresholds[variant] = {
            "primary": select_threshold(curves[variant], "dice"),
            "matched_precision_M1": select_threshold(curves[variant], "precision", m1_row["precision"]),
            "matched_dice_M1": select_threshold(curves[variant], "dice", m1_row["dice"]),
            "mechanism_recall95": structural_thresholds[variant],
        }
    _json(RESULT / "threshold_freeze.json", {"calibration_indices": {"natural": [3072, 4094, 2], "mechanism": [3073, 4095, 2]}, "thresholds": thresholds, "M1_targets": {"precision": m1_row["precision"], "dice": m1_row["dice"]}, "confirm_opened": False})
    _json(RESULT / "calibration_curves.json", curves)

    variants: dict[str, Any] = {}; mechanism_indicators = {}; raw_rows = []
    for variant in VARIANTS:
        natural_prob, natural_samples = predict_split(variant, Path(runs[variant]["checkpoint"]), "dev-natural", list(range(1024)), FREEZE / "feature_norm.json", device=device)
        mechanism_prob, mechanism_samples = predict_split(variant, Path(runs[variant]["checkpoint"]), "dev-mechanism", list(range(1024)), FREEZE / "feature_norm.json", device=device)
        perturbed_prob, _ = predict_split(variant, Path(runs[variant]["checkpoint"]), "dev-natural", list(range(1024)), FREEZE / "feature_norm.json", device=device, perturb=True)
        primary, rows = pixel_summary(natural_prob, natural_samples, thresholds[variant]["primary"])
        matched_precision = _matched_result(natural_prob, natural_samples, thresholds[variant]["matched_precision_M1"])
        matched_dice = _matched_result(natural_prob, natural_samples, thresholds[variant]["matched_dice_M1"])
        perturbed = _matched_result(perturbed_prob, natural_samples, thresholds[variant]["primary"])
        target_scores, distractor_scores = structure_scores(mechanism_prob, mechanism_samples)
        mechanism = structural_summary(target_scores, distractor_scores, thresholds[variant]["mechanism_recall95"])
        mechanism_indicators[variant] = mechanism.pop("false_accept_indicators")
        variants[variant] = {"run": runs[variant], "resources": smoke[variant], "natural_primary": primary, "natural_matched_precision_M1": matched_precision, "natural_matched_dice_M1": matched_dice, "mechanism": mechanism, "robustness": {"clean_cldice": primary["cldice"], "perturbed_cldice": perturbed["cldice"], "cldice_degradation": primary["cldice"] - perturbed["cldice"]}}
        for index, row in enumerate(rows): raw_rows.append({"variant": variant, "split": "dev-natural", "index": index, **row})
    matched = {}
    for comparator in ("M1_static", "M2_shear_ks", "M3_cat_raw"):
        fragmentation_reference = variants[comparator]["natural_primary"]["fragmentation"]
        matched[comparator] = {
            "cldice_delta": variants["M4_anza_ks"]["natural_matched_precision_M1"]["cldice"] - variants[comparator]["natural_primary"]["cldice"] if comparator == "M1_static" else None,
            "fragmentation_ratio": (variants["M4_anza_ks"]["natural_matched_dice_M1"]["fragmentation"] / fragmentation_reference) if comparator == "M1_static" and fragmentation_reference > 0 else None,
        }
    bootstraps = {name: paired_bootstrap_improvement(mechanism_indicators[control], mechanism_indicators["M4_anza_ks"]) for name, control in (("M4_vs_M1", "M1_static"), ("M4_vs_M2", "M2_shear_ks"), ("M4_vs_M3", "M3_cat_raw"))}
    status, gates = apply_gates(variants, matched, bootstraps)
    metrics = {"status": status, "k1_5_status": receipt["k1_5_status"], "seed": 41, "protocol_sha256": receipt["protocol_sha256"], "benchmark_manifest_sha256": receipt["benchmark_manifest_sha256"], "variants": variants, "matched_operating_points": matched, "bootstraps": bootstraps, "gates": gates, "training_performed": True, "seeds_42_43_opened": False, "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False}
    _json(RESULT / "metrics.json", metrics)
    with (RESULT / "raw_per_scene.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0])); writer.writeheader(); writer.writerows(raw_rows)
    (RESULT / "ANZA_KS_K2_REPORT.md").write_text(_report(metrics))
    _json(RESULT / "TASK_STATE.json", {"status": status, "next_action": "STOP after seed-41 per frozen protocol; seeds 42/43 require a separately authorized continuation only if PASS", "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False})
    return metrics
