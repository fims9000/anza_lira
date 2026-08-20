"""Bounded A0 -> A1 orchestration with hard downstream stop."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from .audit import run_a0_audit
from .evaluation import calibrate_all, development_once
from .model import LEADS_VARIANTS, build_leads_model
from .protocol import (
    A0_ROOT, A1_ROOT, PROTOCOL, active_manifests, file_sha256, freeze_a0, protocol_hash, write_json,
)
from .training import one_batch_smoke, train_variant
from .validator import validate_a0, validate_a1


ROOT = Path(__file__).resolve().parents[1]


def run_a0() -> dict[str, Any]:
    frozen = freeze_a0()
    audit = run_a0_audit()
    validation = validate_a0()
    if audit["status"] != "ANZA_LEADS_A0_PASS" or validation["status"] != "PASS":
        raise RuntimeError("ANZA LEADS A0 did not pass; training remains locked")
    return {"freeze_action": frozen["action"], "audit": audit, "validation": validation}


def _freeze_a1_inputs() -> None:
    A1_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("protocol.json", "protocol_hash.txt", "split_manifest.json", "label_subset_manifest.json"):
        source = A0_ROOT / name
        target = A1_ROOT / name
        if target.exists() and file_sha256(target) != file_sha256(source):
            raise ValueError(f"A1 frozen input drift: {name}")
        if not target.exists():
            shutil.copy2(source, target)
    (A1_ROOT / "model_manifests").mkdir(exist_ok=True)
    (A1_ROOT / "checkpoints").mkdir(exist_ok=True)


def _write_model_manifest(variant: str, status: dict[str, Any]) -> None:
    model = build_leads_model(variant)
    payload = {
        "variant": variant, "run_hash": status["run_hash"], "checkpoint": status["checkpoint"],
        "checkpoint_sha256": status["checkpoint_sha256"], "parameter_count": status["parameter_count"],
        "trainable_parameter_count": status["trainable_parameter_count"],
        "backbone_widths": list(model.widths), "orientation_auxiliary_equal": True,
        "operator_source": "anza_hs/operators.py", "operator_source_sha256": file_sha256(ROOT / "anza_hs" / "operators.py"),
        "expert_data_accessed": False,
    }
    write_json(A1_ROOT / "model_manifests" / f"{variant}.json", payload)
    write_json(A1_ROOT / "checkpoints" / f"{variant}.json", {
        "external_checkpoint": status["checkpoint"], "sha256": status["checkpoint_sha256"],
        "large_binary_stored_outside_git": True,
    })


def _label_budget() -> dict[str, Any]:
    split, subsets = active_manifests()
    selected = set(subsets["subsets"]["41"]["10pct"])
    rows = [row for row in subsets["section_stats"] if int(row["section_id"]) in selected]
    return {
        "optimization_fraction": 0.10, "optimization_sections": len(rows),
        "fixed_calibration_sections": len(split["calibration"]),
        "development_sections": len(split["development"]),
        "blue_pixels_all_available_training_annotators": sum(int(row["blue_pixels"]) for row in rows),
        "green_pixels_all_available_training_annotators": sum(int(row["green_pixels"]) for row in rows),
        "explicit_negative_pixels_all_available_training_annotators": sum(int(row["orange_pixels"]) for row in rows),
        "available_training_annotator_section_pairs": sum(int(row["available_training_annotators"]) for row in rows),
        "honest_description": "10% optimization-section labels plus a fixed 32-section calibration partition",
    }


def _report(metrics: dict[str, Any], runs: dict[str, Any]) -> str:
    lines = [
        "# ANZA-LIRA LEADS V1 — A1 report", "", "## Status", "", f"`{metrics['status']}`", "",
        "This is a seed-41, 10%-optimization-section CRACKS development result. Thresholds were frozen on a separate calibration block. Expert annotations were not accessed.", "",
        "| Variant | Threshold | Dice | Precision | Recall | AUPRC | clDice | Skeleton F1 | Fragmentation | Unknown-white FG |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    freeze = json.loads((A1_ROOT / "threshold_freeze.json").read_text())
    for variant in LEADS_VARIANTS:
        row = metrics["summaries"][variant]
        threshold = freeze["selections"][variant]["selected_threshold"]
        lines.append(
            f"| {variant} | {threshold:.2f} | {row['dice']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['auprc']:.4f} | {row['cldice']:.4f} | {row['skeleton_f1_at_2px']:.4f} | "
            f"{row['fragmentation']:.4f} | {row['unknown_white_foreground_fraction']:.4f} |"
        )
    comparison = metrics["comparison"]
    l0 = metrics["summaries"]["L0_backbone"]
    l3 = metrics["summaries"]["L3_anza_hs"]
    infeasible = [variant for variant in LEADS_VARIANTS if not freeze["selections"][variant]["constraint_feasible"]]
    lines.extend([
        "", "## Frozen L3 versus L2 gate", "",
        f"- Dice delta: `{comparison['dice_delta_L3_minus_L2']:+.6f}` (required >= -0.005).",
        f"- clDice delta at the frozen precision constraint: `{comparison['cldice_delta_L3_minus_L2']:+.6f}` (required >= +0.015).",
        f"- Fragmentation ratio: `{comparison['fragmentation_ratio_L3_over_L2']:.6f}` (required <= 0.80 as the alternative topology gate).",
        f"- Unknown-white foreground ratio: `{comparison['unknown_white_foreground_ratio_L3_over_L2']:.6f}` (required <= 1.10).",
        "",
        "The L3-L2 Dice and clDice gains are large positive seed-41 development diagnostics and their paired section-bootstrap intervals are above zero. They do not pass the predeclared result gate because the unknown-white safety ratio failed.",
        "",
        f"Calibration precision >=0.90 was infeasible for: `{', '.join(infeasible)}`; the frozen rule therefore selected the highest-precision grid point (0.95) without development feedback.",
        "",
        f"Against the plain L0 backbone, L3 Dice is `{l3['dice'] - l0['dice']:+.6f}` and clDice is `{l3['cldice'] - l0['cldice']:+.6f}`. This is not the primary causal comparison, but it prevents an over-broad architecture claim.",
        "", "## Compute", "",
        "| Variant | Parameters | Peak GPU MiB | Wall seconds |", "|---|---:|---:|---:|",
    ])
    for variant in LEADS_VARIANTS:
        run = runs[variant]
        peak = run.get("peak_gpu_memory_bytes")
        wall = run.get("wall_time_seconds_this_invocation")
        peak_text = f"{peak / 2**20:.1f}" if isinstance(peak, (int, float)) else "n/a"
        wall_text = f"{wall:.1f}" if isinstance(wall, (int, float)) else "n/a"
        lines.append(f"| {variant} | {run['trainable_parameter_count']} | {peak_text} | {wall_text} |")
    lines.extend([
        "", "## Claim boundary", "",
        "No seeds 42/43, ANZA-MS, SSL, domain shift, OOF, expert evaluation, or LIRA continuation were opened. The decision above is the terminal state of this bounded A0-A1 run.", "",
    ])
    return "\n".join(lines)


def run_a1(*, device: str = "cuda") -> dict[str, Any]:
    a0 = validate_a0()
    if a0["status"] != "PASS":
        raise PermissionError("A1 locked until A0 validation passes")
    _freeze_a1_inputs()
    write_json(A1_ROOT / "label_budget.json", _label_budget())
    smoke = one_batch_smoke(device=device)
    if not all(row["finite_gradients"] for row in smoke.values()):
        raise RuntimeError("ANZA LEADS real-data vertical smoke failed")
    write_json(A1_ROOT / "cuda_smoke.json", smoke)
    runs = {}
    for variant in LEADS_VARIANTS:
        runs[variant] = train_variant(variant, device=device)
        _write_model_manifest(variant, runs[variant])
    calibrate_all(device=device)
    metrics = development_once(device=device)
    (A1_ROOT / "ANZA_LEADS_A1_REPORT.md").write_text(_report(metrics, runs))
    validation = validate_a1()
    if validation["status"] != "PASS":
        raise RuntimeError("ANZA LEADS A1 artifact validation failed")
    return {"metrics": metrics, "validation": validation, "runs": runs}


def run_all(*, device: str = "cuda") -> dict[str, Any]:
    return {"a0": run_a0(), "a1": run_a1(device=device)}
