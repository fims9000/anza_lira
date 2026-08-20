"""Bounded RC1 orchestration: freeze, smoke, seed41, calibration, development, stop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import build_leads_model
from .protocol import file_sha256, write_json
from .rc1_evaluation import calibrate_all, development_once
from .rc1_protocol import ROOT, VARIANTS, freeze_protocol, verify_parent_immutable
from .rc1_training import one_batch_smoke, train_variant
from .rc1_validator import validate_rc1


PROJECT = Path(__file__).resolve().parents[1]


def _model_manifest(variant: str, status: dict[str, Any]) -> None:
    model = build_leads_model(variant)
    write_json(ROOT / "model_manifests" / f"{variant}.json", {
        "variant": variant, "run_hash": status["run_hash"], "checkpoint": status["checkpoint"],
        "checkpoint_sha256": status["checkpoint_sha256"], "parameter_count": status["parameter_count"],
        "trainable_parameter_count": status["trainable_parameter_count"], "backbone_widths": list(model.widths),
        "exact_parent_operator_source": "anza_hs/operators.py",
        "operator_source_sha256": file_sha256(PROJECT / "anza_hs" / "operators.py"),
        "expert_data_accessed": False,
    })


def _report(metrics: dict[str, Any]) -> str:
    lines = [
        "# ANZA-LIRA LEADS RC1 — Risk-calibrated frontier", "", "## Status", "", f"`{metrics['status']}`", "",
        "RC1 changed only the cross-fit sections, score-complete calibration frontier, and unsupported-white safety metric. The parent A1 STOP remains immutable. Expert annotations were not accessed.", "",
    ]
    if not metrics.get("development_opened"):
        lines.extend(["Precision >=0.90 with nonzero recall was infeasible on calibration for L2 or L3. Development remained closed.", ""])
    else:
        freeze = json.loads((ROOT / "threshold_freeze.json").read_text())
        lines.extend([
            "| Variant | Threshold | Precision | Recall | Dice | clDice | AUPRC | Unsupported white |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for variant in VARIANTS:
            row = metrics["summaries"][variant]
            threshold = freeze["selections"][variant]["selected_threshold"]
            lines.append(f"| {variant} | {threshold:.6f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['dice']:.4f} | {row['cldice']:.4f} | {row['auprc']:.4f} | {row['unsupported_white_foreground_fraction']:.4f} |")
        deltas = metrics["deltas"]; bootstrap = metrics["bootstrap"]
        lines.extend([
            "", "## Primary result", "",
            f"- L3-L2 Dice: `{deltas['dice_L3_minus_L2']:+.6f}`; paired section 95% CI `[{bootstrap['dice']['ci95_low']:+.6f}, {bootstrap['dice']['ci95_high']:+.6f}]`.",
            f"- L3-L2 clDice: `{deltas['cldice_L3_minus_L2']:+.6f}`; paired section 95% CI `[{bootstrap['cldice']['ci95_low']:+.6f}, {bootstrap['cldice']['ci95_high']:+.6f}]`.",
            f"- L3-L2 AUPRC: `{deltas['auprc_L3_minus_L2']:+.6f}`.",
            f"- Unsupported-white ratio L3/L2: `{deltas['unsupported_white_ratio_L3_L2']:.6f}`; L3/L0: `{deltas['unsupported_white_ratio_L3_L0']:.6f}`.",
            "", "## Frozen causal checks", "",
        ])
        for key, value in metrics["checks"].items():
            lines.append(f"- `{key}`: `{value}`")
        lines.extend(["", "The topology-precision frontier is diagnostic only and cannot rescue the primary frozen operating-point gate.", ""])
    lines.extend([
        "## Claim boundary", "",
        "No seeds 42/43, ANZA-MS, SSL, domain shift, LIRA continuation, OOF, or expert evaluation were opened. RC1 does not alter the negative parent A1 decision.", "",
    ])
    return "\n".join(lines)


def run_rc1(*, device: str = "cuda") -> dict[str, Any]:
    frozen = freeze_protocol()
    if not verify_parent_immutable():
        raise ValueError("parent A1 changed before RC1 execution")
    smoke = one_batch_smoke(device=device)
    if not all(row["finite_gradients"] for row in smoke.values()):
        raise RuntimeError("RC1 vertical smoke failed")
    write_json(ROOT / "cuda_smoke.json", smoke)
    runs = {}
    for variant in VARIANTS:
        runs[variant] = train_variant(variant, device=device); _model_manifest(variant, runs[variant])
    calibration = calibrate_all(device=device)
    metrics = development_once(device=device)
    (ROOT / "ANZA_LEADS_RC1_REPORT.md").write_text(_report(metrics))
    validation = validate_rc1()
    if validation["status"] != "PASS":
        raise RuntimeError("RC1 artifact validation failed")
    return {"freeze": frozen["action"], "runs": runs, "calibration": calibration, "metrics": metrics, "validation": validation}
