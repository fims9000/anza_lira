"""Bounded seed-41 B0--B3 StressBench H1 experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import evaluate
from .protocol import H0_ROOT, canonical_hash
from .training import one_batch_smoke, predict_variant, train_variant


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "anza_hs" / "h1"
CHECKPOINTS = ROOT.parent / "_wip_backups" / "anza_lira" / "anza_hs_h1_checkpoints"


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(rows)


def _select_max_dice(curve: list[dict[str, float]]) -> dict[str, float]:
    return max(curve, key=lambda row: (row["dice"], row["precision"], -row["threshold"]))


def _report(metrics: dict[str, Any]) -> str:
    lines = ["# ANZA-HS H1 report", "", "## Status", "", f"`{metrics['status']}`", "",
             "This is a seed-41 synthetic development result on frozen StressBench V5. It is not a CRACKS, confirm, multi-seed, or expert result.", "",
             "| Variant | Threshold | Dice | Precision | Recall | clDice | Fragmentation | Branch preservation | Parallel false connection |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, value in metrics["variants"].items():
        row = value["gate"]["overall"]
        lines.append(f"| {variant} | {value['threshold']:.2f} | {row['dice']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['cldice']:.4f} | {row['fragmentation']:.4f} | {row['branch_preservation']:.4f} | {row['parallel_false_connection']:.4f} |")
    lines.extend(["", "## Frozen B3 versus B2 gate", "", f"`{metrics['comparison']}`", "",
                  "No lambda/M/base-scale alternative was used. H2, confirm, CRACKS, continuation, and expert data remained closed.", ""])
    return "\n".join(lines)


def run(device: str = "cuda") -> dict[str, Any]:
    h0 = json.loads((H0_ROOT / "validator.json").read_text()); protocol = json.loads((H0_ROOT / "protocol.json").read_text())
    protocol_hash = canonical_hash(protocol)
    if h0.get("research_status") != "ANZA_HS_H0_PASS" or not h0.get("H1_authorized") or h0.get("protocol_sha256") != protocol_hash:
        raise ValueError("H0 has not authorized H1")
    OUTPUT.mkdir(parents=True, exist_ok=True); (OUTPUT / "runs").mkdir(exist_ok=True)
    _json(OUTPUT / "parent_h0.json", {"protocol_sha256": protocol_hash, "stressbench_sha256": h0["stressbench_sha256"], "validator_status": h0["research_status"]})
    _json(OUTPUT / "data_access_log.json", {"synthetic_train": True, "synthetic_dev_calibration": "dev[0:44]", "synthetic_dev_gate": "dev[44:264]", "confirm": False, "test": False, "cracks": False, "continuation": False, "expert": False})
    smoke = one_batch_smoke(protocol, device=device)
    if not all(row["finite_gradients"] for row in smoke.values()):
        raise ValueError("H1 CUDA vertical smoke failed")
    _json(OUTPUT / "cuda_smoke.json", smoke)
    runs = {}
    for variant in protocol["matrix"]:
        runs[variant] = train_variant(variant, protocol=protocol, protocol_hash=protocol_hash, output_root=OUTPUT / "runs", checkpoint_root=CHECKPOINTS, device=device)
    probabilities = {}; samples = None
    for variant in protocol["matrix"]:
        values, local_samples = predict_variant(variant, Path(runs[variant]["checkpoint"]), device=device)
        probabilities[variant] = values
        if samples is None: samples = local_samples
    calibration_samples = samples[:44]; gate_samples = samples[44:]
    grid = [float(value) for value in protocol["threshold"]["grid"]]
    curves: dict[str, list[dict[str, float]]] = {}
    for variant in protocol["matrix"]:
        curves[variant] = []
        for threshold in grid:
            summary, _rows = evaluate(probabilities[variant][:44], calibration_samples, threshold)
            curves[variant].append({"threshold": threshold, **summary["overall"]})
    selected = {variant: _select_max_dice(curves[variant]) for variant in protocol["matrix"] if variant != "B3_anza_hyperbolic"}
    target_precision = selected["B2_generic_aniso"]["precision"]
    selected["B3_anza_hyperbolic"] = min(curves["B3_anza_hyperbolic"], key=lambda row: (abs(row["precision"] - target_precision), -row["dice"], row["threshold"]))
    variants = {}; raw_rows = []
    for variant in protocol["matrix"]:
        threshold = float(selected[variant]["threshold"])
        gate, rows = evaluate(probabilities[variant][44:], gate_samples, threshold)
        for row in rows: raw_rows.append({"variant": variant, "threshold": threshold, **row})
        variants[variant] = {"threshold": threshold, "calibration": selected[variant], "gate": gate, "run": runs[variant]}
    b2 = variants["B2_generic_aniso"]["gate"]["overall"]; b3 = variants["B3_anza_hyperbolic"]["gate"]["overall"]
    dice_delta = b3["dice"] - b2["dice"]; cldice_delta = b3["cldice"] - b2["cldice"]
    fragmentation_ratio = b3["fragmentation"] / b2["fragmentation"] if b2["fragmentation"] > 0 else None
    dice_ok = dice_delta >= float(protocol["gate"]["dice_noninferiority"])
    cldice_ok = cldice_delta >= float(protocol["gate"]["cldice_gain"])
    fragmentation_ok = fragmentation_ratio is not None and fragmentation_ratio <= float(protocol["gate"]["fragmentation_relative_max"])
    gate_pass = bool(dice_ok and (cldice_ok or fragmentation_ok))
    comparison = {
        "dice_delta_B3_minus_B2": dice_delta, "cldice_delta_B3_minus_B2": cldice_delta,
        "fragmentation_ratio_B3_over_B2": fragmentation_ratio,
        "gate_checks": {"dice_noninferiority": dice_ok, "cldice_gain": cldice_ok, "fragmentation_reduction": fragmentation_ok},
        "matched_precision_target_from_B2_calibration": target_precision,
        "gate_precision_difference_B3_minus_B2": b3["precision"] - b2["precision"],
    }
    metrics = {"status": "ANZA_HS_H1_PASS" if gate_pass else "HYPERBOLIC_CONSTRAINT_NOT_INCREMENTAL", "gate_pass": gate_pass,
               "protocol_sha256": protocol_hash, "stressbench_sha256": h0["stressbench_sha256"], "variants": variants,
               "comparison": comparison, "seed": 41, "training_performed": True, "confirm_opened": False, "test_opened": False,
               "H2_opened": False, "cracks_accessed": False, "continuation_trained": False, "expert_accessed": False,
               "lambda_tuned": False, "M_tuned": False, "base_scale_alternative_used": False}
    _json(OUTPUT / "threshold_freeze.json", {"grid": grid, "selected": selected, "B3_match_target_precision": target_precision})
    _json(OUTPUT / "metrics.json", metrics); _json(OUTPUT / "calibration_curves.json", curves); _csv(OUTPUT / "raw_per_sample.csv", raw_rows)
    (OUTPUT / "ANZA_HS_H1_REPORT.md").write_text(_report(metrics))
    _json(OUTPUT / "TASK_STATE.json", {"status": metrics["status"], "gate_pass": gate_pass, "next_action": "Freeze H2 before shadowing stability" if gate_pass else "STOP local hyperbolic claim; do not run H2 under this path",
          "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False})
    return metrics
