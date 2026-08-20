"""Frozen orchestration for the ANZA-S Phase A2 causal oracle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from synthetic.crossing_trace_bench_v4 import benchmark_v4_config

from .evaluator import calibrate, evaluate, gap_identifiability_control, oracle_rows


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "anza_s" / "a2"
PARENT_ROOT = PROJECT_ROOT / "results" / "anza_s" / "oracle"
VERSION = "ANZA_S_PHASE_A2_CAUCHY_GREEN_CAUSAL_V1"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty artifact: {path.name}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def protocol_payload() -> dict[str, Any]:
    parent = json.loads((PARENT_ROOT / "validator.json").read_text())
    if parent.get("research_status") != "ANZA_S_ORACLE_GATE_A_PASS":
        raise ValueError("frozen Phase A parent status changed")
    return {
        "version": VERSION,
        "question": "Does multi-step hyperbolic covariance composition add causal value beyond identical-centerline local anisotropic shadowing?",
        "parent_protocol_sha256": _digest(PARENT_ROOT / "protocol.json"),
        "parent_validator_sha256": _digest(PARENT_ROOT / "validator.json"),
        "benchmark": benchmark_v4_config(),
        "streams": {"calibration": "train[0:512]", "gate": "validation[0:512]", "confirm": "CLOSED", "test": "CLOSED", "CRACKS": "CLOSED", "expert": "CLOSED"},
        "methods": {
            "A0_tangent_terminal": "same tangent centerline; terminal-distance control",
            "A1_isotropic_shadowing": "same tangent centerline; identity covariance",
            "A2_local_anisotropic_reset": "same tangent centerline; each ellipse uses only the immediately preceding J",
            "A3_cocycle_cg_lambda0": "same tangent centerline; recursively composed covariance with lambda=0 null intervention",
            "A3_cocycle_cg_lambda035": "same tangent centerline; recursively composed covariance with frozen lambda=0.35",
        },
        "frozen_parameters": {"steps_K": 3, "delta": 1.0, "lambda_values": [0.0, 0.35], "eta_theta": 2.0, "shadow_temperature": 0.25, "covariance_epsilon": 1e-6},
        "primary_tasks": {
            "P1_x": "x_correct vs x_wrong_turn",
            "P2_parallel": "parallel_correct vs parallel_wrong",
            "P3_curved": "curved_gap vs curved_confuser; primary only if frozen descriptor-comparability rule passes",
        },
        "curved_comparability_rule": "median endpoint-distance ratio in [0.5,2.0] and median axial-agreement difference <=0.25",
        "leakage_control": "StraightGap vs matched NegativeGap is expected geometry-only AUROC about 0.5 and is excluded from the gate",
        "threshold_rule": "per method and predeclared task, train-only maximum TPR operating point represented by inclusive FPR<=0.05 threshold",
        "gate": {"x_delta_tpr": 0.10, "x_ceiling": 0.95, "macro_delta_tpr": 0.08, "paired_bootstrap_repetitions": 10000, "paired_ci_lower_strictly_above_zero": True, "macro_ranking_improves": True, "parallel_fpr_noninferiority": True, "lambda035_must_improve_non_ceiling_task_over_lambda0": True},
        "training_performed": False, "phase_b_opened": False, "confirm_opened": False,
        "test_opened": False, "cracks_data_accessed": False, "expert_data_accessed": False,
    }


def _git_state() -> dict[str, Any]:
    def call(*args: str) -> str:
        return subprocess.run(args, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    return {"branch": call("git", "branch", "--show-current"), "head": call("git", "rev-parse", "HEAD"), "commit_created": False, "push_performed": False}


def _metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"method": method, "task": task, **values}
        for method, method_values in metrics["methods"].items()
        for task, values in method_values["tasks"].items()
    ]


def _report(metrics: dict[str, Any], comparability: dict[str, Any], leakage: dict[str, Any]) -> str:
    lines = [
        "# ANZA-S Phase A2 causal Anosov audit", "", "## Research status", "",
        f"`{metrics['status']}`", "",
        "This is a zero-training oracle audit. It does not report a learned model or a CRACKS result.", "",
        "## Identifiable validation tasks", "",
        "| Method | Macro TPR | Macro FPR | Macro ranking | Macro pAUC@0.05 |", "|---|---:|---:|---:|---:|",
    ]
    for method, values in metrics["methods"].items():
        lines.append(f"| {method} | {values['macro_tpr']:.4f} | {values['macro_fpr']:.4f} | {values['macro_ranking']:.4f} | {values['macro_pauc_fpr_0_05']:.4f} |")
    lines.extend([
        "", "## Causal decision", "",
        f"Frozen gates: `{metrics['gates']}`.", "",
        f"Paired macro TPR delta A3-A2: `{metrics['paired_bootstrap']['estimate']:.4f}` (95% CI `{metrics['paired_bootstrap']['ci95_low']:.4f}` to `{metrics['paired_bootstrap']['ci95_high']:.4f}`).", "",
        f"Lambda intervention: `{metrics['lambda_intervention']}`.", "",
        "## Controls", "",
        f"Curved-confuser comparability: `{comparability}`.", "",
        f"Matched straight-gap leakage control: `{leakage}`.", "",
        "All A1/A2/A3 methods use the exact same tangent centerline. A2 resets the local ellipse; A3 alone composes covariance across steps. Therefore only A3-vs-A2 can support a causal cocycle claim.", "",
        "No training, Phase B, confirm/test, CRACKS, or expert data were opened.", "",
    ])
    return "\n".join(lines)


def run(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload(); encoded = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    path = output_root / "protocol.json"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("Phase A2 protocol drift")
    path.write_text(encoded); protocol_hash = _hash(protocol)
    (output_root / "protocol_hash.txt").write_text(protocol_hash + "\n")
    _json(output_root / "parent_evidence.json", {"protocol_sha256": _digest(PARENT_ROOT / "protocol.json"), "validator_sha256": _digest(PARENT_ROOT / "validator.json")})
    _json(output_root / "code_state.json", _git_state())
    _json(output_root / "data_access_log.json", {"synthetic_train": "calibration", "synthetic_validation": "causal gate", "synthetic_confirm": False, "synthetic_test": False, "phase_b": False, "cracks": False, "expert": False})
    print("phase=ANZA-S-A2 step=train-calibration training=NO downstream=CLOSED", flush=True)
    train_rows, train_diagnostics, train_comparability = oracle_rows("train")
    freeze = calibrate(train_rows, p3_primary=train_comparability["primary_eligible"])
    _json(output_root / "threshold_freeze.json", freeze)
    _json(output_root / "curved_comparability.json", train_comparability)
    print("phase=ANZA-S-A2 step=validation-causal-gate training=NO downstream=CLOSED", flush=True)
    validation_rows, validation_diagnostics, validation_comparability = oracle_rows("validation")
    if validation_comparability["primary_eligible"] != train_comparability["primary_eligible"]:
        raise ValueError("curved-task comparability eligibility drifted between train and validation")
    metrics = evaluate(validation_rows, freeze); metrics["protocol_sha256"] = protocol_hash
    leakage = gap_identifiability_control(validation_rows)
    _json(output_root / "metrics.json", metrics); _json(output_root / "gap_identifiability_control.json", leakage)
    _csv(output_root / "raw_scores.csv", validation_rows)
    _csv(output_root / "task_metrics.csv", _metric_rows(metrics))
    _csv(output_root / "cauchy_green_diagnostics.csv", validation_diagnostics)
    _json(output_root / "calibration_summary.json", {"row_count": len(train_rows), "diagnostic_count": len(train_diagnostics), "curved_comparability": train_comparability})
    state = {"status": metrics["status"], "gate_pass": metrics["gate_pass"], "training_performed": False,
             "next_action": "A separately frozen Phase B may be proposed" if metrics["gate_pass"] else "STOP ANZA-S training; preserve causal negative result",
             "phase_b_opened": False, "confirm_opened": False, "test_opened": False, "cracks_data_accessed": False, "expert_data_accessed": False}
    _json(output_root / "TASK_STATE.json", state)
    _json(output_root / "EVIDENCE.json", {"status": metrics["status"], "protocol_sha256": protocol_hash, "metrics": metrics, "claim_boundary": "zero-training synthetic oracle only", "training_performed": False})
    (output_root / "ANZA_S_A2_REPORT.md").write_text(_report(metrics, validation_comparability, leakage))
    return metrics
