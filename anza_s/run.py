"""Frozen ANZA-S Phase-A zero-training oracle orchestration."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from anza2.eval.low_fpr import sampled_operating_curve
from synthetic.crossing_trace_bench_v4 import benchmark_v4_config

from .figures import generate_figures
from .oracle_eval import METHODS, calibrate_thresholds, evaluate, oracle_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "anza_s" / "oracle"
PARENT_ROOT = PROJECT_ROOT / "results" / "anza2" / "phase3d_ab"
VERSION = "ANZA_S_ANOSOV_COCYCLE_SHADOWING_ORACLE_V1"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    if parent.get("research_status") != "FINAL_STOP_MODE_STATE_ORACLE_NO_VALUE":
        raise ValueError("frozen ANZA-2 parent status changed")
    return {
        "version": VERSION,
        "hypothesis_scope": "new ANZA-S oracle; does not reopen or rewrite negative ANZA-2 history",
        "parent_protocol_sha256": digest(PARENT_ROOT / "protocol.json"),
        "parent_validator_sha256": digest(PARENT_ROOT / "validator.json"),
        "benchmark": benchmark_v4_config(),
        "streams": {"calibration": "train[0:512]", "gate": "validation[0:512]", "confirm": "CLOSED", "test": "CLOSED"},
        "cases": ["StraightGap", "derived CurvedGap from frozen curved_fault", "X correct", "X wrong turn", "ParallelFault", "NegativeGap", "T/Y diagnostic"],
        "derived_curved_gap": {"source": "curved_fault", "polyline_indices": [43, 53], "generator_modified": False},
        "methods": {
            "O0_scalar_anza": "frozen scalar ANZA oracle widest path",
            "O1_mode_state": "frozen failed mode-state oracle widest path",
            "O2_tangent_streamline": "top-1 axial tangent rollout, terminal spatial meeting",
            "O3_cocycle_rollout": "hyperbolic frame transport rollout, terminal spatial meeting",
            "O4_cocycle_shadowing": "O3 rollout plus exact two-sided soft-min position/orientation energy",
        },
        "cocycle": {"steps_K": 3, "delta": 1.0, "fixed_lambda": 0.35, "mode_match_temperature": 0.08, "determinant": 1.0},
        "shadowing": {"sigma_x": 1.5, "eta_theta": 2.0, "temperature": 0.25, "soft_min": "unnormalized exact packet sum"},
        "threshold_rule": {"source": "train only", "maximum_fpr_each_negative_task": 0.05, "negative_tasks": ["x_wrong_turn", "parallel_wrong", "negative_gap"]},
        "gate": {
            "comparison": "O4 must pass separately against O0 and O2",
            "x_wrong_turn_relative_reduction_minimum": 0.50,
            "macro_positive_noninferiority_margin": 0.01,
            "parallel_false_bridge_noninferiority": True,
            "negative_gap_false_bridge_noninferiority": True,
            "curved_recall_noninferiority_margin": 0.01,
        },
        "matched_negative_identifiability_audit": "Frozen paired positive/negative geometry is identical for trajectory-only O2/O3/O4; retained as required hard negative and disclosed before gate.",
        "training_performed": False, "confirm_opened": False,
        "cracks_data_accessed": False, "expert_data_accessed": False,
    }


def _git_state() -> dict[str, Any]:
    def call(*args: str) -> str:
        return subprocess.run(args, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    return {"branch": call("git", "branch", "--show-current"), "head": call("git", "rev-parse", "HEAD"), "commit_created": False, "push_performed": False}


def _per_case(rows: list[dict[str, Any]], thresholds: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        threshold = thresholds[method]["threshold"]
        for task in sorted({row["task"] for row in rows}):
            selected = [row for row in rows if row["method"] == method and row["task"] == task]
            output.append({"method": method, "task": task, "label": int(selected[0]["label"]), "count": len(selected),
                           "mean_score": float(np.mean([row["score"] for row in selected])),
                           "acceptance_rate": float(np.mean([row["score"] >= threshold for row in selected])),
                           "threshold_from_train": threshold})
    return output


def _operating(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        positive = np.asarray([row["score"] for row in selected if row["label"] == 1])
        negative = np.asarray([row["score"] for row in selected if row["label"] == 0])
        output.extend({"method": method, **point} for point in sampled_operating_curve(positive, negative))
    return output


def _report(metrics: dict[str, Any]) -> str:
    lines = ["# ANZA-S zero-training oracle report", "", "## Status", "", f"`{metrics['status']}`", "",
             "This phase evaluated only perfect generator-derived fields. No neural network, segmentation model, CRACKS section, expert mask, confirm stream, or synthetic test stream was opened.", "",
             "## Validation at train-frozen operating points", "",
             "| Method | Macro positive recall | X wrong-turn FPR | Parallel false bridge | Negative-gap false bridge | Curved recall |", "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        row = metrics["methods"][method]
        lines.append(f"| {method} | {row['positive_recall_macro_primary']:.4f} | {row['x_wrong_turn_fpr']:.4f} | {row['parallel_false_bridge']:.4f} | {row['negative_gap_false_bridge']:.4f} | {row['positive_recall_by_task']['curved_gap']:.4f} |")
    lines.extend(["", "## Gate A", "", f"Checks: `{metrics['baseline_gate_checks']}`.", "",
                  "The frozen v4 matched positive and negative straight-gap geometry is indistinguishable to trajectory-only O2/O3/O4. This was detected before evaluation, retained rather than repaired post hoc, and is part of the reported gate outcome.", "",
                  f"Causal diagnostic: O2/O3 maximum score difference is `{metrics['causal_diagnostics']['o2_o3_max_absolute_score_difference']:.3g}`. Therefore the formal O4 gate is evidence for the combined shadowing readout, not yet for an incremental hyperbolic-cocycle rollout effect. A generic tangent+shadowing control was not part of the frozen O0-O4 packet and was not added post hoc.", "",
                  f"At the safe operating point O4 X-correct recall is `{metrics['methods']['O4_cocycle_shadowing']['positive_recall_by_task']['x_correct']:.4f}` and StraightGap recall is `{metrics['methods']['O4_cocycle_shadowing']['positive_recall_by_task']['straight_gap']:.4f}`; these limitations must remain visible in any next protocol.", "",
                  ("Gate A passed; only a separately frozen field-learning phase may follow. No training was run here." if metrics["gate_pass"] else "Gate A failed. Under the task packet this is the final falsification of the Anosov dynamical architecture; no field learning, CocycleConv training, confirm, CRACKS, or expert evaluation is authorized."), "",
                  "The terms cocycle and shadowing are used as local Anosov-inspired computational constructions, not as a claim that the image plane is a global Anosov system.", ""])
    return "\n".join(lines)


def run(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True); (output_root / "figures").mkdir(exist_ok=True)
    protocol = protocol_payload(); encoded = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists() and protocol_path.read_text() != encoded:
        raise ValueError("ANZA-S oracle protocol drift")
    protocol_path.write_text(encoded); protocol_hash = canonical_hash(protocol)
    (output_root / "protocol_hash.txt").write_text(protocol_hash + "\n")
    _json(output_root / "parent_evidence.json", {"protocol_sha256": digest(PARENT_ROOT / "protocol.json"), "validator_sha256": digest(PARENT_ROOT / "validator.json"), "validator": json.loads((PARENT_ROOT / "validator.json").read_text())})
    _json(output_root / "code_state.json", _git_state())
    _json(output_root / "data_access_log.json", {"synthetic_train": "calibration", "synthetic_validation": "gate", "synthetic_confirm": False, "synthetic_test": False, "cracks": False, "expert": False})
    print("phase=ANZA-S-A step=train-calibration candidates=536 training=NO confirm=CLOSED", flush=True)
    train_rows, _train_trajectories = oracle_rows("train")
    thresholds = calibrate_thresholds(train_rows); _json(output_root / "threshold_freeze.json", thresholds)
    print("phase=ANZA-S-A step=validation-gate candidates=536 training=NO confirm=CLOSED", flush=True)
    validation_rows, trajectories = oracle_rows("validation")
    metrics = evaluate(validation_rows, thresholds); metrics["protocol_sha256"] = protocol_hash
    _json(output_root / "metrics.json", metrics)
    _csv(output_root / "raw_scores.csv", validation_rows)
    _csv(output_root / "per_case.csv", _per_case(validation_rows, thresholds))
    _csv(output_root / "trajectory_points.csv", trajectories)
    _csv(output_root / "shadowing_scores.csv", [row for row in validation_rows if row["method"] == "O4_cocycle_shadowing"])
    _csv(output_root / "operating_curve.csv", _operating(validation_rows))
    figures = generate_figures(output_root / "figures", trajectories, validation_rows)
    _json(output_root / "figure_manifest.json", {"source": "validation oracle rows", "figures": figures, "selection": "lowest deterministic validation index per required case"})
    task = {"status": metrics["status"], "gate_pass": metrics["gate_pass"], "training_performed": False,
            "next_action": "Freeze Phase B before field learning" if metrics["gate_pass"] else "FINAL STOP; do not train ANZA-S",
            "confirm_opened": False, "cracks_data_accessed": False, "expert_data_accessed": False}
    _json(output_root / "TASK_STATE.json", task)
    _json(output_root / "EVIDENCE.json", {"status": metrics["status"], "protocol_sha256": protocol_hash,
          "metrics": metrics, "training_performed": False, "claim_boundary": "zero-training synthetic oracle only"})
    (output_root / "ANZA_S_ORACLE_REPORT.md").write_text(_report(metrics))
    return metrics
