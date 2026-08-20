#!/usr/bin/env python3
"""Fail-closed validation of ANZA-S Phase A2 artifacts and access locks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "results" / "anza_s" / "a2"
PARENT = ROOT / "results" / "anza_s" / "oracle"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate() -> dict[str, Any]:
    required = (
        "protocol.json", "protocol_hash.txt", "parent_evidence.json", "code_state.json", "data_access_log.json",
        "threshold_freeze.json", "curved_comparability.json", "calibration_summary.json", "raw_scores.csv",
        "task_metrics.csv", "cauchy_green_diagnostics.csv", "gap_identifiability_control.json", "metrics.json",
        "TASK_STATE.json", "EVIDENCE.json", "ANZA_S_A2_REPORT.md",
    )
    failures = [name for name in required if not (PHASE / name).is_file() or (PHASE / name).stat().st_size == 0]
    if failures:
        result = {"status": "FAIL", "research_status": "INVALID_ANZA_S_A2_ARTIFACTS", "failures": failures}
        PHASE.mkdir(parents=True, exist_ok=True); (PHASE / "validator.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result
    protocol = _load(PHASE / "protocol.json"); metrics = _load(PHASE / "metrics.json")
    freeze = _load(PHASE / "threshold_freeze.json"); access = _load(PHASE / "data_access_log.json")
    leakage = _load(PHASE / "gap_identifiability_control.json")
    state = _load(PHASE / "TASK_STATE.json"); stored_hash = (PHASE / "protocol_hash.txt").read_text().strip()
    if protocol.get("version") != "ANZA_S_PHASE_A2_CAUCHY_GREEN_CAUSAL_V1" or stored_hash != _canonical(protocol):
        failures.append("protocol version/hash mismatch")
    if protocol.get("parent_protocol_sha256") != _digest(PARENT / "protocol.json") or protocol.get("parent_validator_sha256") != _digest(PARENT / "validator.json"):
        failures.append("Phase A parent drift")
    if _load(PARENT / "validator.json").get("research_status") != "ANZA_S_ORACLE_GATE_A_PASS":
        failures.append("Phase A parent is no longer the frozen Gate A artifact")
    if freeze.get("primary_tasks") not in (["P1_x", "P2_parallel"], ["P1_x", "P2_parallel", "P3_curved"]):
        failures.append("primary task freeze invalid")
    for method, tasks in freeze.get("methods", {}).items():
        for task, values in tasks.items():
            if not math.isfinite(values.get("threshold", math.inf)) or values.get("train_fpr", 1.0) > 0.05:
                failures.append(f"invalid train threshold/FPR: {method}/{task}")
    with (PHASE / "raw_scores.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or {row["split"] for row in rows} != {"validation"}:
        failures.append("raw rows are not validation-only")
    expected_methods = {"A0_tangent_terminal", "A1_isotropic_shadowing", "A2_local_anisotropic_reset", "A3_cocycle_cg_lambda0", "A3_cocycle_cg_lambda035"}
    if {row["method"] for row in rows} != expected_methods or set(freeze.get("methods", {})) != expected_methods:
        failures.append("A0-A3 controls incomplete")
    valid_statuses = {"SHADOWING_ONLY_NO_ANOSOV_GAIN", "LOCAL_ANISOTROPY_GAIN_COCYCLE_REDUNDANT", "ANOSOV_COCYCLE_CAUSAL_ORACLE_PASS", "ANOSOV_COCYCLE_UNSAFE", "ANOSOV_COCYCLE_REDUNDANT_AT_ORACLE", "ANOSOV_HYPERBOLICITY_INERT"}
    if metrics.get("status") not in valid_statuses or state.get("status") != metrics.get("status"):
        failures.append("research status invalid")
    if metrics.get("gate_pass") is not (metrics.get("status") == "ANOSOV_COCYCLE_CAUSAL_ORACLE_PASS"):
        failures.append("gate-pass/status mismatch")
    gates = metrics.get("gates", {})
    if metrics.get("gate_pass") is not all(gates.values()):
        failures.append("gate pass does not equal all frozen causal gates")
    expected_access = {"synthetic_train": "calibration", "synthetic_validation": "causal gate", "synthetic_confirm": False, "synthetic_test": False, "phase_b": False, "cracks": False, "expert": False}
    if access != expected_access:
        failures.append("data access lock drift")
    if any(not math.isfinite(value.get("threshold", math.inf)) for value in leakage.values()):
        failures.append("non-finite leakage-control threshold")
    for source in (protocol, metrics, state):
        if source.get("training_performed") is not False:
            failures.append("training occurred")
        for key in ("confirm_opened", "test_opened", "cracks_data_accessed", "expert_data_accessed"):
            if source.get(key) is not False:
                failures.append(f"downstream lock violated: {key}")
    if protocol.get("phase_b_opened") is not False or state.get("phase_b_opened") is not False:
        failures.append("Phase B was opened")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "research_status": metrics.get("status") if not failures else "INVALID_ANZA_S_A2_EVIDENCE",
        "failures": failures, "protocol_sha256": stored_hash,
        "causal_gate_pass": bool(metrics.get("gate_pass")) and not failures,
        "training_performed": False, "phase_b_opened": False, "confirm_opened": False,
        "test_opened": False, "cracks_data_accessed": False, "expert_data_accessed": False,
        "next_action": ("Freeze a separate Phase B protocol" if metrics.get("gate_pass") and not failures else "STOP; do not train ANZA-S"),
    }
    (PHASE / "validator.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    result = validate(); print(json.dumps(result, indent=2, sort_keys=True)); raise SystemExit(0 if result["status"] == "PASS" else 1)
