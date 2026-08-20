#!/usr/bin/env python3
"""Fail-closed validator for the ANZA-S zero-training oracle."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "results" / "anza_s" / "oracle"
PARENT = ROOT / "results" / "anza2" / "phase3d_ab"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate() -> dict[str, Any]:
    required = (
        "protocol.json", "protocol_hash.txt", "parent_evidence.json", "code_state.json",
        "data_access_log.json", "threshold_freeze.json", "raw_scores.csv", "per_case.csv",
        "trajectory_points.csv", "shadowing_scores.csv", "metrics.json", "operating_curve.csv",
        "figure_manifest.json", "TASK_STATE.json", "EVIDENCE.json", "ANZA_S_ORACLE_REPORT.md",
    )
    failures = [name for name in required if not (PHASE / name).is_file() or (PHASE / name).stat().st_size == 0]
    for name in (
        "F1_x_scalar_state_cocycle", "F2_x_shadowing_correct_wrong",
        "F3_parallel_separation", "F4_curved_gap_shadowing",
    ):
        for suffix in ("png", "svg"):
            path = PHASE / "figures" / f"{name}.{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                failures.append(f"missing figure: {path.name}")
    if failures:
        return {"status": "FAIL", "research_status": "INVALID_ANZA_S_ORACLE_ARTIFACTS", "failures": failures}

    protocol = _load(PHASE / "protocol.json"); metrics = _load(PHASE / "metrics.json")
    thresholds = _load(PHASE / "threshold_freeze.json"); access = _load(PHASE / "data_access_log.json")
    state = _load(PHASE / "TASK_STATE.json"); stored_hash = (PHASE / "protocol_hash.txt").read_text().strip()
    if protocol.get("version") != "ANZA_S_ANOSOV_COCYCLE_SHADOWING_ORACLE_V1":
        failures.append("protocol version drift")
    if stored_hash != _canonical_hash(protocol):
        failures.append("protocol hash mismatch")
    if protocol.get("parent_protocol_sha256") != _digest(PARENT / "protocol.json") or protocol.get("parent_validator_sha256") != _digest(PARENT / "validator.json"):
        failures.append("frozen ANZA-2 parent changed")
    if _load(PARENT / "validator.json").get("research_status") != "FINAL_STOP_MODE_STATE_ORACLE_NO_VALUE":
        failures.append("ANZA-2 negative history was not preserved")

    if set(thresholds) != {"O0_scalar_anza", "O1_mode_state", "O2_tangent_streamline", "O3_cocycle_rollout", "O4_cocycle_shadowing"}:
        failures.append("O0-O4 thresholds missing")
    for method, value in thresholds.items():
        if not math.isfinite(value.get("threshold", math.inf)):
            failures.append(f"non-finite threshold: {method}")
        if any(fpr > 0.05 for fpr in value.get("train_fpr", {}).values()):
            failures.append(f"train task FPR exceeded: {method}")
    with (PHASE / "raw_scores.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2680 or {row["split"] for row in rows} != {"validation"}:
        failures.append("raw validation O0-O4 row contract failed")
    if len({(row["index"], row["task"], row["pair_id"]) for row in rows}) != 536:
        failures.append("candidate count is not 536")
    if {row["method"] for row in rows} != set(thresholds):
        failures.append("method rows incomplete")

    checks = metrics.get("baseline_gate_checks", {})
    if set(checks) != {"O0_scalar_anza", "O2_tangent_streamline"}:
        failures.append("formal baseline gate schema drift")
    computed_gate = all(all(value.values()) for value in checks.values())
    expected_status = "ANZA_S_ORACLE_GATE_A_PASS" if computed_gate else "FINAL_STOP_ANOSOV_DYNAMICAL_ARCHITECTURE"
    if metrics.get("gate_pass") is not computed_gate or metrics.get("status") != expected_status:
        failures.append("status does not follow formal frozen gate")
    causal = metrics.get("causal_diagnostics", {})
    if causal.get("generic_tangent_plus_shadowing_control_included") is not False:
        failures.append("post-hoc generic shadowing control was silently added")
    if causal.get("cocycle_rollout_incremental_effect_established") is not False:
        failures.append("O2/O3 equality claim drift")

    if access != {"synthetic_train": "calibration", "synthetic_validation": "gate", "synthetic_confirm": False, "synthetic_test": False, "cracks": False, "expert": False}:
        failures.append("data access lock drift")
    for source in (protocol, metrics, state):
        if source.get("training_performed") is not False:
            failures.append("training was performed")
    for source in (protocol, metrics, state):
        if source.get("confirm_opened") is not False or source.get("cracks_data_accessed") is not False or source.get("expert_data_accessed") is not False:
            failures.append("downstream data lock violated")

    result = {
        "status": "PASS" if not failures else "FAIL",
        "research_status": expected_status if not failures else "INVALID_ANZA_S_ORACLE_EVIDENCE",
        "failures": failures, "protocol_sha256": stored_hash,
        "formal_gate_a_pass": computed_gate,
        "cocycle_incremental_effect_established": False,
        "phase_b_eligible_under_formal_packet": computed_gate and not failures,
        "training_performed": False, "confirm_opened": False,
        "cracks_data_accessed": False, "expert_data_accessed": False,
        "next_action": "Stop now; freeze a separate Phase-B protocol before any field learning" if computed_gate and not failures else "FINAL STOP; do not train ANZA-S",
    }
    (PHASE / "validator.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    result = validate(); print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
