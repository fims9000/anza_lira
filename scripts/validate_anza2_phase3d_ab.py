#!/usr/bin/env python3
"""Fail-closed validator for bounded ANZA-2 Phase 3D-A/B."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "results" / "anza2" / "phase3d_ab"
PARENT = ROOT / "results" / "anza2" / "phase3c_b_rc1"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate() -> dict[str, Any]:
    required = (
        "protocol.json", "protocol_hash.txt", "parent_evidence.json", "code_state.json",
        "data_access_log.json", "split_manifest.json", "case_manifest.csv",
        "PHASE3D_CASE_MANIFEST.csv", "visible_latent_target_audit.json",
        "CONTEXT_SUFFICIENCY.json", "strata_curriculum_manifest.json",
        "threshold_freeze.json", "metrics.json", "mode_state_paths.csv",
        "per_candidate.csv", "per_gap.csv", "per_case.csv", "operating_curve.csv",
        "bootstrap.json", "TASK_STATE.json", "EVIDENCE.json", "REPORT.md",
        "PHASE3D_AB_REPORT.md",
    )
    failures = [name for name in required if not (PHASE / name).is_file() or (PHASE / name).stat().st_size == 0]
    if failures:
        return {"status": "FAIL", "research_status": "INVALID_PHASE3D_AB_ARTIFACTS", "failures": failures}

    protocol = _load(PHASE / "protocol.json")
    metrics = _load(PHASE / "metrics.json")
    manifest = _load(PHASE / "split_manifest.json")
    target_audit = _load(PHASE / "visible_latent_target_audit.json")
    access = _load(PHASE / "data_access_log.json")
    threshold = _load(PHASE / "threshold_freeze.json")
    state = _load(PHASE / "TASK_STATE.json")
    stored_hash = (PHASE / "protocol_hash.txt").read_text().strip()

    if stored_hash != _canonical_hash(protocol):
        failures.append("protocol hash mismatch")
    if protocol.get("version") != "ANZA2_PHASE3D_CONTEXT_MODE_STATE_V1":
        failures.append("protocol version drift")
    if protocol.get("parent_protocol_sha256") != _digest(PARENT / "protocol.json"):
        failures.append("RC1 parent protocol changed")
    if protocol.get("parent_validator_sha256") != _digest(PARENT / "validator.json"):
        failures.append("RC1 parent validator changed")
    if _load(PARENT / "validator.json").get("research_status") != "STOP_RC1_MEMBERSHIP_REPAIR_FAILED":
        failures.append("RC1 negative parent status changed")

    if any(manifest.get("splits", {}).get(split, {}).get("count") != 512 for split in ("train", "validation", "confirm")):
        failures.append("manifest is not exactly 512 samples per split")
    if any(value != 0 for value in manifest.get("seed_overlap", {}).values()):
        failures.append("synthetic split seed overlap")
    if manifest.get("all_mandatory_strata_present") is not True:
        failures.append("a mandatory stratum is absent")
    if any(count <= 0 for counts in manifest.get("mandatory_strata_counts", {}).values() for count in counts.values()):
        failures.append("zero-count mandatory stratum")
    with (PHASE / "case_manifest.csv").open(newline="") as handle:
        case_rows = list(csv.DictReader(handle))
    if len(case_rows) != 1536:
        failures.append("case manifest row count drift")
    if target_audit.get("pass") is not True:
        failures.append("visible/latent target audit failed")
    for values in target_audit.get("splits", {}).values():
        if values.get("visible_latent_overlap_axes") != 0 or values.get("all_privileged_gap_local_supervision_removed") is not True:
            failures.append("privileged latent direction leaked into local targets")

    if threshold.get("source") != "all eligible train[0:512] oracle pairs":
        failures.append("threshold source drift")
    if threshold.get("maximum_fpr") != 0.05 or threshold.get("validation_used") is not False or threshold.get("confirm_used") is not False:
        failures.append("threshold lock violation")
    with (PHASE / "mode_state_paths.csv").open(newline="") as handle:
        path_rows = list(csv.DictReader(handle))
    if not path_rows or {row["split"] for row in path_rows} != {"validation"}:
        failures.append("oracle evaluation rows are not validation-only")
    if {row["method"] for row in path_rows} != {"G0_scalar", "G1_mode_state"}:
        failures.append("G0/G1 comparison rows missing")

    expected_checks = {
        "positive_noninferiority", "x_wrong_turn_relative_reduction_at_least_50pct",
        "parallel_false_bridge_noninferiority", "negative_gap_false_bridge_noninferiority",
        "curved_continuation_noninferiority",
    }
    if set(metrics.get("gate_checks", {})) != expected_checks:
        failures.append("oracle gate schema drift")
    computed_pass = all(metrics.get("gate_checks", {}).values())
    expected_research = "PHASE3D_ORACLE_MODE_STATE_PASS" if computed_pass else "FINAL_STOP_MODE_STATE_ORACLE_NO_VALUE"
    if metrics.get("gate_pass") is not computed_pass or metrics.get("status") != expected_research:
        failures.append("oracle status does not follow frozen gate")
    if state.get("phase3d_c_authorized") is not computed_pass:
        failures.append("Phase 3D-C authorization does not follow oracle gate")

    lock_sources = (protocol, metrics, state)
    for source in lock_sources:
        for key in ("training_performed", "cracks_data_accessed", "expert_data_accessed"):
            if source.get(key) is not False:
                failures.append(f"lock violated: {key}")
    if protocol.get("confirm_evaluation_opened") is not False or metrics.get("confirm_evaluation_opened") is not False or state.get("confirm_evaluation_opened") is not False:
        failures.append("confirm evaluation opened")
    if access != {
        "cracks": False, "expert": False, "synthetic_confirm_manifest_metadata_only": True,
        "synthetic_confirm_scored": False, "synthetic_train_manifest": True,
        "synthetic_train_scored_for_thresholds": True, "synthetic_validation_manifest": True,
        "synthetic_validation_scored_for_gate": True,
    }:
        failures.append("data access log drift")

    result = {
        "status": "PASS" if not failures else "FAIL",
        "research_status": expected_research if not failures else "INVALID_PHASE3D_AB_EVIDENCE",
        "failures": failures,
        "protocol_sha256": stored_hash,
        "phase3d_a_pass": metrics.get("phase3d_a_pass"),
        "oracle_gate_pass": computed_pass,
        "phase3d_c_authorized": computed_pass and not failures,
        "training_performed": False,
        "confirm_evaluation_opened": False,
        "cracks_data_accessed": False,
        "expert_data_accessed": False,
        "next_action": (
            "Freeze Phase 3D-C curriculum/training protocol before any training"
            if computed_pass and not failures else
            "STOP ANZA-2 mode-state development; no Phase 3D-C/confirm/CRACKS/expert"
        ),
    }
    (PHASE / "validator.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    value = validate(); print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["status"] == "PASS" else 1)

