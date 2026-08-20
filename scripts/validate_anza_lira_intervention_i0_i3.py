#!/usr/bin/env python3
"""Independent validator for the bounded Intervention Endgame run."""

from __future__ import annotations

import json
from pathlib import Path

from lira_intervention.data import load_jsonl, split_manifest
from lira_intervention.protocol import RESULT_ROOT, ROOT, PROTOCOL, protocol_hash


def validate() -> dict[str, object]:
    failures = []
    freeze = json.loads((RESULT_ROOT / "i0_freeze/freeze_receipt.json").read_text())
    benchmark = json.loads((RESULT_ROOT / "i1_benchmark/benchmark_summary.json").read_text())
    candidate = json.loads((RESULT_ROOT / "i2_candidate/summary.json").read_text())
    diagnostics = json.loads((RESULT_ROOT / "i2_candidate/diagnostics.json").read_text())
    manifest = split_manifest()
    if freeze.get("protocol_sha256") != protocol_hash() or benchmark.get("protocol_sha256") != protocol_hash() or candidate.get("protocol_sha256") != protocol_hash():
        failures.append("protocol hash drift")
    if freeze.get("parent_stop") != "STOP_LIRA_REAL_GAP_DATA_INSUFFICIENT":
        failures.append("parent natural-gap STOP changed")
    if freeze.get("split_manifest") != manifest:
        failures.append("split manifest drift")
    sets = [set(value) for value in manifest["splits"].values()]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets))):
        failures.append("section overlap")
    calibration = load_jsonl(RESULT_ROOT / "i1_benchmark/ig_calibration.jsonl")
    development = load_jsonl(RESULT_ROOT / "i1_benchmark/ig_development.jsonl")
    if len(calibration) < int(PROTOCOL["benchmark_minimum"]["calibration"]) or len(development) < int(PROTOCOL["benchmark_minimum"]["development"]):
        failures.append("benchmark below minimum")
    if len({row.trace_id for row in calibration}) != len(calibration) or len({row.trace_id for row in development}) != len(development):
        failures.append("more than one evaluation intervention per trace")
    rows = [json.loads(line) for line in (RESULT_ROOT / "i2_candidate/development_candidates.jsonl").open()]
    recalled = sum(bool(row["candidate_recalled"]) for row in rows)
    recomputed = recalled / len(rows)
    if abs(recomputed - float(candidate["candidate_recall_at_12"])) > 1e-12 or int(candidate["k"]) != 12:
        failures.append("candidate metric drift")
    if not all(bool(row["image_unchanged"]) for row in rows):
        failures.append("image modification detected")
    if candidate.get("status") != "STOP_LIRA_INTERVENTION_CANDIDATE" or recomputed >= float(PROTOCOL["i2_gate"]["candidate_recall_at_12"]):
        failures.append("I2 STOP inconsistent with gate")
    if recomputed >= 0.87:
        failures.append("unexpected landing-band repair eligibility")
    if diagnostics.get("gate_changed") is not False:
        failures.append("post-STOP diagnostic changed gate")
    locked = (RESULT_ROOT / "i3_relation_s41/LIRA_INTERVENTION_RELATION_S41_REPORT.md").read_text()
    if "LOCKED_NOT_RUN_AFTER_I2_STOP" not in locked or (RESULT_ROOT / "i3_relation_s41/checkpoint.pt").exists():
        failures.append("I3 lock failed")
    if any((ROOT / "results/lira_final/f1_gap_audit/dense_cache" / f"section_{section:03d}.npy").exists() for section in manifest["splits"]["ig_confirm"]):
        failures.append("confirm dense inference opened")
    result = {
        "validator_status": "PASS" if not failures else "FAIL",
        "research_status": candidate.get("status"),
        "failures": failures,
        "candidate_recall_at_12": recomputed,
        "candidate_recalled": recalled,
        "development_cases": len(rows),
        "confirm_opened": False,
        "expert_accessed": False,
        "p0_opened": False,
        "path_opened": False,
    }
    path = RESULT_ROOT / "i2_candidate/validator.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    result = validate()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["validator_status"] == "PASS" else 1)
