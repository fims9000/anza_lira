#!/usr/bin/env python3
"""Fail-closed validator for the structural-affinity repair evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from affinity_repair.matrix import affinity_matrix, affinity_protocol_hash


RESULT_ROOT = ROOT / "results" / "affinity_repair"


def _pass(name: str, condition: bool) -> None:
    print(f"{name:<28} {'PASS' if condition else 'FAIL'}")
    if not condition:
        raise ValueError(name)


def validate() -> dict:
    protocol = json.loads((RESULT_ROOT / "protocol.json").read_text())
    benchmark = json.loads((RESULT_ROOT / "benchmark_v4_config.json").read_text())
    _pass("PROTOCOL", protocol["protocol_hash"] == affinity_protocol_hash())
    _pass("V4 TEST LOCK", benchmark["test_status"] == "LOCKED_UNOPENED")
    _pass("NO C4", [spec.candidate_id for spec in affinity_matrix()] == ["C0", "C1", "C2", "C3"])
    validations = {}
    for spec in affinity_matrix():
        status_path = RESULT_ROOT / "development" / f"{spec.candidate_id}-{spec.run_hash}" / "status.json"
        status = json.loads(status_path.read_text())
        _pass(f"TRAIN {spec.candidate_id}", status["status"] == "COMPLETE" and status["checkpoint_reload"] == "PASS")
        for field, expected in (("expert_data_accessed", False), ("v4_test_samples_opened", 0), ("cracks_samples_opened", 0)):
            _pass(f"{spec.candidate_id} {field}", status[field] == expected)
        validation = json.loads((RESULT_ROOT / "validation" / f"{spec.candidate_id}-{spec.run_hash}.json").read_text())
        _pass(f"EVAL {spec.candidate_id}", validation["status"] == "COMPLETE" and validation["sample_count"] == 512)
        validations[spec.candidate_id] = validation
    gate = json.loads((RESULT_ROOT / "mechanism_gate.json").read_text())
    _pass("GATE STATUS", gate["status"] in {"AFFINITY_MECHANISM_PASS", "AFFINITY_MECHANISM_FAIL"})
    final = RESULT_ROOT / "final"
    root_cause = json.loads((final / "ROOT_CAUSE.json").read_text())
    if gate["status"] == "AFFINITY_MECHANISM_FAIL":
        _pass("NEGATIVE STATUS", root_cause["final_status"] == "AFFINITY_REPAIR_NEGATIVE_WITH_ROOT_CAUSE")
        _pass("CONFIRM LOCK", gate["confirm_authorized"] is False)
        final_status = "AFFINITY_REPAIR_NEGATIVE_WITH_ROOT_CAUSE"
    else:
        # Development success is not full success. Independent three-seed
        # confirmation is mandatory and deliberately cannot be inferred.
        confirm = RESULT_ROOT / "confirm" / "CONFIRM_RECEIPT.json"
        _pass("CONFIRM RECEIPT", confirm.exists())
        receipt = json.loads(confirm.read_text())
        _pass("CONFIRM PASS", receipt.get("status") == "PASS" and receipt.get("seeds") == [41, 42, 43])
        final_status = "AFFINITY_REPAIR_SUCCESS"
    checksums = (final / "SHA256SUMS.txt").read_text().splitlines()
    for line in checksums:
        digest, relative = line.split("  ", 1)
        _pass(f"SHA {relative}", hashlib.sha256((final / relative).read_bytes()).hexdigest() == digest)
    suffix = "REPAIR_NEGATIVE" if root_cause["final_status"] == "AFFINITY_REPAIR_NEGATIVE_WITH_ROOT_CAUSE" else "REPAIR_SUCCESS"
    packages = list(RESULT_ROOT.glob(f"ANZA_STRUCTURAL_AFFINITY_{suffix}_20260818.zip"))
    _pass("ZIP EXISTS", len(packages) == 1)
    with zipfile.ZipFile(packages[0]) as archive:
        _pass("ZIP CRC", archive.testzip() is None)
    print(f"\nAFFINITY REPAIR STATUS: {final_status}")
    return {"status": final_status, "gate": gate["status"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    validate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
