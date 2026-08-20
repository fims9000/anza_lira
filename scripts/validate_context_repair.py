#!/usr/bin/env python3
"""Fail-closed validator for the bounded context-repair cycle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method_repair.context_matrix import context_matrix, context_protocol_hash
from synthetic.crossing_trace_bench_v3 import benchmark_v3_config


def _print(name: str, passed: bool, detail: str = "") -> bool:
    print(f"{name:<26} {'PASS' if passed else 'FAIL'} {detail}".rstrip())
    return passed


def main() -> int:
    result_root = ROOT / "results" / "context_repair"
    passed = True
    protocol = json.loads((result_root / "protocol.json").read_text())
    passed &= _print("PROTOCOL FREEZE", protocol.get("protocol_hash") == context_protocol_hash())
    benchmark = json.loads((result_root / "benchmark_v3_config.json").read_text())
    passed &= _print("BENCHMARK V3", benchmark.get("sha256") == benchmark_v3_config()["sha256"])
    passed &= _print("TEST V3 LOCK", benchmark.get("test_status") == "LOCKED_UNOPENED")

    for spec in context_matrix():
        run = result_root / "development" / f"{spec.candidate_id}-{spec.run_hash}"
        status_path = run / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        valid = (
            status.get("status") == "COMPLETE"
            and status.get("run_hash") == spec.run_hash
            and status.get("epoch") == 25
            and status.get("checkpoint_reload") == "PASS"
            and status.get("expert_data_accessed") is False
            and status.get("legacy_test_samples_opened") == 0
            and status.get("v3_test_samples_opened") == 0
            and status.get("cracks_samples_opened") == 0
        )
        passed &= _print(f"TRAIN {spec.candidate_id}", valid, "25/25" if valid else "")
        summary_path = result_root / "validation" / f"{spec.candidate_id}-{spec.run_hash}.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        csv_path = Path(summary.get("rows_csv", "missing"))
        row_count = 0
        if csv_path.exists():
            with csv_path.open(newline="") as handle:
                row_count = sum(1 for _ in csv.DictReader(handle))
        valid_evaluation = (
            summary.get("status") == "COMPLETE"
            and summary.get("run_hash") == spec.run_hash
            and summary.get("sample_count") == 512
            and row_count == 512
            and summary.get("expert_data_accessed") is False
            and summary.get("legacy_test_samples_opened") == 0
            and summary.get("v3_test_samples_opened") == 0
            and summary.get("cracks_samples_opened") == 0
        )
        passed &= _print(f"EVALUATE {spec.candidate_id}", valid_evaluation, f"rows={row_count}")

    gate_path = result_root / "mechanism_gate.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    gate_status = gate.get("status")
    passed &= _print("FROZEN MECHANISM GATE", gate_status in {"CONTEXT_MECHANISM_PASS", "CONTEXT_MECHANISM_FAIL"}, str(gate_status))
    passed &= _print("CRACKS LOCK", gate.get("cracks_authorized") is False)
    passed &= _print("EXPERT LOCK", gate.get("expert_data_accessed") is False)

    if gate_status == "CONTEXT_MECHANISM_FAIL":
        expected_status = "CONTEXT_REPAIR_NEGATIVE_WITH_ROOT_CAUSE"
        final = result_root / "final"
        numbers_path = final / "THESIS_NUMBERS.json"
        numbers = json.loads(numbers_path.read_text()) if numbers_path.exists() else {}
        passed &= _print("ROOT CAUSE", (final / "ROOT_CAUSE.json").exists())
        passed &= _print("FINAL REPORT", (final / "FINAL_REPORT.md").exists())
        passed &= _print("TERMINAL STATUS", numbers.get("status") == expected_status, str(numbers.get("status")))
        archive = result_root / "ANZA_CONTEXT_REPAIR_20260818.zip"
        checksum_path = result_root / f"{archive.name}.sha256"
        archive_ok = False
        if archive.exists() and checksum_path.exists():
            expected = checksum_path.read_text().split()[0]
            actual = hashlib.sha256(archive.read_bytes()).hexdigest()
            with zipfile.ZipFile(archive) as handle:
                archive_ok = expected == actual and handle.testzip() is None
        passed &= _print("PACKAGE", archive_ok)
        terminal = expected_status
    elif gate_status == "CONTEXT_MECHANISM_PASS":
        confirmation = result_root / "confirm" / "CONFIRMATION_RECEIPT.json"
        confirm = json.loads(confirmation.read_text()) if confirmation.exists() else {}
        passed &= _print("CONFIRM V3", confirm.get("status") == "PASS")
        terminal = "CONTEXT_REPAIR_SUCCESS" if confirm.get("status") == "PASS" else "INCOMPLETE_CONFIRM_REQUIRED"
    else:
        terminal = "INCOMPLETE"
    print(f"CONTEXT REPAIR STATUS: {terminal}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
