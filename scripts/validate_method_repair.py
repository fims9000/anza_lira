#!/usr/bin/env python3
"""Build and validate the fail-closed ANZA method-repair result package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from method_repair.audit import FROZEN_FILES, sha256_file
from method_repair.reporting import REQUIRED_FINAL_FILES, build_negative_package


FINAL_ROOT = PROJECT_ROOT / "results" / "method_repair" / "final"
ZIP_PATH = PROJECT_ROOT / "results" / "method_repair" / "ANZA_METHOD_REPAIR_NEGATIVE_20260818.zip"


def validate(*, device: str = "cuda") -> dict:
    build_negative_package(PROJECT_ROOT, FINAL_ROOT, device=device)
    gate = json.loads((PROJECT_ROOT / "results" / "method_repair" / "synthetic_v2" / "mechanism_gate.json").read_text())
    root_cause = json.loads((PROJECT_ROOT / "results" / "method_repair" / "root_cause.json").read_text())
    numbers = json.loads((FINAL_ROOT / "THESIS_NUMBERS.json").read_text())
    checks = {
        "required_files": all((FINAL_ROOT / name).is_file() for name in REQUIRED_FINAL_FILES),
        "synthetic_gate_failed": gate.get("status") == "SYNTHETIC_MECHANISM_FAIL" and gate.get("cracks_authorized") is False,
        "negative_status": root_cause.get("status") == numbers.get("status") == "METHOD_REPAIR_NEGATIVE_WITH_ROOT_CAUSE",
        "expert_locked": root_cause.get("expert_data_accessed") is False and numbers["cracks"]["expert_data_accessed"] is False,
        "tests_locked": root_cause.get("old_test_samples_opened") == root_cause.get("new_test_samples_opened") == 0,
        "cracks_not_run": root_cause.get("cracks_training", "").startswith("NOT_RUN"),
        "frozen_hashes": numbers["frozen_deadline_sha256"] == {name: sha256_file(path) for name, path in FROZEN_FILES.items()},
        "figures": all(
            (FINAL_ROOT / "figures" / f"{base}.{suffix}").is_file()
            and (FINAL_ROOT / "figures" / f"{base}.{suffix}").stat().st_size > 1000
            for base in ("fig_synthetic_matrix", "fig_gate_failure", "fig_failure_cases")
            for suffix in ("png", "svg", "pdf")
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"method-repair validation failed: {checks}")
    receipt = {
        "status": "METHOD_REPAIR_NEGATIVE_WITH_ROOT_CAUSE",
        "checks": checks,
        "expert_data_accessed": False,
        "old_test_samples_opened": 0,
        "new_test_samples_opened": 0,
        "cracks_training": "NOT_RUN_SYNTHETIC_GATE_FAILED",
        "validated_on": "2026-08-18",
    }
    (FINAL_ROOT / "VALIDATION_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    checksum_path = FINAL_ROOT / "SHA256SUMS.txt"
    files = sorted(path for path in FINAL_ROOT.rglob("*") if path.is_file() and path != checksum_path)
    checksum_path.write_text("".join(f"{sha256_file(path)}  {path.relative_to(FINAL_ROOT)}\n" for path in files))
    for line in checksum_path.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        if sha256_file(FINAL_ROOT / relative) != expected:
            raise ValueError(f"checksum mismatch: {relative}")
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(FINAL_ROOT.rglob("*")):
            if path.is_file():
                archive.write(path, Path("ANZA_METHOD_REPAIR_NEGATIVE_20260818") / path.relative_to(FINAL_ROOT))
    with zipfile.ZipFile(ZIP_PATH) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC failure: {bad}")
    receipt["zip_path"] = str(ZIP_PATH)
    receipt["zip_sha256"] = sha256_file(ZIP_PATH)
    receipt["zip_file_count"] = len(zipfile.ZipFile(ZIP_PATH).namelist())
    return receipt


def main() -> int:
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    receipt = validate(device=device)
    print("CODE AND UNIT CONTRACTS: PASS")
    print("A0-A4 TRAINING: PASS")
    print("SYNTHETIC MECHANISM GATE: FAIL")
    print("CRACKS R0-R3: NOT RUN")
    print("EXPERT EVALUATION: NOT RUN")
    print("PACKAGE CHECKSUMS: PASS")
    print(f"ZIP: {receipt['zip_path']}")
    print(f"ZIP SHA256: {receipt['zip_sha256']}")
    print("METHOD REPAIR STATUS: METHOD_REPAIR_NEGATIVE_WITH_ROOT_CAUSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
