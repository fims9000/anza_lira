#!/usr/bin/env python3
"""Generate final practical reports, validate, checksum, and package evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_completion.final_practical_reporting import FINAL_ROOT, STUDY_ROOT, build_closeout
from scripts.validate_final_practical_cycle import validate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    build_closeout()
    receipt = validate()
    if receipt["status"] != "PASS":
        raise RuntimeError(json.dumps(receipt, sort_keys=True))
    (FINAL_ROOT / "VALIDATION_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    checksum_path = FINAL_ROOT / "SHA256SUMS.txt"
    package_path = FINAL_ROOT / "PACKAGE.json"
    files = sorted(
        path for path in FINAL_ROOT.rglob("*")
        if path.is_file() and path not in {checksum_path, package_path}
    )
    checksum_path.write_text("".join(f"{_sha(path)}  {path.relative_to(FINAL_ROOT).as_posix()}\n" for path in files))
    include = [
        *sorted(path for path in FINAL_ROOT.rglob("*") if path != package_path),
        STUDY_ROOT / "path_calibration" / "calibration_freeze.json",
        STUDY_ROOT / "path_v5_test" / "test_result.json",
        STUDY_ROOT / "path_v5_test" / "test_scores.csv",
        STUDY_ROOT / "realistic_synthetic" / "development_result.json",
        STUDY_ROOT / "realistic_synthetic" / "development_cells.csv",
        STUDY_ROOT / "cracks_t1" / "audit.json",
        STUDY_ROOT / "cracks_t1" / "analysis" / "result.json",
        STUDY_ROOT / "cracks_t1" / "analysis" / "paired_comparisons.csv",
        STUDY_ROOT / "cracks_t1" / "analysis" / "raw_per_section.csv",
        STUDY_ROOT / "cracks_pairs" / "STRICT_CAPACITY_ROOT_CAUSE.json",
        STUDY_ROOT / "cracks_pairs" / "manifest.json",
        STUDY_ROOT / "cracks_pairs" / "scores.csv",
        STUDY_ROOT / "cracks_pairs" / "result.json",
        PROJECT_ROOT / "cracks_experiment" / "partial_labels.py",
        PROJECT_ROOT / "cracks_experiment" / "partial_label_training.py",
        PROJECT_ROOT / "cracks_experiment" / "partial_label_evaluation.py",
        PROJECT_ROOT / "path_completion" / "cracks_pairs.py",
        PROJECT_ROOT / "path_completion" / "cracks_pair_training.py",
        PROJECT_ROOT / "path_completion" / "final_practical_reporting.py",
        PROJECT_ROOT / "scripts" / "run_cracks_partial_label_study.py",
        PROJECT_ROOT / "scripts" / "run_cracks_pair_classifier.py",
        PROJECT_ROOT / "scripts" / "validate_final_practical_cycle.py",
        PROJECT_ROOT / "tests" / "test_cracks_partial_labels.py",
        PROJECT_ROOT / "tests" / "test_cracks_partial_label_training.py",
        PROJECT_ROOT / "tests" / "test_cracks_pair_classifier.py",
        PROJECT_ROOT / "tests" / "test_final_practical_reporting.py",
    ]
    include = sorted({path for path in include if path.is_file()})
    forbidden = {"checkpoint.pt", "checkpoint-last.pt", "pairs.npz"}
    if any(path.name in forbidden for path in include):
        raise AssertionError("Closeout package attempted to include a checkpoint or generated pair tensor")
    archive = STUDY_ROOT / "ANZA_FINAL_PRACTICAL_CLOSEOUT_20260818.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in include:
            handle.write(path, path.relative_to(PROJECT_ROOT).as_posix())
    with zipfile.ZipFile(archive) as handle:
        bad = handle.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC failed at {bad}")
    package = {
        "status": "PASS",
        "path": str(archive),
        "sha256": _sha(archive),
        "size_bytes": archive.stat().st_size,
        "members": len(include),
        "crc": "PASS",
        "checkpoints_included": False,
        "pairs_npz_included": False,
    }
    (FINAL_ROOT / "PACKAGE.json").write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"validation": receipt, "package": package}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
