#!/usr/bin/env python3
"""Print a compact deterministic gate for the current GeoCrack phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.geocrack import read_split_csv, sha256_file
from scripts.check_geocrack_split import assert_no_source_leakage, load_sources
from scripts.geocrack_study import RUN_MATRIX, SPLIT_DIR, STUDY_ROOT, split_bundle_hash
from scripts.validate_geocrack_study import collect_failures


def _line(name: str, passed: bool) -> tuple[str, bool]:
    return f"{name:<22} {'PASS' if passed else 'FAIL'}", passed


def dataset_checks() -> list[tuple[str, bool]]:
    paths = {split: SPLIT_DIR / f"geocrack_small_v1_{split}.csv" for split in ("train", "val", "test")}
    split_files = all(path.is_file() for path in paths.values())
    rows_valid = False
    leakage = False
    manifest_valid = False
    normalization = (SPLIT_DIR / "train_normalization.json").is_file()
    if split_files:
        try:
            rows_valid = all(bool(read_split_csv(path)) for path in paths.values())
        except (OSError, ValueError):
            rows_valid = False
        try:
            assert_no_source_leakage(load_sources(paths["train"]), load_sources(paths["val"]), load_sources(paths["test"]))
            leakage = True
        except ValueError:
            leakage = False
    manifest_path = SPLIT_DIR / "geocrack_small_v1_manifest.json"
    if manifest_path.is_file() and paths["test"].is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_valid = manifest.get("frozen_test_csv_sha256") == sha256_file(paths["test"])
    return [
        _line("SPLIT FILES", split_files),
        _line("PAIR ROWS", rows_valid),
        _line("SOURCE LEAKAGE", leakage),
        _line("TRAIN NORMALIZATION", normalization),
        _line("FROZEN TEST HASH", manifest_valid),
    ]


def smoke_checks() -> list[tuple[str, bool]]:
    report_path = STUDY_ROOT / "smoke_test_report.json"
    report_ok = report_path.is_file() and json.loads(report_path.read_text(encoding="utf-8")).get("status") == "PASS"
    runs = []
    for path in (STUDY_ROOT / "smoke").glob("*/run_metadata.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("status") == "COMPLETE":
            runs.append(metadata.get("model"))
    return [
        _line("BASELINE SMOKE", "baseline" in runs),
        _line("AZ SMOKE", "az_thesis" in runs),
        _line("VERTICAL REPORT", report_ok),
        _line("SMOKE FIGURE", (STUDY_ROOT / "smoke_vertical_slice.png").is_file()),
    ]


def training_checks() -> list[tuple[str, bool]]:
    found = set()
    hashes = True
    try:
        current_split = split_bundle_hash()
    except (OSError, ValueError):
        current_split = None
        hashes = False
    for path in (STUDY_ROOT / "runs").glob("*/run_metadata.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("status") == "COMPLETE":
            found.add((metadata.get("model"), int(metadata.get("seed", -1))))
            hashes = hashes and bool(metadata.get("config_hash")) and metadata.get("split_hash") == current_split
    required = set(RUN_MATRIX)
    return [
        _line("REQUIRED 9 RUNS", required <= found),
        _line("CONFIG/SPLIT HASHES", hashes and required <= found),
        _line("BASELINE 3 SEEDS", all(("baseline", seed) in found for seed in (41, 42, 43))),
        _line("AZ 3 SEEDS", all(("az_thesis", seed) in found for seed in (41, 42, 43))),
        _line("ABLATIONS", all(key in found for key in (("az_no_fuzzy", 42), ("az_no_aniso", 42), ("attention_unet", 42)))),
    ]


def trace_checks() -> list[tuple[str, bool]]:
    runs = list((STUDY_ROOT / "runs").glob("*/evaluation_summary.json"))
    geojson = list((STUDY_ROOT / "traces").glob("**/*.geojson"))
    valid_geojson = bool(geojson)
    for path in geojson:
        payload = json.loads(path.read_text(encoding="utf-8"))
        valid_geojson = valid_geojson and payload.get("type") == "FeatureCollection"
    return [
        _line("RUN EVALUATIONS", len(runs) >= 9),
        _line("TRACE GEOJSON", valid_geojson),
        _line("TRACE TABLE", (STUDY_ROOT / "tables" / "trace_metrics.csv").is_file()),
        _line("BOOTSTRAP TABLE", (STUDY_ROOT / "tables" / "bootstrap_comparison.csv").is_file()),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("dataset", "smoke", "training", "traces", "final"))
    args = parser.parse_args()
    if args.phase == "dataset":
        checks = dataset_checks()
    elif args.phase == "smoke":
        checks = smoke_checks()
    elif args.phase == "training":
        checks = training_checks()
    elif args.phase == "traces":
        checks = trace_checks()
    else:
        failures = collect_failures()
        checks = [_line("FINAL VALIDATOR", not failures)]
        for failure in failures[:18]:
            print(f"- {failure}")
    for line, _passed in checks:
        print(line)
    passed = all(passed for _line_text, passed in checks)
    print(f"\nPHASE STATUS: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
