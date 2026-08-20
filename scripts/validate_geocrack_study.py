#!/usr/bin/env python3
"""Claim-safe prepared and final validators for the GeoCrack study."""

from __future__ import annotations

import csv
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = PROJECT_ROOT / "results" / "geocrack_study"
SPLIT_DIR = PROJECT_ROOT / "data" / "geocrack" / "splits"


def required_run_keys() -> set[tuple[str, int]]:
    return {
        ("baseline", 41),
        ("baseline", 42),
        ("baseline", 43),
        ("az_thesis", 41),
        ("az_thesis", 42),
        ("az_thesis", 43),
        ("az_no_fuzzy", 42),
        ("az_no_aniso", 42),
        ("attention_unet", 42),
    }


def all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    return True


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_hash() -> str:
    digest_parts = []
    for split in ("train", "val", "test"):
        path = SPLIT_DIR / f"geocrack_small_v1_{split}.csv"
        digest_parts.append((path.name, _sha256(path)))
    import hashlib

    digest = hashlib.sha256()
    for name, value in digest_parts:
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def collect_failures() -> list[str]:
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require((PROJECT_ROOT / "AGENTS.md").is_file(), "AGENTS.md missing")
    rules = list((PROJECT_ROOT / ".cursor" / "rules").glob("*.mdc"))
    require(len(rules) >= 6, "required Cursor project rules missing")
    try:
        output = subprocess.run(
            [sys.executable, "scripts/list_agent_skills.py"],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        require(output.returncode == 0 and "geocrack-data" in output.stdout, "skill registry failed")
    except (OSError, subprocess.SubprocessError) as exc:
        failures.append(f"skill registry failed: {exc}")

    definitions = 0
    for path in list((PROJECT_ROOT / "datasets").glob("*.py")) + list((PROJECT_ROOT / "utils.py",)):
        definitions += len(re.findall(r"^class\s+GeoCrackDataset\b", path.read_text(encoding="utf-8"), re.MULTILINE))
    require(definitions == 1, f"expected one GeoCrackDataset implementation, found {definitions}")
    if shutil.which("rtk"):
        require((STUDY_ROOT / "rtk_before.txt").is_file(), "RTK is available but rtk_before.txt is missing")

    data_root = PROJECT_ROOT / "data" / "geocrack"
    download_manifest_path = data_root / "download_manifest.json"
    metadata_path = data_root / "dataverse_metadata.json"
    manual_manifest_path = data_root / "manual_import_manifest.json"
    require(download_manifest_path.is_file() or manual_manifest_path.is_file(), "official or manual import manifest missing")
    require(metadata_path.is_file() or manual_manifest_path.is_file(), "official metadata/import provenance missing")
    if download_manifest_path.is_file():
        download_manifest = _json(download_manifest_path)
        files = download_manifest.get("files", [])
        require(bool(files), "official Patched Data download manifest is empty")
        for item in files:
            archive = data_root / "archives" / str(item.get("filename", ""))
            require(archive.is_file(), f"downloaded official file missing: {archive.name}")
            if archive.is_file():
                require(int(item.get("size", -1)) == archive.stat().st_size, f"downloaded size mismatch: {archive.name}")
                require(item.get("sha256") == _sha256(archive), f"downloaded SHA-256 mismatch: {archive.name}")
    if manual_manifest_path.is_file():
        manual = _json(manual_manifest_path)
        validation = manual.get("validation", {})
        require(manual.get("network_requests") == 0, "manual import unexpectedly used network requests")
        require(validation.get("status") == "PASS", "manual import validation is not PASS")
        require(validation.get("pair_count") == 12158, "manual import pair count is not 12158")
        require(bool(validation.get("dataset_sha256")), "manual import dataset checksum missing")

    required_splits = [SPLIT_DIR / f"geocrack_small_v1_{split}.csv" for split in ("train", "val", "test")]
    required_splits += [
        SPLIT_DIR / "geocrack_small_v1_manifest.json",
        SPLIT_DIR / "train_normalization.json",
        SPLIT_DIR / "test_split.sha256",
        SPLIT_DIR / "site_mapping_audit.json",
    ]
    require(all(path.is_file() for path in required_splits), "dataset split/normalization artifacts missing")
    current_split_hash = None
    if all(path.is_file() for path in required_splits):
        manifest = _json(SPLIT_DIR / "geocrack_small_v1_manifest.json")
        require(manifest.get("dataset_pair_count") == 12158, "official GeoCrack pair count is not 12158")
        require(manifest.get("source_leakage") == 0, "source leakage is not zero")
        require(
            manifest.get("frozen_test_csv_sha256") == _sha256(SPLIT_DIR / "geocrack_small_v1_test.csv"),
            "frozen test split hash changed",
        )
        require(
            (SPLIT_DIR / "test_split.sha256").read_text(encoding="utf-8").strip()
            == _sha256(SPLIT_DIR / "geocrack_small_v1_test.csv"),
            "standalone test split contract changed",
        )
        source_sets = {
            split: {row["source_image_id"] for row in _csv(SPLIT_DIR / f"geocrack_small_v1_{split}.csv")}
            for split in ("train", "val", "test")
        }
        require(not (source_sets["train"] & source_sets["val"]), "TRAIN/VAL source leakage")
        require(not (source_sets["train"] & source_sets["test"]), "TRAIN/TEST source leakage")
        require(not (source_sets["val"] & source_sets["test"]), "VAL/TEST source leakage")
        require(len(source_sets["train"]) >= 6, "train split has fewer than 6 source images")
        require(len(source_sets["val"]) >= 3, "validation split has fewer than 3 source images")
        require(len(source_sets["test"]) >= 3, "test split has fewer than 3 source images")
        normalization = _json(SPLIT_DIR / "train_normalization.json")
        require(
            normalization.get("source_csv_sha256") == _sha256(SPLIT_DIR / "geocrack_small_v1_train.csv"),
            "normalization is not tied to the frozen train CSV",
        )
        current_split_hash = _split_hash()

    found_runs: dict[tuple[str, int], Path] = {}
    for metadata_path in sorted((STUDY_ROOT / "runs").glob("*/run_metadata.json")):
        metadata = _json(metadata_path)
        if metadata.get("status") != "COMPLETE":
            continue
        key = (str(metadata.get("model")), int(metadata.get("seed", -1)))
        if key in required_run_keys() and key not in found_runs:
            found_runs[key] = metadata_path.parent
            require(bool(metadata.get("config_hash")), f"{metadata_path.parent.name}: config hash missing")
            require(bool(metadata.get("commit_hash")), f"{metadata_path.parent.name}: commit hash missing")
            if current_split_hash:
                require(metadata.get("split_hash") == current_split_hash, f"{metadata_path.parent.name}: split hash mismatch")
            for filename in ("checkpoint_best.pt", "checkpoint_last.pt", "metrics.json", "evaluation_summary.json", "per_patch_metrics.csv"):
                require((metadata_path.parent / filename).is_file(), f"{metadata_path.parent.name}: {filename} missing")
            metrics_path = metadata_path.parent / "metrics.json"
            if metrics_path.is_file():
                require(all_finite(_json(metrics_path)), f"{metadata_path.parent.name}: non-finite metrics")
    missing_runs = required_run_keys() - set(found_runs)
    require(not missing_runs, f"required run matrix incomplete: {sorted(missing_runs)}")

    tables = STUDY_ROOT / "tables"
    table_names = ("summary_by_seed.csv", "summary_mean_std.csv", "bootstrap_comparison.csv", "trace_metrics.csv")
    require(all((tables / name).is_file() for name in table_names), "required tables missing")
    if (tables / "summary_by_seed.csv").is_file():
        rows = _csv(tables / "summary_by_seed.csv")
        keys = {(row["model"], int(row["seed"])) for row in rows}
        require(required_run_keys() <= keys, "summary_by_seed does not contain the required matrix")
    if (tables / "bootstrap_comparison.csv").is_file():
        rows = _csv(tables / "bootstrap_comparison.csv")
        require(bool(rows) and all(int(row["replicates"]) == 2000 for row in rows), "bootstrap is not 2000 replicates")

    figure_stems = (
        "fig_segmentation_median",
        "fig_error_median",
        "fig_geometry_traces",
        "fig_model_comparison",
        "fig_best_case",
        "fig_worst_case",
    )
    require(
        all((STUDY_ROOT / "figures" / f"{stem}.{suffix}").is_file() for stem in figure_stems for suffix in ("png", "svg", "pdf")),
        "required figure formats missing",
    )
    geojson_paths = list((STUDY_ROOT / "traces").glob("**/*.geojson"))
    require(bool(geojson_paths), "GeoJSON traces missing")
    for path in geojson_paths:
        payload = _json(path)
        if payload.get("type") != "FeatureCollection" or any(
            feature.get("geometry", {}).get("type") != "LineString" for feature in payload.get("features", [])
        ):
            failures.append(f"invalid GeoJSON: {path}")
            break

    thesis_path = STUDY_ROOT / "THESIS_NUMBERS.json"
    report_path = STUDY_ROOT / "FINAL_REPORT.md"
    require(thesis_path.is_file(), "THESIS_NUMBERS.json missing")
    require(report_path.is_file(), "FINAL_REPORT.md missing")
    if thesis_path.is_file():
        require(all_finite(_json(thesis_path)), "THESIS_NUMBERS.json contains NaN or Inf")
    provenance_path = STUDY_ROOT / "REPORT_PROVENANCE.json"
    require(provenance_path.is_file(), "REPORT_PROVENANCE.json missing")
    if provenance_path.is_file() and thesis_path.is_file() and report_path.is_file():
        provenance = _json(provenance_path)
        require(provenance.get("thesis_numbers_sha256") == _sha256(thesis_path), "report thesis provenance hash mismatch")
        require(provenance.get("report_sha256") == _sha256(report_path), "report content provenance hash mismatch")
        require(not provenance.get("untraced_numeric_tokens"), "report contains untraced numeric values")
    research_paths = [
        *list((PROJECT_ROOT / "datasets").glob("*.py")),
        *list((PROJECT_ROOT / "trace_extraction").glob("*.py")),
        *list((PROJECT_ROOT / "scripts").glob("*geocrack*.py")),
    ]
    forbidden = re.compile(r"\bTODO\b")
    for path in research_paths:
        if path.resolve() == Path(__file__).resolve():
            continue
        if forbidden.search(path.read_text(encoding="utf-8")):
            failures.append(f"forbidden TODO token in research path: {path.relative_to(PROJECT_ROOT)}")

    smoke = STUDY_ROOT / "smoke_test_report.json"
    require(smoke.is_file() and _json(smoke).get("status") == "PASS" if smoke.is_file() else False, "vertical smoke is not PASS")
    return failures


def mark_complete() -> None:
    state_path = PROJECT_ROOT / ".agent-state" / "TASK_STATE.json"
    evidence_path = PROJECT_ROOT / ".agent-state" / "EVIDENCE.json"
    state = _json(state_path)
    state.update(
        {
            "phase": "complete",
            "status": "complete",
            "current_acceptance_gate": "GE0CRACK STUDY STATUS: COMPLETE",
            "next_action": None,
            "blockers": [],
        }
    )
    completed = list(state.get("completed", []))
    if "full_geocrack_study" not in completed:
        completed.append("full_geocrack_study")
    state["completed"] = completed
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = _json(evidence_path)
    import_manifest = PROJECT_ROOT / "data" / "geocrack" / "manual_import_manifest.json"
    if not import_manifest.is_file():
        import_manifest = PROJECT_ROOT / "data" / "geocrack" / "download_manifest.json"
    evidence.update(
        {
            "split_hash": _split_hash(),
            "dataset_checksum": _sha256(import_manifest),
            "frozen_test_split_hash": _sha256(SPLIT_DIR / "geocrack_small_v1_test.csv"),
            "leakage_check": "PASS",
            "smoke_test": "PASS",
            "baseline_complete": True,
            "az_complete": True,
            "ablations_complete": True,
            "statistics_complete": True,
            "figures_complete": True,
            "final_validation": "PASS",
        }
    )
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepared_artifacts_ok() -> bool:
    root = STUDY_ROOT / "prepared" / "synthetic_pipeline"
    status_path = root / "synthetic_pipeline_status.json"
    if not status_path.is_file():
        return False
    status = _json(status_path)
    if status.get("status") != "PASS" or status.get("scientific_result") is not False:
        return False
    if set(status.get("steps", {}).values()) != {"PASS"}:
        return False
    stems = ("synthetic_pipeline_overview", "synthetic_metric_comparison")
    if not all((root / "figures" / f"{stem}.{suffix}").is_file() for stem in stems for suffix in ("png", "svg", "pdf")):
        return False
    try:
        from PIL import Image

        for stem in stems:
            with Image.open(root / "figures" / f"{stem}.png") as image:
                if min(image.info.get("dpi", (0, 0))) < 299:
                    return False
        provenance = _json(root / "REPORT_PROVENANCE.json")
        if provenance.get("thesis_numbers_sha256") != _sha256(root / "THESIS_NUMBERS.json"):
            return False
        if provenance.get("report_sha256") != _sha256(root / "FINAL_REPORT.md"):
            return False
    except (OSError, ValueError, KeyError):
        return False
    return True


def _real_data_ready() -> bool:
    manual = PROJECT_ROOT / "data" / "geocrack" / "manual_import_manifest.json"
    if manual.is_file():
        payload = _json(manual)
        return payload.get("validation", {}).get("pair_count") == 12158
    official = PROJECT_ROOT / "data" / "geocrack" / "download_manifest.json"
    return official.is_file() and bool(_json(official).get("files"))


def prepared_checks() -> tuple[list[tuple[str, str]], dict[str, str]]:
    evidence = _json(PROJECT_ROOT / ".agent-state" / "EVIDENCE.json")
    prepared_root = STUDY_ROOT / "prepared"
    prepared_root.mkdir(parents=True, exist_ok=True)
    compile_result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "datasets", "trace_extraction", "scripts"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    pytest_result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
    )
    (prepared_root / "pytest.log").write_text(pytest_result.stdout, encoding="utf-8")
    launcher = subprocess.run(
        ["bash", "scripts/run_geocrack_full_study.sh", "--dry-run"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    expected_runs = [
        "baseline seed 41",
        "baseline seed 42",
        "baseline seed 43",
        "az_thesis seed 41",
        "az_thesis seed 42",
        "az_thesis seed 43",
        "az_no_fuzzy seed 42",
        "az_no_aniso seed 42",
        "attention_unet seed 42",
    ]
    launcher_lines = launcher.stdout.splitlines()
    synthetic_ok = _prepared_artifacts_ok()
    trace_ok = evidence.get("trace_boundary_and_math", {}).get("status") == "PASS"
    orchestration_ok = evidence.get("orchestration_contracts", {}).get("status") == "PASS"
    checks = [
        ("CODE", "PASS" if compile_result.returncode == 0 else "FAIL"),
        ("UNIT TESTS", "PASS" if pytest_result.returncode == 0 else "FAIL"),
        ("SYNTHETIC PIPELINE", "PASS" if synthetic_ok else "FAIL"),
        ("MANUAL IMPORT", "PASS" if evidence.get("manual_import", {}).get("status") == "PASS" else "FAIL"),
        ("TRACE EXTRACTION", "PASS" if trace_ok else "FAIL"),
        ("METRICS", "PASS" if trace_ok and synthetic_ok else "FAIL"),
        ("BOOTSTRAP", "PASS" if evidence.get("bootstrap_validation", {}).get("status") == "PASS" else "FAIL"),
        ("FIGURE GENERATION", "PASS" if synthetic_ok else "FAIL"),
        ("RESUME", "PASS" if orchestration_ok else "FAIL"),
        (
            "LINUX LAUNCHER",
            "PASS"
            if launcher.returncode == 0 and launcher_lines[:9] == expected_runs and "PROTOCOL FAIRNESS: PASS" in launcher.stdout
            else "FAIL",
        ),
        ("REAL DATA", "PASS" if _real_data_ready() else "WAITING"),
    ]
    diagnostics = {
        "compile_output": compile_result.stdout.strip(),
        "pytest_output": pytest_result.stdout.strip(),
        "launcher_output": launcher.stdout.strip(),
    }
    return checks, diagnostics


def mark_prepared_complete() -> None:
    state_path = PROJECT_ROOT / ".agent-state" / "TASK_STATE.json"
    evidence_path = PROJECT_ROOT / ".agent-state" / "EVIDENCE.json"
    state = _json(state_path)
    completed = list(state.get("completed", []))
    if "geocrack_preparation" not in completed:
        completed.append("geocrack_preparation")
    state.update(
        {
            "phase": "waiting_real_data",
            "status": "prepared_complete",
            "completed": completed,
            "current_acceptance_gate": "GE0CRACK PREPARATION STATUS: COMPLETE",
            "next_action": (
                "/home/lebedeffson/Code/venv/bin/python scripts/download_geocrack.py "
                "--local-archive data/geocrack/incoming/geocrack_patched_data.zip"
            ),
            "current_failure": None,
            "blockers": [
                {
                    "scope": "real_geocrack_data",
                    "detail": "WAITING for a completed official archive; partial browser downloads are intentionally rejected",
                }
            ],
        }
    )
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = _json(evidence_path)
    evidence.update({"preparation_validation": "PASS", "final_validation": "WAITING_REAL_DATA"})
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepared", "final"), default="final")
    args = parser.parse_args()
    if args.phase == "prepared":
        checks, diagnostics = prepared_checks()
        for name, status in checks:
            print(f"{name:<22} {status}")
        failed = [name for name, status in checks if status == "FAIL"]
        if failed:
            print(f"\nGE0CRACK PREPARATION STATUS: INCOMPLETE ({len(failed)} failed checks)")
            if "UNIT TESTS" in failed:
                print(diagnostics["pytest_output"])
            return 1
        mark_prepared_complete()
        print("\nGE0CRACK PREPARATION STATUS: COMPLETE")
        return 0
    failures = collect_failures()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print(f"GE0CRACK STUDY STATUS: INCOMPLETE ({len(failures)} failed checks)")
        return 1
    mark_complete()
    state = _json(PROJECT_ROOT / ".agent-state" / "TASK_STATE.json")
    if state.get("status") != "complete":
        print("FAIL: TASK_STATE did not become complete")
        return 1
    print("================================================")
    print("GE0CRACK STUDY STATUS: COMPLETE")
    print("================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
