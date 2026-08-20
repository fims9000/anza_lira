#!/usr/bin/env python3
"""Fail-closed final validator for the ANZA-LIRA v2 CRACKS study."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cracks_experiment.evaluation import verify_threshold_freeze
from cracks_experiment.evidence import validate_report_numbers
from cracks_experiment.finetuning import FOLDS, setting_b_sources, verify_setting_a_complete
from cracks_experiment.matrix import setting_a_matrix
from cracks_experiment.robustness import setting_c_models


STUDY = PROJECT_ROOT / "results" / "anza_v2_study"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _finite_artifact(path: Path) -> bool:
    if path.suffix == ".json":
        payload: Any = json.loads(path.read_text())

        def check(value: Any) -> bool:
            if isinstance(value, float):
                return math.isfinite(value)
            if isinstance(value, dict):
                return all(check(item) for item in value.values())
            if isinstance(value, list):
                return all(check(item) for item in value)
            return True

        return check(payload)
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                for value in row.values():
                    if value and value.strip().lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf"}:
                        return False
    return True


def _archive_gate() -> None:
    inventory = _json(PROJECT_ROOT / "results" / "cracks_study" / "archive_inventory.json")
    expected = {
        "images.zip": "6557236191763af7bd8298ecb136d41e",
        "Fault segmentations.zip": "01e1697e886da2079ff3c1967334a7ca",
    }
    for name, md5 in expected.items():
        record = inventory["archives"][name]
        if record["md5"] != md5 or record["md5_status"] != "PASS" or record["zip_crc_status"] != "PASS":
            raise AssertionError(f"Archive verification failed for {name}")
    if inventory["images"]["file_count"] != 396 or inventory["annotations"]["file_count"] != 12603:
        raise AssertionError("CRACKS extracted inventory count mismatch")
    policies = _json(STUDY / "cracks" / "mask_semantics" / "policies.json")
    if set(policies) != {"paper_like", "conservative"}:
        raise AssertionError("Both explicit CRACKS mask policies are required")
    crowd = _json(STUDY / "cracks" / "crowd_target" / "manifest.json")
    if crowd.get("status") != "COMPLETE" or len(crowd.get("records", [])) != 1570:
        raise AssertionError("Crowd target manifest incomplete")
    protocol = _json(STUDY / "protocol.json")
    if protocol.get("status") != "FROZEN_BEFORE_TRAINING" or protocol.get("source_archive_status") != "VERIFIED":
        raise AssertionError("CRACKS protocol was not frozen before training")


def _synthetic_gate() -> None:
    frozen = _json(STUDY / "synthetic" / "frozen_v2.json")
    receipt = _json(STUDY / "synthetic" / "test" / "test_open_receipt.json")
    if frozen["status"] != "FROZEN_NEGATIVE_DEVELOPMENT_RESULT":
        raise AssertionError("Synthetic frozen result missing")
    if receipt["status"] != "OPENED_ONCE" or receipt["freeze_sha256"] != frozen["freeze_sha256"]:
        raise AssertionError("Synthetic test freeze/open provenance mismatch")
    if receipt["sample_count"] != 2000 or set(receipt["candidate_ids"]) != {"B0", "B1", "C0", "C3"}:
        raise AssertionError("Synthetic test matrix incomplete")


def _setting_a_gate() -> None:
    root = STUDY / "cracks" / "setting_a"
    for spec in setting_a_matrix():
        run_dir = root / f"{spec.run_id}-{spec.run_hash}"
        status = _json(run_dir / "status.json")
        if status.get("status") != "COMPLETE" or status.get("epoch") != 20 or status.get("expert_scores_used") is not False:
            raise AssertionError(f"Incomplete or contaminated Setting A run {spec.run_id}")
        validation = _json(run_dir / "crowd_validation.json")
        if validation.get("section_count") != 392 or validation.get("expert_scores_used") is not False:
            raise AssertionError(f"Incomplete Setting A crowd validation {spec.run_id}")
    verify_threshold_freeze(root)
    verify_setting_a_complete(root, STUDY / "cracks" / "setting_a_expert")


def _fold_gate(setting: str, models: tuple[Any, ...]) -> None:
    root = STUDY / "cracks" / setting
    for spec in models:
        covered = []
        for fold in FOLDS["folds"]:
            prefix = f"{spec.model}_fold{fold['fold']}-"
            runs = sorted(root.glob(prefix + "*"))
            if len(runs) != 1:
                raise AssertionError(f"Expected exactly one {setting} run for {prefix}")
            status = _json(runs[0] / "status.json")
            selection = _json(runs[0] / "selection.json")
            if status.get("status") != "COMPLETE" or not status.get("test_scores_used_after_selection"):
                raise AssertionError(f"Incomplete {setting} result for {prefix}")
            if selection.get("test_scores_used") is not False:
                raise AssertionError(f"Test leakage in {setting} selection for {prefix}")
            selection_sha = selection.pop("sha256", None)
            expected_sha = hashlib.sha256(
                json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if selection_sha != expected_sha or status.get("selection_sha256") != selection_sha:
                raise AssertionError(f"Selection receipt checksum mismatch in {setting} for {prefix}")
            rows = list(csv.DictReader((runs[0] / "test_sections.csv").open(newline="")))
            ids = [int(row["section_id"]) for row in rows]
            if ids != list(fold["test"]):
                raise AssertionError(f"Frozen fold mismatch in {setting} for {prefix}")
            covered.extend(ids)
        if sorted(covered) != sorted(FOLDS["available_expert_subset"]):
            raise AssertionError(f"Expert sections not covered exactly once in {setting} for {spec.model}")


def _human_statistics_gate() -> None:
    human = _json(STUDY / "cracks" / "human_comparison" / "summary.json")
    disagreement = _json(STUDY / "cracks" / "disagreement" / "summary.json")
    statistics = _json(STUDY / "tables" / "statistics.json")
    if human.get("status") != "COMPLETE" or human.get("expert_section_count") != 40:
        raise AssertionError("Human baseline incomplete")
    if disagreement.get("status") != "COMPLETE" or disagreement.get("statistical_unit") != "seismic_section":
        raise AssertionError("Disagreement analysis incomplete or wrong statistical unit")
    if statistics.get("status") != "COMPLETE" or statistics.get("bootstrap_resamples") != 2000:
        raise AssertionError("Section bootstrap incomplete")
    traces = _json(STUDY / "cracks" / "traces" / "manifest.json")
    if traces.get("status") != "COMPLETE" or traces.get("geojson_count") != 160:
        raise AssertionError("Candidate trace GeoJSON export incomplete")
    for record in traces.get("records", []):
        path = STUDY / "cracks" / "traces" / record["geojson"]
        payload = _json(path)
        if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
            raise AssertionError(f"Invalid GeoJSON trace artifact {path}")


def _figure_evidence_gate() -> None:
    figure_root = STUDY / "cracks" / "figures"
    for index in range(1, 6):
        for suffix in ("png", "svg", "pdf"):
            path = figure_root / f"figure_{index}.{suffix}"
            if not path.exists() or path.stat().st_size == 0:
                raise AssertionError(f"Missing figure artifact {path.name}")
    numbers = _json(STUDY / "THESIS_NUMBERS.json")
    report = (STUDY / "FINAL_REPORT.md").read_text()
    validate_report_numbers(report, numbers)
    audit = (STUDY / "SCIENTIFIC_AUDIT.md").read_text()
    if "SCIENTIFIC AUDIT: PASS" not in audit:
        raise AssertionError("Scientific audit did not pass")
    receipt = _json(STUDY / "pytest_receipt.json")
    if receipt.get("status") != "PASS" or receipt.get("exit_code") != 0:
        raise AssertionError("Full unit-test receipt missing")


def _finiteness_gate() -> None:
    artifacts = list(STUDY.rglob("*.json")) + list(STUDY.rglob("*.csv"))
    failed = [str(path.relative_to(STUDY)) for path in artifacts if not _finite_artifact(path)]
    if failed:
        raise AssertionError(f"NaN/Inf found in artifacts: {failed[:5]}")
    source_paths = [
        PROJECT_ROOT / "cracks_experiment",
        PROJECT_ROOT / "synthetic",
        PROJECT_ROOT / "models" / "azconv_v2.py",
        PROJECT_ROOT / "scripts" / "anza_v2_study.py",
    ]
    todo_files = []
    for root in source_paths:
        paths = [root] if root.is_file() else list(root.rglob("*.py"))
        todo_files.extend(path for path in paths if "TODO" in path.read_text())
    if todo_files:
        raise AssertionError(f"TODO remains in study code: {todo_files}")


def main() -> int:
    gates: list[tuple[str, Callable[[], None]]] = [
        ("ARCHIVES", _archive_gate),
        ("SYNTHETIC BENCHMARK", _synthetic_gate),
        ("SETTING A", _setting_a_gate),
        ("SETTING B", lambda: _fold_gate("setting_b", setting_b_sources())),
        ("SETTING C", lambda: _fold_gate("setting_c", setting_c_models())),
        ("HUMAN / BOOTSTRAP", _human_statistics_gate),
        ("FIGURES / EVIDENCE", _figure_evidence_gate),
        ("FINITE / SOURCE", _finiteness_gate),
    ]
    failures = []
    for name, gate in gates:
        try:
            gate()
        except Exception as error:  # validator must report every incomplete gate compactly
            failures.append((name, str(error)))
            print(f"{name:<24} FAIL  {error}")
        else:
            print(f"{name:<24} PASS")
    if failures:
        print("ANZA-LIRA V2 STUDY STATUS: INCOMPLETE")
        return 1
    print("====================================================")
    print("ANZA-LIRA V2 STUDY STATUS: COMPLETE")
    print("====================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
