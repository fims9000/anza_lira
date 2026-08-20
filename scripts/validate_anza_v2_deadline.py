#!/usr/bin/env python3
"""Fail-closed validator for the deadline-scoped ANZA-LIRA evidence package."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.anza_v2_deadline_common import (
    ALLOWED_VERDICTS,
    READY_VERDICTS,
    atomic_write_json,
    finite_json,
    read_json,
    sha256_file,
    utc_now,
)


STUDY_ROOT = PROJECT_ROOT / "results" / "anza_v2_study"
DEADLINE_ROOT = STUDY_ROOT / "deadline_20260817"


def _require(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"Missing or empty artifact: {path}")
    return path


def _runs_gate() -> None:
    audit = read_json(_require(DEADLINE_ROOT / "RUNS_15_AUDIT.json"))
    if audit.get("status") != "COMPLETE" or audit.get("run_count") != 15:
        raise AssertionError("Setting A training is not 15/15 COMPLETE")
    records = audit.get("records", [])
    if len(records) != 15 or len({item.get("run_id") for item in records}) != 15:
        raise AssertionError("Setting A audit has duplicate or missing runs")
    if any(item.get("expert_scores_used") is not False for item in records):
        raise AssertionError("Expert-score contamination recorded during training")
    for record in records:
        run_dir = STUDY_ROOT / "cracks" / "setting_a" / f"{record['run_id']}-{record['run_hash']}"
        checkpoint = _require(run_dir / "checkpoint-last.pt")
        if sha256_file(checkpoint) != record.get("checkpoint_sha256"):
            raise AssertionError(f"Checkpoint changed after completion: {record['run_id']}")


def _synthetic_gate() -> str:
    legacy = read_json(_require(STUDY_ROOT / "synthetic" / "test" / "test_open_receipt.json"))
    if legacy.get("sample_count") != 2000 or legacy.get("status") != "OPENED_ONCE":
        raise AssertionError("Legacy synthetic test-open provenance changed")
    freeze = read_json(_require(STUDY_ROOT / "synthetic" / "replacement_confirmation" / "freeze.json"))
    validation = read_json(_require(STUDY_ROOT / "synthetic" / "evaluator_audit" / "validation_audit.json"))
    posthoc = read_json(_require(STUDY_ROOT / "synthetic" / "evaluator_audit" / "legacy_test_reanalysis" / "summary.json"))
    replacement_freeze = read_json(_require(STUDY_ROOT / "synthetic" / "replacement_confirmation" / "freeze.json"))
    replacement = read_json(_require(STUDY_ROOT / "synthetic" / "replacement_confirmation" / "summary.json"))
    gate = read_json(_require(DEADLINE_ROOT / "SYNTHETIC_GATE_AUDIT.json"))
    if freeze.get("status") != "FROZEN_BEFORE_OPENING":
        raise AssertionError("Corrected evaluator was not frozen")
    if validation.get("status") != "COMPLETE":
        raise AssertionError("Corrected validation audit incomplete")
    if posthoc.get("status") != "POSTHOC_REANALYSIS_NOT_CONFIRMATORY":
        raise AssertionError("Legacy test reanalysis is not labeled posthoc")
    if replacement_freeze.get("original_test_indices") != [0, 2000] or replacement_freeze.get("replacement_test_indices") != [2000, 4000]:
        raise AssertionError("Replacement confirmation ranges overlap or drifted")
    if replacement.get("status") != "REPLACEMENT_CONFIRMATION_AFTER_EVALUATOR_AUDIT":
        raise AssertionError("Replacement confirmation incomplete")
    if replacement.get("sample_count") != 2000 or replacement.get("no_retraining") is not True:
        raise AssertionError("Replacement sample count or no-retraining provenance failed")
    if gate.get("legacy_gate_validity") != "PARTIALLY_INVALIDATED":
        raise AssertionError("Legacy gate invalidation is not explicit")
    mechanism = gate.get("corrected_mechanism_evidence", {})
    verdict = mechanism.get("verdict")
    if verdict not in {"SUPPORTED_ABOVE_CHANCE", "NOT_ESTABLISHED", "NEGATIVE"}:
        raise AssertionError(f"Unsupported corrected mechanism verdict: {verdict}")
    return verdict


def _threshold_gate() -> None:
    from cracks_experiment.evaluation import verify_threshold_freeze

    training_root = STUDY_ROOT / "cracks" / "setting_a"
    verify_threshold_freeze(training_root)
    receipt = read_json(_require(training_root / "threshold_freeze.json"))
    if receipt.get("expert_scores_used") is not False:
        raise AssertionError("Expert scores were used before threshold freeze")
    for record in read_json(DEADLINE_ROOT / "RUNS_15_AUDIT.json")["records"]:
        validation = read_json(
            training_root / f"{record['run_id']}-{record['run_hash']}" / "crowd_validation.json"
        )
        if validation.get("section_count") != 392 or validation.get("expert_scores_used") is not False:
            raise AssertionError(f"Incomplete crowd validation: {record['run_id']}")


def _expert_gate() -> None:
    receipt = read_json(_require(STUDY_ROOT / "cracks" / "setting_a_expert" / "complete.json"))
    if receipt.get("status") != "COMPLETE" or receipt.get("run_count") != 15:
        raise AssertionError("Setting A expert evaluation receipt incomplete")
    for record in read_json(DEADLINE_ROOT / "RUNS_15_AUDIT.json")["records"]:
        path = _require(
            STUDY_ROOT / "cracks" / "setting_a_expert" / f"{record['run_id']}-{record['run_hash']}.csv"
        )
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        paper_rows = [row for row in rows if row.get("policy") == "paper_like"]
        if len(paper_rows) != 40 or {int(row["section_id"]) for row in paper_rows}.__len__() != 40:
            raise AssertionError(f"Expert raw section coverage is not exactly 40: {record['run_id']}")


def _table_gate() -> None:
    required = {
        "main_cracks.csv": {"unet", "deformable_unet", "anza_v1", "anza_v2b"},
        "paired_comparisons.csv": None,
        "ablations.csv": None,
        "synthetic_corrected.csv": None,
    }
    for name, models in required.items():
        path = _require(DEADLINE_ROOT / "tables" / name)
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise AssertionError(f"Empty deadline table: {name}")
        if models is not None and not models.issubset({row.get("model") for row in rows}):
            raise AssertionError("Main table lacks one or more frozen model families")
        for row in rows:
            if any(value.strip().lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf"} for value in row.values() if value):
                raise AssertionError(f"Non-finite table value: {name}")
    with (DEADLINE_ROOT / "tables" / "synthetic_corrected.csv").open(newline="") as handle:
        synthetic = list(csv.DictReader(handle))
    route_score_fields = (
        "route_top1_hit",
        "route_true_probability_mass",
        "route_mrr",
        "route_average_precision",
        "route_entropy_normalized",
        "route_excess_over_chance",
        "topology_constrained_pairing_score",
    )
    for row in synthetic:
        if row.get("model") in {"unet", "deformable_unet", "anza_v1"}:
            route_values = [row.get(key, "") for key in route_score_fields]
            if any(value not in {"", "NA", "N/A", "null"} for value in route_values):
                raise AssertionError("Baseline was assigned model-specific route metrics")
    geometry = [row for row in synthetic if row.get("model") == "geometry_only_minimum_angle_heuristic"]
    if len(geometry) != 1 or geometry[0].get("uses_generator_branch_geometry", "").lower() != "true":
        raise AssertionError("Geometry-only diagnostic is absent or mislabeled")


def _evidence_gate() -> None:
    for path in (
        DEADLINE_ROOT / "DEADLINE_SCOPE.md",
        DEADLINE_ROOT / "README_FIRST.md",
        DEADLINE_ROOT / "DEADLINE_REPORT.md",
        DEADLINE_ROOT / "THESIS_NUMBERS.json",
        DEADLINE_ROOT / "THESIS_EVIDENCE.md",
        DEADLINE_ROOT / "figures" / "fig_cracks_main.png",
        DEADLINE_ROOT / "figures" / "fig_cracks_examples.png",
        DEADLINE_ROOT / "figures" / "fig_ablation.png",
        DEADLINE_ROOT / "training_history.csv",
        DEADLINE_ROOT / "RUN_INDEX_FINAL.tsv",
    ):
        _require(path)
    scope = (DEADLINE_ROOT / "DEADLINE_SCOPE.md").read_text()
    if "NOT_RUN_DEADLINE_SCOPE" not in scope or "not used in submitted claims" not in scope:
        raise AssertionError("Deadline deferral of Settings B/C is not explicit")
    report = (DEADLINE_ROOT / "DEADLINE_REPORT.md").read_text()
    forbidden = ("Anosov", "ergodic", "unseen-image generalization")
    if any(term.lower() in report.lower() for term in forbidden):
        raise AssertionError("Deadline report contains a forbidden scientific claim")
    numbers = json.loads((DEADLINE_ROOT / "THESIS_NUMBERS.json").read_text())
    if not finite_json(numbers):
        raise AssertionError("THESIS_NUMBERS contains NaN or Inf")
    gate = read_json(DEADLINE_ROOT / "SYNTHETIC_GATE_AUDIT.json")
    mechanism_verdict = gate.get("corrected_mechanism_evidence", {}).get("verdict")
    expected_supported = mechanism_verdict == "SUPPORTED_ABOVE_CHANCE"
    if numbers.get("synthetic_mechanism_supported") is not expected_supported:
        raise AssertionError("THESIS_NUMBERS mechanism claim disagrees with synthetic gate")
    if numbers.get("synthetic_mechanism_verdict") != mechanism_verdict:
        raise AssertionError("THESIS_NUMBERS mechanism verdict disagrees with synthetic gate")
    false_bridge = gate.get("false_bridge_verdict", {}).get("status")
    if numbers.get("false_bridge_verdict") != false_bridge:
        raise AssertionError("THESIS_NUMBERS false-bridge verdict disagrees with synthetic gate")
    positive_sentence = "Transport carries continuation information above chance"
    if (positive_sentence in report) is not expected_supported:
        raise AssertionError("Deadline report mechanism claim disagrees with synthetic gate")


def validate_deadline() -> dict[str, Any]:
    checks: list[tuple[str, Callable[[], Any], str]] = [
        ("RUNS_15", _runs_gate, "BLOCKED_EVALUATOR"),
        ("CORRECTED_SYNTHETIC", _synthetic_gate, "BLOCKED_EVALUATOR"),
        ("THRESHOLD_FREEZE", _threshold_gate, "BLOCKED_THRESHOLD_FREEZE"),
        ("EXPERT_EVALUATION", _expert_gate, "BLOCKED_EXPERT_EVALUATION"),
        ("TABLES", _table_gate, "BLOCKED_EXPERT_EVALUATION"),
        ("EVIDENCE", _evidence_gate, "BLOCKED_EXPERT_EVALUATION"),
    ]
    statuses = []
    mechanism = None
    verdict = "DEADLINE_RESULT_READY"
    blocker = None
    for name, check, blocked_verdict in checks:
        try:
            result = check()
            if name == "CORRECTED_SYNTHETIC":
                mechanism = result
            statuses.append({"check": name, "status": "PASS"})
        except Exception as error:  # the validator records the exact failing gate
            statuses.append({"check": name, "status": "FAIL", "error": str(error)})
            verdict = blocked_verdict
            blocker = str(error)
            break
    if blocker is None and mechanism in {"NOT_ESTABLISHED", "NEGATIVE"}:
        verdict = "DEADLINE_RESULT_READY_WITH_NEGATIVE_MECHANISM"
    if verdict not in ALLOWED_VERDICTS:
        raise AssertionError(f"Internal invalid verdict: {verdict}")
    payload = {
        "schema_version": 1,
        "status": "PASS" if verdict in READY_VERDICTS else "BLOCKED",
        "verdict": verdict,
        "created_utc": utc_now(),
        "checks": statuses,
        "blocker": blocker,
        "settings_b_c": "NOT_RUN_DEADLINE_SCOPE",
        "full_cracks_study_complete": False,
    }
    atomic_write_json(DEADLINE_ROOT / "validator_result.json", payload)
    return payload


def main() -> int:
    result = validate_deadline()
    for item in result["checks"]:
        print(f"{item['check']:<24} {item['status']}")
    print(f"DEADLINE VERDICT: {result['verdict']}")
    print("CRACKS STUDY STATUS: NOT_DECLARED_BY_DEADLINE_VALIDATOR")
    return 0 if result["verdict"] in READY_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
