from __future__ import annotations

import csv
import json

import pytest

import scripts.validate_anza_v2_deadline as validator
from scripts.anza_v2_deadline_common import ALLOWED_VERDICTS, READY_VERDICTS, atomic_write_json


def _write(path, payload) -> None:
    atomic_write_json(path, payload)


def test_allowed_verdicts_are_exactly_deadline_contract() -> None:
    assert ALLOWED_VERDICTS == (
        "DEADLINE_RESULT_READY",
        "DEADLINE_RESULT_READY_WITH_NEGATIVE_MECHANISM",
        "BLOCKED_EVALUATOR",
        "BLOCKED_THRESHOLD_FREEZE",
        "BLOCKED_EXPERT_EVALUATION",
    )


def test_validator_maps_negative_mechanism_to_ready_with_negative(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(validator, "DEADLINE_ROOT", tmp_path)
    monkeypatch.setattr(validator, "_runs_gate", lambda: None)
    monkeypatch.setattr(validator, "_synthetic_gate", lambda: "NOT_ESTABLISHED")
    monkeypatch.setattr(validator, "_threshold_gate", lambda: None)
    monkeypatch.setattr(validator, "_expert_gate", lambda: None)
    monkeypatch.setattr(validator, "_table_gate", lambda: None)
    monkeypatch.setattr(validator, "_evidence_gate", lambda: None)
    result = validator.validate_deadline()
    assert result["verdict"] == "DEADLINE_RESULT_READY_WITH_NEGATIVE_MECHANISM"
    assert result["verdict"] in READY_VERDICTS
    assert result["full_cracks_study_complete"] is False


def test_validator_stops_at_threshold_freeze_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(validator, "DEADLINE_ROOT", tmp_path)
    monkeypatch.setattr(validator, "_runs_gate", lambda: None)
    monkeypatch.setattr(validator, "_synthetic_gate", lambda: "SUPPORTED_ABOVE_CHANCE")
    monkeypatch.setattr(validator, "_threshold_gate", lambda: (_ for _ in ()).throw(AssertionError("freeze")))
    monkeypatch.setattr(validator, "_expert_gate", lambda: (_ for _ in ()).throw(RuntimeError("must not run")))
    result = validator.validate_deadline()
    assert result["verdict"] == "BLOCKED_THRESHOLD_FREEZE"
    assert [row["check"] for row in result["checks"]] == [
        "RUNS_15", "CORRECTED_SYNTHETIC", "THRESHOLD_FREEZE"
    ]


def test_synthetic_gate_enforces_nonoverlap_and_evidence_labels(tmp_path, monkeypatch) -> None:
    study = tmp_path / "study"
    deadline = study / "deadline_20260817"
    monkeypatch.setattr(validator, "STUDY_ROOT", study)
    monkeypatch.setattr(validator, "DEADLINE_ROOT", deadline)
    _write(study / "synthetic" / "test" / "test_open_receipt.json", {"status": "OPENED_ONCE", "sample_count": 2000})
    _write(
        study / "synthetic" / "replacement_confirmation" / "freeze.json",
        {
            "status": "FROZEN_BEFORE_OPENING",
            "original_test_indices": [0, 2000],
            "replacement_test_indices": [2000, 4000],
        },
    )
    _write(study / "synthetic" / "evaluator_audit" / "validation_audit.json", {"status": "COMPLETE"})
    _write(
        study / "synthetic" / "evaluator_audit" / "legacy_test_reanalysis" / "summary.json",
        {"status": "POSTHOC_REANALYSIS_NOT_CONFIRMATORY"},
    )
    replacement_freeze = study / "synthetic" / "replacement_confirmation" / "freeze.json"
    _write(
        study / "synthetic" / "replacement_confirmation" / "summary.json",
        {
            "status": "REPLACEMENT_CONFIRMATION_AFTER_EVALUATOR_AUDIT",
            "sample_count": 2000,
            "no_retraining": True,
        },
    )
    _write(
        deadline / "SYNTHETIC_GATE_AUDIT.json",
        {"legacy_gate_validity": "PARTIALLY_INVALIDATED", "corrected_mechanism_evidence": {"verdict": "NEGATIVE"}},
    )
    assert validator._synthetic_gate() == "NEGATIVE"

    _write(
        replacement_freeze,
        {
            "status": "FROZEN_BEFORE_OPENING",
            "original_test_indices": [0, 2000],
            "replacement_test_indices": [1999, 3999],
        },
    )
    with pytest.raises(AssertionError, match="overlap or drifted"):
        validator._synthetic_gate()


def test_synthetic_table_rejects_fake_baseline_route_metrics(tmp_path, monkeypatch) -> None:
    study = tmp_path / "study"
    deadline = tmp_path / "deadline"
    tables = deadline / "tables"
    tables.mkdir(parents=True)
    monkeypatch.setattr(validator, "STUDY_ROOT", study)
    monkeypatch.setattr(validator, "DEADLINE_ROOT", deadline)
    for name in ("paired_comparisons.csv", "ablations.csv"):
        with (tables / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["status"], lineterminator="\n")
            writer.writeheader()
            writer.writerow({"status": "COMPLETE"})
    with (tables / "main_cracks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model"], lineterminator="\n")
        writer.writeheader()
        for model in ("unet", "deformable_unet", "anza_v1", "anza_v2b"):
            writer.writerow({"model": model})
    synthetic = tables / "synthetic_corrected.csv"
    with synthetic.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "route_top1_hit", "uses_generator_branch_geometry"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow({"model": "unet", "route_top1_hit": "0.7", "uses_generator_branch_geometry": "false"})
        writer.writerow(
            {
                "model": "geometry_only_minimum_angle_heuristic",
                "route_top1_hit": "NA",
                "uses_generator_branch_geometry": "true",
            }
        )
    with pytest.raises(AssertionError, match="assigned model-specific route"):
        validator._table_gate()


def test_synthetic_table_allows_baseline_route_availability_metadata(tmp_path, monkeypatch) -> None:
    deadline = tmp_path / "deadline"
    tables = deadline / "tables"
    tables.mkdir(parents=True)
    monkeypatch.setattr(validator, "DEADLINE_ROOT", deadline)
    for name in ("paired_comparisons.csv", "ablations.csv"):
        with (tables / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["status"], lineterminator="\n")
            writer.writeheader()
            writer.writerow({"status": "COMPLETE"})
    with (tables / "main_cracks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model"], lineterminator="\n")
        writer.writeheader()
        for model in ("unet", "deformable_unet", "anza_v1", "anza_v2b"):
            writer.writerow({"model": model})
    with (tables / "synthetic_corrected.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "route_available",
                "route_row_count",
                "route_top1_hit",
                "uses_generator_branch_geometry",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "unet",
                "route_available": "False",
                "route_row_count": "0.0",
                "route_top1_hit": "NA",
                "uses_generator_branch_geometry": "false",
            }
        )
        writer.writerow(
            {
                "model": "deformable_unet",
                "route_available": "False",
                "route_row_count": "0.0",
                "route_top1_hit": "NA",
                "uses_generator_branch_geometry": "false",
            }
        )
        writer.writerow(
            {
                "model": "anza_v1",
                "route_available": "False",
                "route_row_count": "0.0",
                "route_top1_hit": "NA",
                "uses_generator_branch_geometry": "false",
            }
        )
        writer.writerow(
            {
                "model": "geometry_only_minimum_angle_heuristic",
                "route_available": "True",
                "route_row_count": "1.0",
                "route_top1_hit": "0.8",
                "uses_generator_branch_geometry": "true",
            }
        )

    validator._table_gate()
