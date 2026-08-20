from __future__ import annotations

import json
from pathlib import Path
import sys
import types
import zipfile

import pytest

import scripts.anza_v2_deadline_finalize as deadline
from scripts.anza_v2_deadline_common import atomic_write_json


def test_phase_receipt_skips_only_when_input_and_output_hashes_match(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(deadline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(deadline, "PHASE_ROOT", tmp_path / "receipts")
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text("v1")
    calls = []

    def runner() -> dict[str, object]:
        calls.append("run")
        output.write_text(f"result-{len(calls)}")
        return {"artifacts": [output]}

    first = deadline.run_phase("example", runner, input_paths=(source,), required_outputs=(output,))
    second = deadline.run_phase("example", runner, input_paths=(source,), required_outputs=(output,))
    assert first["action"] == "RUN"
    assert second["action"] == "SKIP"
    assert calls == ["run"]

    source.write_text("v2")
    third = deadline.run_phase("example", runner, input_paths=(source,), required_outputs=(output,))
    assert third["action"] == "RUN"
    assert calls == ["run", "run"]

    output.write_text("tampered")
    fourth = deadline.run_phase("example", runner, input_paths=(source,), required_outputs=(output,))
    assert fourth["action"] == "RUN"
    assert calls == ["run", "run", "run"]


def test_corrected_evaluator_is_imported_only_when_invoked(monkeypatch) -> None:
    name = "synthetic.late_deadline_test_module"
    sys.modules.pop(name, None)
    with pytest.raises(RuntimeError, match="not implemented yet"):
        deadline._invoke(name, ("run",), study_root=Path("study"))

    module = types.ModuleType(name)
    module.run = lambda study_root: {"study_root": str(study_root)}
    monkeypatch.setitem(sys.modules, name, module)
    assert deadline._invoke(name, ("run",), study_root=Path("study"), ignored=True) == {
        "study_root": "study"
    }


def test_deadline_phase_plan_excludes_settings_b_and_c() -> None:
    names = [name for name, *_ in deadline._phases()]
    assert "crowd_threshold_freeze" in names
    assert "setting_a_expert" in names
    assert not any("setting_b" in name or "setting_c" in name for name in names)
    assert deadline.ORIGINAL_RANGE == (0, 2000)
    assert deadline.REPLACEMENT_RANGE == (2000, 4000)
    assert set(range(*deadline.ORIGINAL_RANGE)).isdisjoint(range(*deadline.REPLACEMENT_RANGE))
    expert_phase = next(item for item in deadline._phases() if item[0] == "setting_a_expert")
    assert expert_phase[3] == (
        deadline.STUDY_ROOT / "cracks" / "setting_a_expert" / "complete.json",
    )


def test_replacement_accepts_reporting_rows_from_corrected_runner(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        deadline,
        "_compute_corrected",
        lambda kind: {
            "sample_count": 2000,
            "synthetic_corrected_rows": [{"model": "anza_v2b"}],
        },
    )
    import synthetic.evaluator_audit as evaluator_audit

    def persist(_study_root, *, confirmation):
        captured.update(confirmation)
        return {"status": "REPLACEMENT_CONFIRMATION_AFTER_EVALUATOR_AUDIT"}

    monkeypatch.setattr(evaluator_audit, "run_replacement_confirmation", persist)
    result = deadline._run_replacement()
    assert result["status"] == "REPLACEMENT_CONFIRMATION_AFTER_EVALUATOR_AUDIT"
    assert captured["synthetic_rows"] == [{"model": "anza_v2b"}]


def test_existing_pre_fix_snapshot_is_immutable_and_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(deadline, "DEADLINE_ROOT", tmp_path)
    snapshot = tmp_path / "PRE_FIX_SNAPSHOT.json"
    payload = {"status": "CAPTURED_BEFORE_EVALUATOR_FIX", "snapshot_sha256": "frozen"}
    atomic_write_json(snapshot, payload)
    result = deadline.create_pre_fix_snapshot()
    assert result["action"] == "SKIP"
    assert json.loads(snapshot.read_text()) == payload


def test_run_audit_rejects_incomplete_run_before_reading_experts(tmp_path, monkeypatch) -> None:
    spec = types.SimpleNamespace(
        run_id="unet_s41",
        run_hash="hash",
        model="unet",
        seed=41,
        comparison_family="main",
    )
    monkeypatch.setattr(deadline, "STUDY_ROOT", tmp_path)
    monkeypatch.setattr(deadline, "DEADLINE_ROOT", tmp_path / "deadline")
    monkeypatch.setattr(deadline, "_setting_a_specs", lambda: (spec,))
    run_dir = tmp_path / "cracks" / "setting_a" / "unet_s41-hash"
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint-last.pt").write_bytes(b"checkpoint")
    atomic_write_json(
        run_dir / "status.json",
        {
            "status": "IN_PROGRESS",
            "epoch": 19,
            "epoch_budget": 20,
            "run_hash": "hash",
            "expert_scores_used": False,
            "checkpoint_reload": "PASS",
            "history": [],
        },
    )
    with pytest.raises(RuntimeError, match="Invalid completed"):
        deadline.audit_completed_runs()
    assert not (tmp_path / "deadline" / "RUNS_15_AUDIT.json").exists()


def test_package_arcname_handles_study_artifacts_and_project_docs(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    study = project / "results" / "anza_v2_study"
    monkeypatch.setattr(deadline, "PROJECT_ROOT", project)
    monkeypatch.setattr(deadline, "STUDY_ROOT", study)

    assert deadline._package_arcname(study / "deadline" / "report.md") == Path(
        "deadline/report.md"
    )
    assert deadline._package_arcname(project / "docs" / "research" / "audit.md") == Path(
        "docs/research/audit.md"
    )
    with pytest.raises(ValueError, match="outside the project"):
        deadline._package_arcname(tmp_path / "external.md")


def test_package_requires_raw_per_section_csv(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    study = project / "results" / "anza_v2_study"
    deadline_root = study / "deadline_20260817"
    monkeypatch.setattr(deadline, "PROJECT_ROOT", project)
    monkeypatch.setattr(deadline, "STUDY_ROOT", study)
    monkeypatch.setattr(deadline, "DEADLINE_ROOT", deadline_root)
    required = (
        "README_FIRST.md",
        "DEADLINE_SCOPE.md",
        "SYNTHETIC_GATE_AUDIT.json",
        "training_history.csv",
        "RUN_INDEX_FINAL.tsv",
        "DEADLINE_REPORT.md",
        "THESIS_NUMBERS.json",
        "THESIS_EVIDENCE.md",
        "raw_per_section.csv",
        "validator_result.json",
    )
    for name in required:
        path = deadline_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    threshold = study / "cracks" / "setting_a" / "threshold_freeze.json"
    threshold.parent.mkdir(parents=True)
    threshold.write_text("freeze")
    for name in ("main_cracks.csv", "paired_comparisons.csv", "ablations.csv", "synthetic_corrected.csv"):
        path = deadline_root / "tables" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)

    result = deadline._package()
    with zipfile.ZipFile(result["package"]) as archive:
        assert "deadline_20260817/raw_per_section.csv" in archive.namelist()
