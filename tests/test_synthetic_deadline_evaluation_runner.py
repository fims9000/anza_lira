from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from synthetic import deadline_evaluation_runner as runner
from synthetic.evaluator_audit import freeze_corrected_evaluator


class _FakeModel(torch.nn.Module):
    def forward(self, image: torch.Tensor, *, return_diagnostics: bool = False):
        logits = torch.full((image.shape[0], 1, image.shape[-2], image.shape[-1]), -0.25, device=image.device)
        if not return_diagnostics:
            return logits
        return {"visible_logits": logits, "transport_diagnostics": []}


def _study(tmp_path: Path) -> Path:
    study = tmp_path / "study"
    synthetic = study / "synthetic"
    for candidate_id, spec in runner._specs().items():
        run = synthetic / "development" / f"{candidate_id}-{spec.run_hash}"
        run.mkdir(parents=True)
        (run / "checkpoint-last.pt").write_bytes(f"checkpoint-{candidate_id}".encode())
        (run / "status.json").write_text(json.dumps({"status": "COMPLETE", "run_hash": spec.run_hash}))
        validation = synthetic / "validation" / f"{candidate_id}-{spec.run_hash}.json"
        validation.parent.mkdir(parents=True, exist_ok=True)
        validation.write_text(json.dumps({"selected_visible_threshold": 0.40}))
    legacy = synthetic / "test"
    legacy.mkdir(parents=True)
    (legacy / "summary.json").write_text(json.dumps({"models": {}}))
    return study


def test_extended_stream_preserves_legacy_boundary_and_opens_replacement_at_2000() -> None:
    legacy = runner._extended_test_sample(1999)
    replacement = runner._extended_test_sample(2000)
    assert legacy["index"] == 1999 and legacy["seed"] == 30_001_999
    assert replacement["index"] == 2000 and replacement["seed"] == 30_002_000
    with pytest.raises(IndexError):
        runner._extended_test_sample(4000)


def test_gap_summary_uses_negative_gaps_not_all_samples_as_denominator() -> None:
    empty = {
        "positive": {"rows": []},
        "negative": {"rows": []},
        "false_bridge_rate_at_fixed_0_5": 0.0,
    }
    bridged = {
        "positive": {"rows": []},
        "negative": {
            "rows": [{"coverage_at_0.50": 0.9, "connected_at_0.50": True}]
        },
        "false_bridge_rate_at_fixed_0_5": 1.0,
    }
    summary = runner._summarize_gap_audits([empty, empty, bridged])
    assert summary["negative"]["count"] == 1
    assert summary["false_bridge_count_at_fixed_0_5"] == 1
    assert summary["false_bridge_rate_at_fixed_0_5"] == 1.0


def test_validation_uses_real_corrected_computation_path_and_baselines_are_na(tmp_path, monkeypatch) -> None:
    study = _study(tmp_path)
    monkeypatch.setattr(runner, "_load_frozen_model", lambda spec, checkpoint, device: _FakeModel().to(device))
    calls = 0
    real = runner.evaluate_sample_corrected

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "evaluate_sample_corrected", counted)
    result = runner.compute_validation_audit(study, device="cpu", max_samples=1)
    assert calls == 4
    assert result["indices"] == [0, 1]
    assert result["test_samples_opened"] == 0
    for candidate_id in ("B0", "B1", "C0"):
        assert result["models"][candidate_id]["metrics"]["route_top1_hit"] is None
        assert result["models"][candidate_id]["metrics"]["route_excess_over_chance_ci95_low"] is None
    assert result["geometry_only_minimum_angle_heuristic"]["is_model_specific"] is False
    assert result["synthetic_corrected_rows"][-1]["model"] == "geometry_only_minimum_angle_heuristic"
    assert result["synthetic_corrected_rows"][-1]["uses_generator_branch_geometry"] is True


def test_legacy_reanalysis_never_writes_legacy_and_reports_hashes(tmp_path, monkeypatch) -> None:
    study = _study(tmp_path)
    legacy_path = study / "synthetic" / "test" / "summary.json"
    original = legacy_path.read_bytes()
    monkeypatch.setattr(runner, "_load_frozen_model", lambda spec, checkpoint, device: _FakeModel().to(device))
    result = runner.compute_legacy_reanalysis(study, device="cpu", max_samples=1)
    assert result["status"] == "POSTHOC_REANALYSIS_NOT_CONFIRMATORY"
    assert result["indices"] == [0, 1]
    assert result["legacy_originals_modified"] is False
    assert "summary.json" in result["legacy_output_hashes"]
    assert legacy_path.read_bytes() == original


def test_replacement_requires_matching_freeze_opens_2000_once_and_refuses_rerun(tmp_path, monkeypatch) -> None:
    study = _study(tmp_path)
    context = runner._frozen_context(study)
    freeze_corrected_evaluator(
        study,
        model_checkpoint_hashes=context["checkpoint_hashes"],
        visible_thresholds=context["thresholds"],
    )
    monkeypatch.setattr(runner, "_load_frozen_model", lambda spec, checkpoint, device: _FakeModel().to(device))
    result = runner.compute_replacement_confirmation(study, device="cpu", max_samples=1)
    assert result["indices"] == [2000, 2001]
    assert result["scientific_result"] is False
    receipt = json.loads((study / "synthetic" / "replacement_confirmation" / "open_receipt.json").read_text())
    assert receipt["frozen_range"] == [2000, 4000]
    assert receipt["executed_range"] == [2000, 2001]
    with pytest.raises(RuntimeError, match="already opened"):
        runner.compute_replacement_confirmation(study, device="cpu", max_samples=1)
