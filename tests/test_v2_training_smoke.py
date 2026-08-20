from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.segmentation_v2 import build_comparable_model
from synthetic.experiment_matrix import development_matrix
from synthetic.training import load_checkpoint, run_candidate_development, run_candidate_smoke


def test_full_structural_candidate_trains_reloads_and_skips_same_hash(tmp_path) -> None:
    spec = next(run for run in development_matrix() if run.candidate_id == "C5")
    first = run_candidate_smoke(spec, tmp_path, epochs=1, image_size=16)
    second = run_candidate_smoke(spec, tmp_path, epochs=1, image_size=16)
    assert first["status"] == "COMPLETE" and first["action"] == "RUN"
    assert first["checkpoint_reload"] == "PASS"
    assert first["test_samples_opened"] == 0
    assert second["status"] == "COMPLETE" and second["action"] == "SKIP"


def test_changed_candidate_config_gets_a_new_run_hash_and_directory(tmp_path) -> None:
    spec = next(run for run in development_matrix() if run.candidate_id == "B0")
    changed = replace(spec, kappa_theta=8.0)
    assert changed.run_hash != spec.run_hash
    run_candidate_smoke(spec, tmp_path, epochs=1, image_size=16)
    run_candidate_smoke(changed, tmp_path, epochs=1, image_size=16)
    assert (tmp_path / f"{spec.candidate_id}-{spec.run_hash}").is_dir()
    assert (tmp_path / f"{changed.candidate_id}-{changed.run_hash}").is_dir()


def test_checkpoint_hash_mismatch_is_rejected(tmp_path) -> None:
    spec = next(run for run in development_matrix() if run.candidate_id == "B0")
    run_candidate_smoke(spec, tmp_path, epochs=1, image_size=16)
    checkpoint = tmp_path / f"{spec.candidate_id}-{spec.run_hash}" / "checkpoint-last.pt"
    model = build_comparable_model(spec.model, widths=(4, 8, 12, 16))
    with pytest.raises(ValueError, match="config hash"):
        load_checkpoint(checkpoint, expected_hash="wrong", model=model)


def test_checkpoint_payload_is_finite(tmp_path) -> None:
    spec = next(run for run in development_matrix() if run.candidate_id == "B1")
    run_candidate_smoke(spec, tmp_path, epochs=1, image_size=16)
    checkpoint = torch.load(
        tmp_path / f"{spec.candidate_id}-{spec.run_hash}" / "checkpoint-last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert all(torch.isfinite(value).all() for value in checkpoint["model_state"].values())


def test_development_runner_resumes_and_skips_completed_budget(tmp_path) -> None:
    spec = next(run for run in development_matrix() if run.candidate_id == "B0")
    first = run_candidate_development(
        spec,
        tmp_path,
        device="cpu",
        epochs=1,
        train_samples=4,
        validation_samples=2,
        image_size=16,
    )
    second = run_candidate_development(
        spec,
        tmp_path,
        device="cpu",
        epochs=1,
        train_samples=4,
        validation_samples=2,
        image_size=16,
    )
    assert first["status"] == "COMPLETE" and first["checkpoint_reload"] == "PASS"
    assert second["action"] == "SKIP"
