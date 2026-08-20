from __future__ import annotations

import json

import pytest
import torch

from method_repair.matrix import MethodRepairSpec, synthetic_matrix
from method_repair.training import (
    build_candidate_model,
    load_candidate_checkpoint,
    repaired_candidate_loss,
    run_synthetic_candidate,
)
from synthetic.crossing_trace_bench_v2 import generate_sample_v2


def test_direct_supervision_loss_is_finite_and_records_separate_terms() -> None:
    spec = next(item for item in synthetic_matrix() if item.candidate_id == "A3")
    model = build_candidate_model(spec, widths=(4, 8, 12, 16))
    sample = generate_sample_v2("train", 0, image_size=24, case="x_junction")
    loss, parts = repaired_candidate_loss(model, spec, sample, torch.device("cpu"))
    loss.backward()
    assert torch.isfinite(loss)
    assert {"visible_bce_dice", "mode_set", "mode_specific_route", "total"}.issubset(parts)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_a0_and_a1_share_exact_initial_segmentation_path() -> None:
    a0, a1 = synthetic_matrix()[:2]
    torch.manual_seed(a0.seed)
    base = build_candidate_model(a0, widths=(4, 8, 12, 16)).eval()
    torch.manual_seed(a1.seed)
    repaired = build_candidate_model(a1, widths=(4, 8, 12, 16)).eval()
    sample = torch.randn(1, 3, 24, 24)
    with torch.no_grad():
        assert torch.equal(base(sample), repaired(sample))


def test_smoke_is_resumable_and_checkpoint_hash_fails_closed(tmp_path) -> None:
    spec = synthetic_matrix()[3]
    first = run_synthetic_candidate(
        spec,
        tmp_path,
        device="cpu",
        epochs=1,
        train_samples=2,
        validation_samples=1,
        image_size=24,
        widths=(4, 8, 12, 16),
    )
    second = run_synthetic_candidate(
        spec,
        tmp_path,
        device="cpu",
        epochs=1,
        train_samples=2,
        validation_samples=1,
        image_size=24,
        widths=(4, 8, 12, 16),
    )
    assert first["status"] == "COMPLETE"
    assert second["action"] == "SKIP"
    checkpoint = next(tmp_path.glob("A3-*/checkpoint-last.pt"))
    model = build_candidate_model(spec, widths=(4, 8, 12, 16))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_candidate_checkpoint(checkpoint, expected_hash="different", model=model)
    status = json.loads(next(tmp_path.glob("A3-*/status.json")).read_text())
    assert status["expert_data_accessed"] is False
    assert status["old_test_samples_opened"] == 0
    assert status["new_test_samples_opened"] == 0


def test_changed_configuration_changes_run_id() -> None:
    original = synthetic_matrix()[3]
    changed = MethodRepairSpec(
        original.candidate_id,
        original.model,
        original.use_ambiguity_gate,
        original.direct_mode_supervision,
        5,
        original.seed,
    )
    assert changed.run_hash != original.run_hash
