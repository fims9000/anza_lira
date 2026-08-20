from __future__ import annotations

from dataclasses import replace

import torch

from method_repair.context_matrix import context_matrix
from method_repair.context_training import (
    build_context_candidate,
    context_candidate_loss,
    load_context_checkpoint,
    run_context_candidate,
)
from synthetic.crossing_trace_bench_v3 import generate_sample_v3


def test_b0_b3_matrix_is_exact_and_hashed_distinctly() -> None:
    matrix = context_matrix()
    assert [spec.candidate_id for spec in matrix] == ["B0", "B1", "B2", "B3"]
    assert len({spec.run_hash for spec in matrix}) == 4
    assert [spec.seed for spec in matrix] == [42] * 4
    assert matrix[0].model == "a3_pointwise"
    assert matrix[1].contextual_gate and not matrix[1].contrastive_route
    assert matrix[2].contrastive_route and not matrix[2].paired_gap_loss
    assert matrix[3].paired_gap_loss


def test_each_candidate_loss_is_finite_and_backpropagates() -> None:
    sample = generate_sample_v3("train", 256, image_size=32)
    for spec in context_matrix():
        model = build_context_candidate(spec, widths=(4, 6, 8, 10))
        loss, _ = context_candidate_loss(model, spec, sample, torch.device("cpu"))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_checkpoint_resume_hash_is_fail_closed(tmp_path) -> None:
    spec = context_matrix()[1]
    result = run_context_candidate(
        spec,
        tmp_path,
        device="cpu",
        epochs=1,
        train_samples=2,
        validation_samples=2,
        image_size=32,
        widths=(4, 6, 8, 10),
    )
    assert result["status"] == "COMPLETE"
    repeated = run_context_candidate(
        spec,
        tmp_path,
        device="cpu",
        epochs=1,
        train_samples=2,
        validation_samples=2,
        image_size=32,
        widths=(4, 6, 8, 10),
    )
    assert repeated["action"] == "SKIP"
    model = build_context_candidate(spec, widths=(4, 6, 8, 10))
    checkpoint = tmp_path / f"{spec.candidate_id}-{spec.run_hash}" / "checkpoint-last.pt"
    try:
        load_context_checkpoint(checkpoint, expected_hash=replace(spec, seed=41).run_hash, model=model)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("changed config must not reuse checkpoint")
