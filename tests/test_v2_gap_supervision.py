from __future__ import annotations

import torch

from synthetic.crossing_trace_bench import generate_sample
from synthetic.structural_losses import structural_gap_loss, visible_segmentation_loss


def _tensor(sample: dict, name: str) -> torch.Tensor:
    return torch.as_tensor(sample[name]).unsqueeze(0).unsqueeze(0)


def test_observed_segmentation_and_positive_completion_are_distinct_objectives() -> None:
    sample = generate_sample("validation", 140, case="fault_with_gap")
    visible = _tensor(sample, "visible_fault_mask").float()
    positive_gap = _tensor(sample, "positive_gap_mask").bool()
    negative_gap = torch.zeros_like(positive_gap)
    observed_logits = torch.where(visible.bool(), torch.tensor(6.0), torch.tensor(-6.0))
    completed_logits = observed_logits.masked_fill(positive_gap, 6.0)
    observed_loss = visible_segmentation_loss(observed_logits, visible)
    completed_observed_loss = visible_segmentation_loss(completed_logits, visible)
    observed_gap_loss, _ = structural_gap_loss(
        torch.sigmoid(observed_logits), positive_gap, negative_gap
    )
    completed_gap_loss, _ = structural_gap_loss(
        torch.sigmoid(completed_logits), positive_gap, negative_gap
    )
    assert completed_observed_loss > observed_loss
    assert completed_gap_loss < observed_gap_loss


def test_matched_negative_gap_penalizes_unconditional_completion() -> None:
    positive_sample = generate_sample("validation", 141, case="fault_with_gap")
    negative_sample = generate_sample("validation", 141, case="negative_gap")
    positive_one = _tensor(positive_sample, "positive_gap_mask").bool()
    negative_one = _tensor(negative_sample, "negative_gap_mask").bool()
    positive = torch.cat([positive_one, torch.zeros_like(positive_one)], dim=0)
    negative = torch.cat([torch.zeros_like(negative_one), negative_one], dim=0)
    cautious = torch.full_like(positive, 0.01, dtype=torch.float32)
    cautious[positive] = 0.99
    close_everything = cautious.clone()
    close_everything[negative] = 0.99
    cautious_loss, cautious_parts = structural_gap_loss(cautious, positive, negative)
    closing_loss, closing_parts = structural_gap_loss(close_everything, positive, negative)
    assert closing_loss > cautious_loss
    assert closing_parts["negative_gap_loss"] > cautious_parts["negative_gap_loss"]


def test_gap_loss_empty_case_is_zero_finite_and_differentiable() -> None:
    logits = torch.randn(1, 1, 5, 5, requires_grad=True)
    probability = torch.sigmoid(logits)
    empty = torch.zeros_like(probability, dtype=torch.bool)
    loss, parts = structural_gap_loss(probability, empty, empty)
    loss.backward()
    assert loss.item() == 0.0
    assert all(value.item() == 0.0 for value in parts.values())
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
