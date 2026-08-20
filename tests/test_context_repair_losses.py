from __future__ import annotations

import torch

from synthetic.context_repair_losses import (
    contextual_gate_loss,
    matched_negative_route_loss,
    mode_cardinality_diagnostics,
    paired_gap_logit_loss,
)


def test_context_gate_prefers_junction_high_and_hard_negative_low() -> None:
    target = torch.tensor([[[1.0, 0.0], [0.5, 0.0]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    correct = torch.tensor([[[0.99, 0.01], [0.5, 0.01]]])
    inverted = 1.0 - correct
    assert contextual_gate_loss(correct, target, valid) < contextual_gate_loss(inverted, target, valid)


def test_contrastive_route_prefers_true_continuation_over_matched_negative() -> None:
    target = torch.tensor([[False, True, False], [True, False, False], [False, False, False]])
    eligible = torch.tensor([[False, True, True], [True, False, True], [True, True, False]])
    good = torch.tensor([[0.0, 3.0, -2.0], [3.0, 0.0, -2.0], [0.0, 0.0, 0.0]])
    bad = torch.tensor([[0.0, -2.0, 3.0], [-2.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
    assert matched_negative_route_loss(good, target, eligible, temperature=0.1) < matched_negative_route_loss(
        bad, target, eligible, temperature=0.1
    )


def test_paired_gap_loss_rewards_positive_and_rejects_negative_corridor() -> None:
    positive = torch.tensor([[[[True, False]]]])
    negative = ~positive
    correct_logits = torch.tensor([[[[8.0, -8.0]]]])
    closed_everywhere = torch.tensor([[[[8.0, 8.0]]]])
    correct, parts = paired_gap_logit_loss(correct_logits, positive, negative)
    closed, _ = paired_gap_logit_loss(closed_everywhere, positive, negative)
    assert correct < closed
    assert parts["positive_gap_loss"] < 0.001
    assert parts["negative_gap_loss"] < 0.001


def test_mode_cardinality_diagnostics_are_exact_on_known_membership() -> None:
    membership = torch.tensor([[[[1.0, 0.5]], [[0.0, 0.5]], [[0.0, 0.0]], [[0.0, 0.0]]]])
    truth = torch.tensor([[[1, 2]]])
    metrics = mode_cardinality_diagnostics(membership, truth)
    assert metrics["mode_count_accuracy"] == 1.0
    assert metrics["neff_mae"] < 1e-5
