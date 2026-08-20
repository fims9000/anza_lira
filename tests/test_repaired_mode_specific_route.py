from __future__ import annotations

import torch

from synthetic.mode_supervision import (
    branch_mode_masks_from_tangents,
    mode_specific_branch_transition_logits,
)
from synthetic.structural_losses import branch_transition_logits


def test_mode_specific_route_distinguishes_allocations_hidden_by_old_marginal() -> None:
    branch_masks = torch.ones(1, 2, 2)
    branch_mode_masks = torch.zeros(1, 2, 2, 2)
    branch_mode_masks[:, 0] = 1.0
    spatial = torch.full((9, 4), 0.5)
    first = torch.zeros(1, 2, 2, 9, 4)
    second = torch.zeros_like(first)
    first[0, 0, 0] = spatial
    second[0, 1, 0] = spatial

    assert torch.equal(
        branch_transition_logits(first, branch_masks, variant="v2a"),
        branch_transition_logits(second, branch_masks, variant="v2a"),
    )
    first_specific = mode_specific_branch_transition_logits(
        first, branch_mode_masks, kernel_size=3
    )
    second_specific = mode_specific_branch_transition_logits(
        second, branch_mode_masks, kernel_size=3
    )
    assert first_specific.item() > second_specific.item()


def test_branch_mode_masks_follow_axial_tangent_not_branch_order() -> None:
    predicted = torch.tensor([0.0, 1.57079632679]).view(1, 2, 1, 1)
    truth = torch.tensor([1.57079632679, 0.0]).view(1, 2, 1, 1)
    valid = torch.ones_like(truth, dtype=torch.bool)
    masks = branch_mode_masks_from_tangents(predicted, truth, valid)
    assert masks.shape == (1, 2, 2, 1, 1)
    assert masks[0, 0, 1, 0, 0] == 1
    assert masks[0, 1, 0, 0, 0] == 1


def test_mode_specific_route_keeps_gradient_to_selected_transport() -> None:
    transport = torch.full((1, 2, 2, 9, 4), 0.1, requires_grad=True)
    masks = torch.zeros(1, 2, 2, 2)
    masks[:, 0] = 1.0
    loss = -mode_specific_branch_transition_logits(transport, masks, kernel_size=3).mean()
    loss.backward()
    assert transport.grad is not None and torch.isfinite(transport.grad).all()
    assert transport.grad[0, 0, 0].abs().sum() > 0
    assert transport.grad[0, 1].abs().sum() == 0
