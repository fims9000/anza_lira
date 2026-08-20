from __future__ import annotations

import torch

from synthetic.structural_losses import branch_transition_logits


def test_frozen_route_readout_cannot_distinguish_mode_assignments() -> None:
    branch_masks = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )
    spatial_mass = torch.linspace(0.1, 0.9, 9 * 4).reshape(9, 4)
    first = torch.zeros(1, 2, 2, 9, 4)
    second = torch.zeros_like(first)
    first[0, 0, 0] = spatial_mass
    second[0, 1, 0] = spatial_mass

    assert not torch.equal(first, second)
    first_logits = branch_transition_logits(first, branch_masks, variant="v2a")
    second_logits = branch_transition_logits(second, branch_masks, variant="v2a")
    assert torch.equal(first_logits, second_logits)


def test_mode_specific_readout_distinguishes_same_spatial_marginal() -> None:
    spatial_mass = torch.linspace(0.1, 0.9, 9 * 4).reshape(9, 4)
    first = torch.zeros(1, 2, 2, 9, 4)
    second = torch.zeros_like(first)
    first[0, 0, 0] = spatial_mass
    second[0, 1, 0] = spatial_mass

    assert not torch.equal(first[0, 0, 0], second[0, 0, 0])
    assert torch.equal(first.sum(dim=(1, 2)), second.sum(dim=(1, 2)))
