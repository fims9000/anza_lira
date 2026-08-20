from __future__ import annotations

import math

import torch

from synthetic.structural_losses import (
    cone_consistency_loss,
    cone_consistency_values,
    cone_half_angle,
)


def test_junction_score_widens_but_bounds_the_cone() -> None:
    score = torch.tensor([0.0, 0.5, 1.0])
    alpha = cone_half_angle(score, alpha_straight=0.1, alpha_junction=0.7)
    assert torch.allclose(alpha, torch.tensor([0.1, 0.4, 0.7]))


def test_aligned_axial_neighbor_is_more_consistent_than_orthogonal_neighbor() -> None:
    membership = torch.ones(1, 1, 1, 3)
    theta = torch.tensor([[[[math.pi / 2, 0.0, 0.0]]]])
    junction = torch.zeros(1, 1, 3)
    values, valid = cone_consistency_values(
        membership,
        theta,
        junction,
        kernel_size=3,
        alpha_straight=0.1,
        alpha_junction=0.7,
        kappa_cone=4.0,
    )
    center_x = 1
    left_offset = 3
    right_offset = 5
    assert valid[0, 0, left_offset, 0, center_x] == 1
    assert values[0, 0, right_offset, 0, center_x] > values[0, 0, left_offset, 0, center_x]


def test_wider_junction_cone_relaxes_nonparallel_continuation() -> None:
    membership = torch.ones(1, 1, 1, 3)
    theta = torch.tensor([[[[0.6, 0.0, 0.0]]]])
    straight, _ = cone_consistency_values(
        membership,
        theta,
        torch.zeros(1, 1, 3),
        alpha_straight=0.1,
        alpha_junction=0.8,
    )
    junction, _ = cone_consistency_values(
        membership,
        theta,
        torch.ones(1, 1, 3),
        alpha_straight=0.1,
        alpha_junction=0.8,
    )
    assert junction[0, 0, 3, 0, 1] > straight[0, 0, 3, 0, 1]


def test_cone_loss_is_finite_differentiable_and_axial() -> None:
    membership_logits = torch.randn(1, 2, 4, 5, requires_grad=True)
    membership = torch.softmax(membership_logits, dim=1)
    theta = torch.randn(1, 2, 4, 5, requires_grad=True)
    junction = torch.rand(1, 4, 5)
    fault = torch.sigmoid(torch.randn(1, 1, 4, 5))
    loss = cone_consistency_loss(membership, theta, junction, fault)
    shifted = cone_consistency_loss(membership, theta + math.pi, junction, fault)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.allclose(loss, shifted, atol=1e-5)
    assert torch.isfinite(membership_logits.grad).all()
    assert torch.isfinite(theta.grad).all()
