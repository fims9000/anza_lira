from __future__ import annotations

import math

import torch

from synthetic.mode_supervision import axial_mode_set_loss


def _truth() -> tuple[torch.Tensor, torch.Tensor]:
    theta = torch.zeros(1, 4, 1, 2)
    valid = torch.zeros(1, 4, 1, 2, dtype=torch.bool)
    theta[:, 0, 0, 0] = 0.0
    valid[:, 0, 0, 0] = True
    theta[:, 0, 0, 1] = 0.0
    theta[:, 1, 0, 1] = math.pi / 2
    valid[:, :2, 0, 1] = True
    return theta, valid


def test_perfect_mode_set_has_zero_orientation_and_membership_kl() -> None:
    truth, valid = _truth()
    predicted = torch.tensor([0.0, math.pi / 2, 0.3, 1.0]).view(1, 4, 1, 1).expand(-1, -1, -1, 2).clone()
    membership = torch.zeros_like(predicted)
    membership[:, 0, 0, 0] = 1.0
    membership[:, 0, 0, 1] = 0.5
    membership[:, 1, 0, 1] = 0.5
    loss, details = axial_mode_set_loss(predicted, membership, truth, valid)
    assert loss.abs() < 1e-6
    assert details["orientation_set_loss"].abs() < 1e-6
    assert details["membership_set_kl"].abs() < 1e-6


def test_set_loss_is_permutation_and_pi_invariant() -> None:
    truth, valid = _truth()
    predicted = torch.tensor([0.0, math.pi / 2, 0.3, 1.0]).view(1, 4, 1, 1).expand(-1, -1, -1, 2).clone()
    membership = torch.softmax(torch.tensor([3.0, 3.0, -2.0, -2.0]).view(1, 4, 1, 1).expand_as(predicted), dim=1)
    first = axial_mode_set_loss(predicted, membership, truth, valid)[0]
    permutation = torch.tensor([2, 0, 3, 1])
    second = axial_mode_set_loss(
        predicted[:, permutation] + math.pi,
        membership[:, permutation],
        truth,
        valid,
    )[0]
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_set_loss_produces_finite_orientation_gradient() -> None:
    truth, valid = _truth()
    predicted = torch.randn(1, 4, 1, 2, requires_grad=True)
    membership_logits = torch.randn(1, 4, 1, 2, requires_grad=True)
    loss = axial_mode_set_loss(predicted, torch.softmax(membership_logits, dim=1), truth, valid)[0]
    loss.backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    assert membership_logits.grad is not None and torch.isfinite(membership_logits.grad).all()
