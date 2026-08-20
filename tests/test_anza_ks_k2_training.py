import torch

from anza_ks_k2.training import balanced_occupancy_loss, orientation_loss


def test_auxiliary_losses_are_finite_and_differentiable() -> None:
    orientation = torch.randn(2, 8, 12, 12, requires_grad=True)
    bank = torch.rand(2, 8, 48, 48)
    valid = torch.ones(2, 1, 48, 48)
    occupancy = torch.randn(2, 1, 12, 12, requires_grad=True)
    target = (torch.rand(2, 1, 48, 48) > 0.8).float()
    loss = orientation_loss(orientation, bank, valid) + balanced_occupancy_loss(occupancy, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(orientation.grad).all()
    assert torch.isfinite(occupancy.grad).all()
