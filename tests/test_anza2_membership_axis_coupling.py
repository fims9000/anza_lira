import torch

from models.anza2.losses import membership_weighted_axis_set_coverage_loss


def test_correct_angle_in_inactive_mode_is_not_accepted():
    predicted = torch.tensor([[[[[1.0]], [[0.0]]], [[[-1.0]], [[0.0]]]]])
    target = torch.tensor([[[[[1.0]], [[0.0]]]]])
    valid = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    inactive_correct = torch.tensor([[[[0.0]], [[1.0]]]])
    active_correct = torch.tensor([[[[1.0]], [[0.0]]]])
    bad = membership_weighted_axis_set_coverage_loss(predicted, inactive_correct, target, valid)
    good = membership_weighted_axis_set_coverage_loss(predicted, active_correct, target, valid)
    assert good == 0
    assert bad > 0.9


def test_multiple_active_modes_can_cover_crossing_axes():
    predicted = torch.tensor([[[[[1.0]], [[0.0]]], [[[-1.0]], [[0.0]]]]])
    target = predicted.clone()
    valid = torch.ones(1, 2, 1, 1, dtype=torch.bool)
    membership = torch.ones(1, 2, 1, 1)
    loss = membership_weighted_axis_set_coverage_loss(predicted, membership, target, valid)
    assert loss == 0
