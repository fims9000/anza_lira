import torch

from models.anza2.losses import (
    background_membership_loss,
    membership_axis_coverage_loss,
    positive_exact_mode_count_loss,
    rc1_membership_loss,
)


def _orientation(vectors):
    return torch.tensor(vectors, dtype=torch.float32).view(1, len(vectors), 2, 1, 1)


def test_membership_cover_loss_is_finite_and_mode_permutation_invariant():
    predicted = _orientation([[1, 0], [0, 1], [-1, 0]])
    membership = torch.tensor([[[[0.8]], [[0.3]], [[0.1]]]])
    target = _orientation([[1, 0]])
    valid = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    left = membership_axis_coverage_loss(predicted, membership, target, valid)
    order = torch.tensor([2, 0, 1])
    right = membership_axis_coverage_loss(predicted[:, order], membership[:, order], target, valid)
    assert torch.isfinite(left)
    assert torch.equal(left, right)


def test_correct_membership_improves_coverage_but_wrong_axis_does_not():
    predicted = _orientation([[1, 0], [-1, 0]])  # doubled-angle orthogonal axes
    target = _orientation([[1, 0]])
    valid = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    weak = torch.tensor([[[[0.2]], [[0.9]]]])
    strong = torch.tensor([[[[0.9]], [[0.9]]]])
    assert membership_axis_coverage_loss(predicted, strong, target, valid) < membership_axis_coverage_loss(predicted, weak, target, valid)
    wrong_only = torch.tensor([[[[0.0]], [[0.99]]]])
    assert membership_axis_coverage_loss(predicted, wrong_only, target, valid) > 5.0


def test_x_crossing_needs_two_compatible_active_modes():
    predicted = _orientation([[1, 0], [-1, 0]])
    target = _orientation([[1, 0], [-1, 0]])
    valid = torch.ones(1, 2, 1, 1, dtype=torch.bool)
    one_active = torch.tensor([[[[0.95]], [[0.05]]]])
    two_active = torch.tensor([[[[0.95]], [[0.95]]]])
    assert membership_axis_coverage_loss(predicted, two_active, target, valid) < membership_axis_coverage_loss(predicted, one_active, target, valid)


def test_background_loss_decreases_with_memberships_and_count_ignores_background():
    valid = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    high = torch.full((1, 3, 2, 2), 0.8)
    low = torch.full((1, 3, 2, 2), 0.1)
    assert background_membership_loss(low, valid) < background_membership_loss(high, valid)
    count = torch.zeros(1, 2, 2)
    assert positive_exact_mode_count_loss(high, count).item() == 0.0


def test_positive_count_is_minimized_at_exact_fuzzy_mass():
    count = torch.tensor([[[2.0]]])
    exact = torch.tensor([[[[0.8]], [[0.7]], [[0.5]]]])
    low = torch.tensor([[[[0.2]], [[0.3]], [[0.1]]]])
    assert positive_exact_mode_count_loss(exact, count).item() == 0.0
    assert positive_exact_mode_count_loss(low, count) > 0


def test_rc1_loss_backpropagates_finitely_to_membership():
    predicted = _orientation([[1, 0], [-1, 0]])
    membership = torch.full((1, 2, 1, 2), 0.4, requires_grad=True)
    target = torch.zeros(1, 2, 2, 1, 2)
    target[:, 0, 0] = 1.0
    valid = torch.zeros(1, 2, 1, 2, dtype=torch.bool); valid[:, 0, 0, 0] = True
    count = valid.sum(dim=1).float()
    loss, terms = rc1_membership_loss(predicted.expand(-1, -1, -1, -1, 2), membership, target, valid, count, lambda_bg=0.25)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(value) for value in terms.values())
    assert membership.grad is not None and torch.isfinite(membership.grad).all()
