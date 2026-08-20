from __future__ import annotations

import math

import torch

from models.azconv_v2 import junction_score


def _score(membership: list[float], theta: list[float]) -> float:
    membership_tensor = torch.tensor(membership, dtype=torch.float32).view(1, -1, 1, 1)
    theta_tensor = torch.tensor(theta, dtype=torch.float32).view(1, -1, 1, 1)
    return float(junction_score(membership_tensor, theta_tensor).item())


def test_one_dominant_mode_has_low_junction_score() -> None:
    assert _score([1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 1.0, 1.5]) == 0.0


def test_two_strong_parallel_axial_modes_have_low_junction_score() -> None:
    score = _score([0.5, 0.5, 0.0, 0.0], [0.2, 0.2 + math.pi, 0.7, 1.2])
    assert score < 1e-10


def test_two_strong_orthogonal_modes_have_high_junction_score() -> None:
    score = _score([0.5, 0.5, 0.0, 0.0], [0.0, math.pi / 2, 0.2, 0.4])
    assert score > 0.60


def test_three_separated_orientations_have_high_junction_score() -> None:
    score = _score([1 / 3, 1 / 3, 1 / 3], [0.0, math.pi / 3, 2 * math.pi / 3])
    assert score > 0.70


def test_junction_score_is_finite_and_bounded_for_random_fields() -> None:
    torch.manual_seed(47)
    membership = torch.softmax(torch.randn(2, 5, 7, 9), dim=1)
    theta = torch.randn_like(membership)
    score = junction_score(membership, theta)
    assert torch.isfinite(score).all()
    assert torch.all((score >= 0) & (score <= 1))
