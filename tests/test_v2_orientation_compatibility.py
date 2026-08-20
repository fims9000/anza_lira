from __future__ import annotations

import math

import torch

from models.azconv_v2 import orientation_compatibility


def test_orientation_compatibility_is_axial_and_suppresses_orthogonal_modes() -> None:
    theta = torch.tensor(0.37)
    same = orientation_compatibility(theta, theta, 4.0)
    opposite = orientation_compatibility(theta, theta + math.pi, 4.0)
    orthogonal = orientation_compatibility(theta, theta + math.pi / 2, 4.0)
    assert torch.allclose(same, torch.tensor(1.0))
    assert torch.allclose(opposite, torch.tensor(1.0), atol=1e-6)
    assert orthogonal < 0.02


def test_zero_kappa_disables_orientation_routing() -> None:
    first = torch.randn(3, 4)
    second = torch.randn(3, 4)
    assert torch.allclose(orientation_compatibility(first, second, 0.0), torch.ones_like(first))
