from __future__ import annotations

import math

import torch

from models.azconv_repaired import (
    AmbiguityGatedResidualANZA,
    RepairedAZConvConfig,
    ambiguity_components,
)


def test_one_active_mode_has_zero_ambiguity() -> None:
    membership = torch.tensor([[[[1.0]], [[0.0]], [[0.0]], [[0.0]]]])
    theta = torch.tensor([[[[0.0]], [[0.5]], [[1.0]], [[1.5]]]])
    diversity, angular, ambiguity = ambiguity_components(membership, theta)
    assert diversity.item() == 0.0
    assert ambiguity.item() == 0.0
    assert 0.0 <= angular.item() <= 1.0


def test_two_orthogonal_modes_are_more_ambiguous_than_two_aligned_modes() -> None:
    membership = torch.tensor([[[[0.5]], [[0.5]], [[0.0]], [[0.0]]]])
    aligned_theta = torch.zeros_like(membership)
    crossing_theta = aligned_theta.clone()
    crossing_theta[:, 1] = math.pi / 2
    aligned = ambiguity_components(membership, aligned_theta)[2]
    crossing = ambiguity_components(membership, crossing_theta)[2]
    assert aligned.item() == 0.0
    assert crossing.item() > 0.5


def test_ambiguity_is_axial_pi_invariant() -> None:
    torch.manual_seed(30)
    membership = torch.softmax(torch.randn(2, 4, 5, 6), dim=1)
    theta = torch.randn(2, 4, 5, 6)
    first = ambiguity_components(membership, theta)[2]
    second = ambiguity_components(membership, theta + math.pi)[2]
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_no_gate_ablation_is_explicit_unit_gate() -> None:
    operator = AmbiguityGatedResidualANZA(
        2,
        3,
        cfg=RepairedAZConvConfig(use_ambiguity_gate=False),
    )
    diagnostics = operator(torch.randn(1, 2, 7, 8), return_diagnostics=True)
    assert torch.equal(
        diagnostics["ambiguity_gate"],
        torch.ones_like(diagnostics["ambiguity_gate"]),
    )
