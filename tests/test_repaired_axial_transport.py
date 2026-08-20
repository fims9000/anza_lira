from __future__ import annotations

import math

import torch

from models.azconv_repaired import AmbiguityGatedResidualANZA


def test_initial_state_uses_membership_once_and_sum_preserves_value() -> None:
    torch.manual_seed(20)
    operator = AmbiguityGatedResidualANZA(3, 4).eval()
    sample = torch.randn(1, 3, 8, 9)
    diagnostics = operator(sample, return_diagnostics=True)
    value = operator.value_projection(sample)
    assert diagnostics["initial_state"].shape[1] == operator.modes
    assert torch.allclose(diagnostics["initial_state"].sum(dim=1), value)


def test_transition_is_invariant_to_axial_pi_shift() -> None:
    torch.manual_seed(21)
    operator = AmbiguityGatedResidualANZA(2, 3).eval()
    sample = torch.randn(1, 2, 7, 8)
    geometry = operator.geometry(sample)
    shifted = {**geometry, "theta": geometry["theta"] + math.pi}
    first, first_mass = operator._transition(geometry)
    second, second_mass = operator._transition(shifted)
    assert torch.allclose(first, second, atol=2e-6, rtol=2e-6)
    assert torch.allclose(first_mass, second_mass, atol=2e-6, rtol=2e-6)


def test_transition_is_source_row_stochastic_without_half_modes() -> None:
    torch.manual_seed(22)
    operator = AmbiguityGatedResidualANZA(2, 3).eval()
    diagnostics = operator(torch.randn(1, 2, 9, 10), return_diagnostics=True)
    assert diagnostics["transport"].ndim == 5
    assert torch.allclose(
        diagnostics["transport_source_mass"],
        torch.ones_like(diagnostics["transport_source_mass"]),
        atol=2e-5,
        rtol=2e-5,
    )


def test_ranges_are_finite() -> None:
    operator = AmbiguityGatedResidualANZA(2, 3).eval()
    diagnostics = operator(torch.randn(1, 2, 7, 7), return_diagnostics=True)
    for name in ("membership", "ambiguity", "ambiguity_gate", "transport", "transport_source_mass"):
        value = diagnostics[name]
        assert torch.isfinite(value).all(), name
        assert torch.all(value >= 0), name
        assert torch.all(value <= 1 + 2e-5), name
