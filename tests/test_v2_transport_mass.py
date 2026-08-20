from __future__ import annotations

import torch

from models.azconv_v2 import AZConvV2Config, ModeResolvedAZConv2d, paired_sigmas


def test_v2a_transport_is_nonnegative_and_normalized_per_destination_mode() -> None:
    torch.manual_seed(41)
    layer = ModeResolvedAZConv2d(
        3,
        4,
        cfg=AZConvV2Config(num_modes=4, state_channels=5, transport_steps=1),
    )
    diagnostics = layer(torch.randn(2, 3, 7, 9), return_diagnostics=True)
    transport = diagnostics["transport"]
    assert transport.shape == (2, 4, 4, 9, 63)
    assert torch.isfinite(transport).all()
    assert torch.all(transport >= 0)
    expected = torch.ones_like(diagnostics["transport_mass"])
    assert torch.allclose(diagnostics["transport_mass"], expected, atol=1e-5, rtol=1e-5)


def test_paired_scales_preserve_local_determinant_and_isotropic_limit() -> None:
    base = torch.tensor([0.5, 1.0, 2.0])
    hyperbolicity = torch.tensor([0.0, 0.3, 0.9])
    sigma_u, sigma_s = paired_sigmas(base, hyperbolicity)
    assert torch.allclose(sigma_u * sigma_s, base.square(), atol=1e-6)
    iso_u, iso_s = paired_sigmas(base, torch.zeros_like(base))
    assert torch.allclose(iso_u, base)
    assert torch.allclose(iso_s, base)


def test_v2b_transport_is_row_stochastic_per_valid_source_half_mode() -> None:
    torch.manual_seed(45)
    layer = ModeResolvedAZConv2d(
        2,
        2,
        cfg=AZConvV2Config(
            num_modes=3,
            state_channels=3,
            transport_steps=1,
            variant="v2b",
        ),
    )
    diagnostics = layer(torch.randn(2, 2, 6, 7), return_diagnostics=True)
    source_mass = diagnostics["transport_mass"]
    assert source_mass.shape == (2, 3, 2, 6, 7)
    assert torch.isfinite(source_mass).all()
    assert torch.allclose(source_mass, torch.ones_like(source_mass), atol=2e-5, rtol=2e-5)
