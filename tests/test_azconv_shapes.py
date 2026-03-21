"""Shape and finiteness checks for AZConv2d (pytest)."""

import torch

from models.azconv import AZConv2d, AZConvConfig


def test_azconv_output_shape():
    device = torch.device("cpu")
    B, C, H, W, R, k, out_c = 2, 8, 32, 32, 4, 3, 16
    x = torch.randn(B, C, H, W, device=device)
    layer = AZConv2d(C, out_c, kernel_size=k, num_rules=R).to(device)
    y = layer(x)
    assert y.shape == (B, out_c, H, W)


def test_unfold_matches_expected_length():
    B, C, H, W, k = 1, 3, 32, 32, 3
    S, L = k * k, H * W
    x = torch.randn(B, C, H, W)
    u = torch.nn.functional.unfold(x, k, padding=k // 2, stride=1)
    assert u.shape == (B, C * S, L)


def test_ablations_run():
    x = torch.randn(2, 4, 16, 16)
    for cfg in [
        AZConvConfig(use_fuzzy=True, use_anisotropy=True, learn_directions=True, geometry_mode="local_hyperbolic"),
        AZConvConfig(use_fuzzy=False, use_anisotropy=True, learn_directions=True, geometry_mode="local_hyperbolic"),
        AZConvConfig(use_fuzzy=True, use_anisotropy=False, learn_directions=True, geometry_mode="local_hyperbolic"),
        AZConvConfig(use_fuzzy=True, use_anisotropy=True, learn_directions=False, geometry_mode="learned_angle"),
        AZConvConfig(use_fuzzy=True, use_anisotropy=True, learn_directions=False, geometry_mode="fixed_cat_map"),
        AZConvConfig(use_fuzzy=True, use_anisotropy=True, learn_directions=True, geometry_mode="learned_hyperbolic"),
    ]:
        m = AZConv2d(4, 8, kernel_size=3, num_rules=3, cfg=cfg)
        y = m(x)
        assert y.shape == (2, 8, 16, 16)
        assert torch.isfinite(y).all()


def test_regularization_terms_are_finite():
    x = torch.randn(2, 4, 16, 16)
    m = AZConv2d(
        4,
        8,
        kernel_size=3,
        num_rules=3,
        cfg=AZConvConfig(use_fuzzy=True, use_anisotropy=True, geometry_mode="local_hyperbolic"),
    )
    _ = m(x)
    terms = m.regularization_terms()
    expected = {
        "membership_entropy",
        "membership_smoothness",
        "geometry_smoothness",
        "hyperbolicity_penalty",
        "anisotropy_gap",
    }
    assert expected.issubset(set(terms.keys()))
    for value in terms.values():
        assert torch.isfinite(value).all()


def test_interpretation_snapshot_contains_kernel_and_memberships():
    x = torch.randn(1, 4, 16, 16)
    m = AZConv2d(
        4,
        8,
        kernel_size=3,
        num_rules=3,
        cfg=AZConvConfig(use_fuzzy=True, use_anisotropy=True, geometry_mode="fixed_cat_map", learn_directions=False),
    )
    _ = m(x)
    snapshot = m.interpretation_snapshot()
    assert snapshot["num_rules"] == 3
    assert snapshot["kernel_size"] == 3
    assert snapshot["mu_map"].shape == (3, 16, 16)
    assert snapshot["kernel_map"].shape == (3, 3, 3)
    assert snapshot["compat_map"].shape == (3, 3, 3)
    assert snapshot["u_vec"].shape == (3, 2)
    assert snapshot["s_vec"].shape == (3, 2)

