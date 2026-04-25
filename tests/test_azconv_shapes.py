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
        AZConvConfig(True, True, True),
        AZConvConfig(False, True, True),
        AZConvConfig(True, False, True),
        AZConvConfig(True, True, False),
    ]:
        m = AZConv2d(4, 8, kernel_size=3, num_rules=3, cfg=cfg)
        y = m(x)
        assert y.shape == (2, 8, 16, 16)
        assert torch.isfinite(y).all()


def test_non_fuzzy_snapshot_has_uniform_memberships_and_normalized_compatibility():
    x = torch.randn(1, 4, 12, 12)
    layer = AZConv2d(
        4,
        8,
        kernel_size=3,
        num_rules=4,
        cfg=AZConvConfig(
            use_fuzzy=False,
            use_anisotropy=True,
            learn_directions=False,
            geometry_mode="fixed_cat_map",
            normalize_kernel=True,
        ),
    )

    _ = layer(x)
    snapshot = layer.interpretation_snapshot()
    mu_rule_mean = snapshot["mu_rule_mean"].float()
    compat_map = snapshot["compat_map"].float()

    assert torch.allclose(mu_rule_mean, torch.full_like(mu_rule_mean, 1.0 / layer.R), atol=1e-5)
    assert abs(float(compat_map.sum().item()) - 1.0) < 1e-3


def test_isotropic_ablation_reports_isotropic_snapshot_mode():
    x = torch.randn(1, 4, 10, 10)
    layer = AZConv2d(
        4,
        8,
        kernel_size=3,
        num_rules=3,
        cfg=AZConvConfig(
            use_fuzzy=True,
            use_anisotropy=False,
            learn_directions=True,
            geometry_mode="local_hyperbolic",
            normalize_kernel=True,
        ),
    )

    _ = layer(x)
    snapshot = layer.interpretation_snapshot()

    assert snapshot["geometry_mode"] == "isotropic"
    assert snapshot["regularization"]["anisotropy_gap"] == 0.0


def test_azconv_backward_propagates_finite_gradients():
    x = torch.randn(2, 4, 14, 18, requires_grad=True)
    layer = AZConv2d(
        4,
        7,
        kernel_size=3,
        num_rules=3,
        cfg=AZConvConfig(
            use_fuzzy=True,
            use_anisotropy=True,
            learn_directions=True,
            geometry_mode="local_hyperbolic",
            normalize_kernel=True,
        ),
    )

    loss = layer(x).square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.gate_conv.weight.grad is not None
    assert torch.isfinite(layer.gate_conv.weight.grad).all()
    assert layer.geometry_conv is not None
    assert layer.geometry_conv.weight.grad is not None
    assert torch.isfinite(layer.geometry_conv.weight.grad).all()
    assert layer.pointwise.weight.grad is not None
    assert torch.isfinite(layer.pointwise.weight.grad).all()
    if isinstance(layer.value_conv, torch.nn.Conv2d):
        assert layer.value_conv.weight.grad is not None
        assert torch.isfinite(layer.value_conv.weight.grad).all()


def test_local_hyperbolic_forward_and_snapshot_stay_finite():
    x = torch.randn(2, 5, 17, 19)
    layer = AZConv2d(
        5,
        6,
        kernel_size=3,
        num_rules=4,
        cfg=AZConvConfig(
            use_fuzzy=True,
            use_anisotropy=True,
            learn_directions=True,
            geometry_mode="local_hyperbolic",
            normalize_kernel=True,
        ),
    )

    y = layer(x)
    snapshot = layer.interpretation_snapshot()

    assert torch.isfinite(y).all()
    for key in ("mu_map", "kernel_map", "compat_map", "theta_map", "hyper_map", "sigma_u_map", "sigma_s_map"):
        assert key in snapshot
        assert torch.isfinite(snapshot[key].float()).all(), key


def test_compatibility_floor_does_not_weight_padded_neighbors():
    x = torch.ones(1, 1, 5, 5)
    layer = AZConv2d(
        1,
        1,
        kernel_size=3,
        num_rules=4,
        cfg=AZConvConfig(
            use_fuzzy=True,
            use_anisotropy=True,
            learn_directions=True,
            geometry_mode="local_hyperbolic",
            normalize_kernel=True,
            normalize_mode="global",
            compatibility_floor=5e-4,
        ),
        bias=False,
    )
    with torch.no_grad():
        layer.value_conv.weight.fill_(1.0)
        layer.pointwise.weight.fill_(1.0)
        layer.gate_conv.weight.zero_()
        layer.gate_conv.bias.zero_()

    y = layer(x)

    assert torch.allclose(y, torch.ones_like(y), atol=1e-5)


def test_local_hyperbolic_pair_geometry_is_symmetric_for_opposite_neighbor_views():
    x = torch.randn(1, 3, 5, 6)
    layer = AZConv2d(
        3,
        4,
        kernel_size=3,
        num_rules=3,
        cfg=AZConvConfig(
            use_fuzzy=True,
            use_anisotropy=True,
            learn_directions=True,
            geometry_mode="local_hyperbolic",
            normalize_kernel=True,
        ),
    )

    kern, _gap, _smoothness, _interp = layer._local_hyperbolic_kernel(x)
    width = x.shape[-1]
    loc_left = 2 * width + 2
    loc_right = 2 * width + 3
    right_offset = 5
    left_offset = 3

    pair_lr = kern[0, :, right_offset, loc_left]
    pair_rl = kern[0, :, left_offset, loc_right]

    assert torch.allclose(pair_lr, pair_rl, atol=1e-5)


def test_ablations_disable_unused_submodules():
    x = torch.randn(1, 4, 12, 12)

    no_fuzzy = AZConv2d(
        4,
        5,
        kernel_size=3,
        num_rules=3,
        cfg=AZConvConfig(
            use_fuzzy=False,
            use_anisotropy=True,
            learn_directions=False,
            geometry_mode="fixed_cat_map",
            normalize_kernel=True,
        ),
    )
    y0 = no_fuzzy(x)
    with torch.no_grad():
        no_fuzzy.gate_conv.weight.normal_(mean=0.0, std=10.0)
        no_fuzzy.gate_conv.bias.uniform_(-50.0, 50.0)
    y1 = no_fuzzy(x)
    assert torch.allclose(y0, y1, atol=1e-6)

    no_aniso = AZConv2d(
        4,
        5,
        kernel_size=3,
        num_rules=3,
        cfg=AZConvConfig(
            use_fuzzy=True,
            use_anisotropy=False,
            learn_directions=True,
            geometry_mode="local_hyperbolic",
            normalize_kernel=True,
        ),
    )
    z0 = no_aniso(x)
    with torch.no_grad():
        assert no_aniso.geometry_conv is not None
        no_aniso.geometry_conv.weight.normal_(mean=0.0, std=10.0)
        no_aniso.geometry_conv.bias.uniform_(-50.0, 50.0)
    z1 = no_aniso(x)
    assert torch.allclose(z0, z1, atol=1e-6)

