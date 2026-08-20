import math

import torch

from models.azconv import AZConvConfig
from models.azconv_clean import CleanANZA2d


def test_clean_anza_positive_weights_normalize_and_gradients_are_finite():
    torch.manual_seed(101)
    layer = CleanANZA2d(3, 5, num_rules=4, cfg=AZConvConfig(normalize_mode="global"))
    x = torch.randn(2, 3, 13, 15, requires_grad=True)
    diagnostics = layer(x, return_diagnostics=True)
    weights = diagnostics["weights"]
    assert torch.all(weights >= 0)
    torch.testing.assert_close(weights.sum((1, 2)), torch.ones_like(weights[:, 0, 0]), atol=1e-6, rtol=1e-6)
    diagnostics["output"].square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in layer.parameters())


def test_independent_memberships_can_coexist_above_half():
    layer = CleanANZA2d(1, 2, num_rules=4)
    with torch.no_grad():
        layer.gate_conv.weight.zero_()
        layer.gate_conv.bias.copy_(torch.tensor([2.0, 1.0, -1.0, -2.0]))
    memberships = layer(torch.ones(1, 1, 7, 7), return_diagnostics=True)["memberships"]
    assert torch.all(memberships[:, 0] > 0.5)
    assert torch.all(memberships[:, 1] > 0.5)
    assert not torch.allclose(memberships.sum(1), torch.ones_like(memberships[:, 0]))


def test_clean_anza_is_axial_and_has_isotropic_limit():
    torch.manual_seed(102)
    first = CleanANZA2d(3, 4, num_rules=4).eval()
    second = CleanANZA2d(3, 4, num_rules=4).eval()
    second.load_state_dict(first.state_dict())
    with torch.no_grad():
        second.geometry_conv.bias[: second.R].add_(math.pi)
    image = torch.randn(2, 3, 11, 13)
    torch.testing.assert_close(first(image), second(image), atol=2e-6, rtol=1e-6)
    isotropic = CleanANZA2d(3, 4, num_rules=4, cfg=AZConvConfig(use_anisotropy=False))
    kernel = isotropic._isotropic_kernel(torch.device("cpu"))[0]
    torch.testing.assert_close(kernel[:, 1], kernel[:, 3])
    torch.testing.assert_close(kernel[:, 1], kernel[:, 5])
    torch.testing.assert_close(kernel[:, 1], kernel[:, 7])


def test_active_mode_count_does_not_attenuate_constant_aggregate_amplitude():
    layer = CleanANZA2d(1, 1, num_rules=4, cfg=AZConvConfig(normalize_mode="global"))
    image = torch.ones(1, 1, 9, 9)
    aggregates = []
    for biases in ((20.0, -20.0, -20.0, -20.0), (20.0, 20.0, 20.0, 20.0)):
        with torch.no_grad():
            layer.gate_conv.weight.zero_()
            layer.gate_conv.bias.copy_(torch.tensor(biases))
        weights = layer(image, return_diagnostics=True)["weights"]
        constant_value_aggregate = weights.sum((1, 2))
        aggregates.append(constant_value_aggregate)
    torch.testing.assert_close(aggregates[0], aggregates[1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(aggregates[0], torch.ones_like(aggregates[0]), atol=1e-6, rtol=1e-6)
