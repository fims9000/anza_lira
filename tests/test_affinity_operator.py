import torch
import torch.nn as nn

from models.azconv import AZConvConfig
from models.azconv_affinity import (
    AffinityAZConvConfig,
    IndependentFuzzyAZConv2d,
    StructuralAffinityAZConv2d,
)
from synthetic.affinity_losses import balanced_affinity_bce, configure_affinity_stage1
from affinity_repair.training import project_affinity_constraints


def _layers():
    cfg = AZConvConfig(geometry_mode="local_hyperbolic", normalize_mode="global")
    clean = IndependentFuzzyAZConv2d(3, 5, num_rules=4, cfg=cfg)
    affinity = StructuralAffinityAZConv2d(
        3, 5, num_rules=4, cfg=cfg, affinity_cfg=AffinityAZConvConfig(context_channels=4)
    )
    affinity.load_state_dict(clean.state_dict(), strict=False)
    return clean, affinity


def test_beta_zero_is_exact_clean_v1_equivalence_and_causal_off():
    torch.manual_seed(4)
    clean, affinity = _layers()
    x = torch.randn(2, 3, 17, 19)
    affinity.raw_beta.data.zero_()
    expected = clean(x)
    actual = affinity(x)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    affinity.raw_beta.data.fill_(0.7)
    affinity.set_affinity_enabled(False)
    torch.testing.assert_close(affinity(x), expected, rtol=0.0, atol=0.0)


def test_independent_memberships_are_not_simplex_and_weights_normalize():
    torch.manual_seed(5)
    clean, affinity = _layers()
    x = torch.randn(2, 3, 13, 11)
    clean_diag = clean(x, return_diagnostics=True)
    assert not torch.allclose(clean_diag["memberships"].sum(1), torch.ones_like(x[:, 0]))
    affinity.raw_beta.data.fill_(0.25)
    diag = affinity(x, return_diagnostics=True)
    torch.testing.assert_close(
        diag["weights"].sum((1, 2)), torch.ones_like(diag["weights"][:, 0, 0]), atol=1e-6, rtol=1e-6
    )


def test_affinity_is_symmetric_for_reverse_edge_interior():
    torch.manual_seed(6)
    _clean, affinity = _layers()
    x = torch.randn(1, 3, 15, 15)
    forward, _ = affinity._pair_scores(x, ((1, 0),))
    reverse, _ = affinity._pair_scores(x, ((-1, 0),))
    torch.testing.assert_close(forward[:, :, 0, :, :-1], reverse[:, :, 0, :, 1:], atol=2e-6, rtol=2e-6)


def test_affinity_pair_features_are_axial_under_theta_plus_pi():
    torch.manual_seed(61)
    _clean, affinity = _layers()
    x = torch.randn(1, 3, 11, 13)
    before, _ = affinity._pair_scores(x, ((1, 0), (0, 1)))
    assert affinity.geometry_conv is not None
    with torch.no_grad():
        affinity.geometry_conv.bias[: affinity.R].add_(torch.pi)
    after, _ = affinity._pair_scores(x, ((1, 0), (0, 1)))
    torch.testing.assert_close(after, before, atol=2e-6, rtol=2e-6)


def test_affinity_loss_has_finite_gradients_and_stage1_cannot_update_base():
    torch.manual_seed(7)
    _clean, affinity = _layers()
    model = nn.Sequential(affinity, nn.ReLU())
    trainable, frozen = configure_affinity_stage1(model)
    assert trainable and frozen
    before = [parameter.detach().clone() for parameter in frozen]
    x = torch.randn(2, 3, 12, 12)
    edges = affinity.edge_logits(x)["logits"]
    positive = torch.zeros_like(edges, dtype=torch.bool)
    negative = torch.zeros_like(edges, dtype=torch.bool)
    positive[:, 0, 3:8, 3:8] = True
    negative[:, 1, 3:8, 3:8] = True
    loss = balanced_affinity_bce(edges, positive, negative)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is None for parameter in frozen)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in trainable)
    optimizer = torch.optim.SGD(trainable, lr=0.01)
    optimizer.step()
    for old, parameter in zip(before, frozen):
        torch.testing.assert_close(old, parameter)


def test_radius2_field_is_used_causally_but_beta_zero_remains_exact():
    torch.manual_seed(71)
    cfg = AZConvConfig(geometry_mode="local_hyperbolic", normalize_mode="global")
    c2 = StructuralAffinityAZConv2d(
        3, 5, num_rules=4, cfg=cfg,
        affinity_cfg=AffinityAZConvConfig(context_channels=4, use_radius2_supervision=False),
    )
    c3 = StructuralAffinityAZConv2d(
        3, 5, num_rules=4, cfg=cfg,
        affinity_cfg=AffinityAZConvConfig(context_channels=4, use_radius2_supervision=True),
    )
    c3.load_state_dict(c2.state_dict(), strict=False)
    x = torch.randn(1, 3, 15, 15)
    c2.raw_beta.data.zero_()
    c3.raw_beta.data.zero_()
    torch.testing.assert_close(c3(x), c2(x), rtol=0.0, atol=0.0)
    c2.raw_beta.data.fill_(0.5)
    c3.raw_beta.data.fill_(0.5)
    assert not torch.allclose(c3(x), c2(x))
    diagnostics = c3(x, return_diagnostics=True)
    assert diagnostics["affinity"]["radius2_logits_by_rule"] is not None
    direct = c3.edge_logits(x, include_radius2=True)
    torch.testing.assert_close(diagnostics["radius2_affinity"]["logits"], direct["logits"])


def test_beta_cannot_invert_supervised_affinity_semantics():
    _clean, affinity = _layers()
    assert float(affinity.beta.detach()) == 0.0
    affinity.raw_beta.data.fill_(-0.5)
    assert float(affinity.beta.detach()) == 0.0
    project_affinity_constraints(affinity)
    assert float(affinity.raw_beta.detach()) == 0.0
    affinity.raw_beta.data.fill_(0.5)
    assert float(affinity.beta.detach()) > 0.0
