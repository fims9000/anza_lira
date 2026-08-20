import torch

from models.anza2.aggregation import aggregate_modes
from models.anza2.block import ANZA2Block
from models.anza2.field import ANZA2Field, ANZA2FieldConfig, ANZA2FieldOutput
from models.anza2.affinity import ANZA2StructuralAffinity


def _grid_field(membership: torch.Tensor) -> ANZA2FieldOutput:
    batch, modes, height, width = membership.shape
    orientation = torch.zeros(batch, modes, 2, height, width, dtype=membership.dtype)
    orientation[:, :, 0] = 1.0
    ell = torch.ones_like(membership)
    h = torch.ones_like(membership)
    return ANZA2FieldOutput(membership, orientation, ell, h, ell * torch.exp(h), ell * torch.exp(-h))


def test_t12_self_mass_fallback_preserves_center_feature() -> None:
    membership = torch.zeros(1, 1, 1, 3, dtype=torch.float64)
    membership[0, 0, 0, 1] = 0.7
    field = _grid_field(membership)
    values = torch.tensor([[[[2.0, 5.0, 11.0]]]], dtype=torch.float64)
    modes, self_mass, neighbor_mass = aggregate_modes(values, field, tau0=1.0, offsets=((1, 0),))
    torch.testing.assert_close(modes[0, 0, 0, 0, 1], torch.tensor(3.5, dtype=torch.float64))
    torch.testing.assert_close(self_mass[0, 0, 0, 1], torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(neighbor_mass[0, 0, 0, 0, 1], torch.tensor(0.0, dtype=torch.float64))


def test_t13_self_and_neighbor_mass_normalize_within_each_mode() -> None:
    torch.manual_seed(4)
    membership = torch.rand(2, 3, 5, 7, dtype=torch.float64)
    field = _grid_field(membership)
    values = torch.randn(2, 2, 5, 7, dtype=torch.float64)
    _modes, self_mass, neighbor_mass = aggregate_modes(values, field, tau0=0.5)
    torch.testing.assert_close(
        self_mass + neighbor_mass.sum(dim=2), torch.ones_like(self_mass), atol=1e-12, rtol=1e-12
    )


def test_residual_block_is_exact_baseline_at_zero_gamma() -> None:
    torch.manual_seed(7)
    block = ANZA2Block(4).eval()
    features = torch.randn(2, 4, 9, 11)
    torch.testing.assert_close(block(features), features, atol=0, rtol=0)


def test_t14_all_field_parameter_gradients_are_finite() -> None:
    torch.manual_seed(9)
    field_module = ANZA2Field(3, ANZA2FieldConfig(num_modes=4))
    features = torch.randn(2, 3, 8, 10, requires_grad=True)
    field = field_module(features)
    affinity = ANZA2StructuralAffinity()(field)
    modes, _self, _neighbor = aggregate_modes(features, field)
    loss = affinity.mean() + modes.square().mean()
    loss.backward()
    for name, parameter in field_module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
