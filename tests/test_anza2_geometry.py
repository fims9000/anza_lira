import math

import torch

from models.anza2.field import ANZA2FieldConfig, ANZA2FieldOutput, field_from_raw
from models.anza2.geometry import directed_geometry, hyperbolic_shape_matrix, quadratic_form


def _manual_field(angles, *, membership=1.0, scale=1.0, hyper=1.0):
    angles = torch.as_tensor(angles, dtype=torch.float64)
    modes = angles.numel()
    orientation = torch.stack((torch.cos(2 * angles), torch.sin(2 * angles)), dim=1).view(1, modes, 2, 1, 1)
    mu = torch.full((1, modes, 1, 1), float(membership), dtype=torch.float64)
    ell = torch.full_like(mu, float(scale))
    h = torch.full_like(mu, float(hyper))
    return ANZA2FieldOutput(mu, orientation, ell, h, ell * torch.exp(h), ell * torch.exp(-h))


def test_t1_axial_invariance_and_t2_unit_orientation() -> None:
    theta = torch.tensor([0.37], dtype=torch.float64)
    first = _manual_field(theta)
    second = _manual_field(theta + math.pi)
    torch.testing.assert_close(first.orientation, second.orientation, atol=1e-12, rtol=0)
    torch.testing.assert_close(first.orientation.square().sum(dim=2), torch.ones(1, 1, 1, 1, dtype=torch.float64))
    torch.testing.assert_close(directed_geometry(first, (1.0, 0.4)), directed_geometry(second, (1.0, 0.4)))


def test_t2_zero_raw_orientation_has_safe_unit_fallback() -> None:
    config = ANZA2FieldConfig(num_modes=2)
    raw = field_from_raw(
        torch.zeros(1, 2, 1, 1),
        torch.zeros(1, 2, 2, 1, 1),
        torch.zeros(1, 2, 1, 1),
        torch.zeros(1, 2, 1, 1),
        config=config,
    )
    torch.testing.assert_close(raw.orientation.square().sum(dim=2), torch.ones(1, 2, 1, 1))
    assert torch.equal(raw.orientation[:, :, 0], torch.ones(1, 2, 1, 1))


def test_t3_hyperbolic_shape_determinant_is_one() -> None:
    field = _manual_field([0.0, 0.3, 1.2], hyper=1.25)
    matrix = hyperbolic_shape_matrix(field)
    determinant = torch.linalg.det(matrix)
    torch.testing.assert_close(determinant, torch.ones_like(determinant), atol=1e-12, rtol=1e-12)


def test_t4_isotropic_limit_is_radial_and_t5_prefers_longitudinal_steps() -> None:
    isotropic = _manual_field([0.0], hyper=0.0, scale=2.0)
    torch.testing.assert_close(directed_geometry(isotropic, (1.0, 0.0)), directed_geometry(isotropic, (0.0, 1.0)))
    elongated = _manual_field([0.0], hyper=1.0, scale=1.0)
    parallel = directed_geometry(elongated, (1.0, 0.0))
    perpendicular = directed_geometry(elongated, (0.0, 1.0))
    assert float(parallel) > float(perpendicular)


def test_exact_hand_computed_quadratic_form_contains_literal_half_gaussian() -> None:
    field = _manual_field([0.0], hyper=0.0, scale=2.0)
    q = quadratic_form(field, (2.0, 0.0))
    torch.testing.assert_close(q, torch.ones_like(q))
    torch.testing.assert_close(directed_geometry(field, (2.0, 0.0)), torch.full_like(q, math.exp(-0.5)))
