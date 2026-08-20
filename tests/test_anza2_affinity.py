import math

import torch

from models.anza2.affinity import GenericAffinityCombiner, structural_affinity_pair
from models.anza2.field import ANZA2FieldOutput
from models.anza2.geometry import directed_step_support


def _field(angles, memberships, *, hyper=1.25, scale=0.75):
    angles = torch.tensor(angles, dtype=torch.float64)
    mu = torch.tensor(memberships, dtype=torch.float64).view(1, -1, 1, 1)
    orientation = torch.stack((torch.cos(2 * angles), torch.sin(2 * angles)), dim=1).view(1, -1, 2, 1, 1)
    ell = torch.full_like(mu, scale)
    h = torch.full_like(mu, hyper)
    return ANZA2FieldOutput(mu, orientation, ell, h, ell * torch.exp(h), ell * torch.exp(-h))


def _permute(field, order):
    return ANZA2FieldOutput(*(value[:, order] for value in (
        field.membership, field.orientation, field.base_scale, field.hyperbolicity,
        field.sigma_parallel, field.sigma_perpendicular,
    )))


def test_t6_mode_permutation_invariance() -> None:
    first = _field([0.0, math.pi / 2, math.pi / 4], [0.9, 0.8, 0.2])
    second = _field([0.1, 0.7, 1.2], [0.4, 0.95, 0.3])
    order = torch.tensor([2, 0, 1])
    torch.testing.assert_close(
        directed_step_support(first, (1.0, 0.0)),
        directed_step_support(_permute(first, order), (1.0, 0.0)),
    )
    torch.testing.assert_close(
        structural_affinity_pair(first, second, (1.0, 0.0)),
        structural_affinity_pair(_permute(first, order), _permute(second, order.flip(0)), (1.0, 0.0)),
    )


def test_t7_edge_symmetry_and_t8_affinity_range() -> None:
    first = _field([0.0, 0.7], [0.9, 0.4])
    second = _field([0.2, 1.1], [0.8, 0.6])
    forward = structural_affinity_pair(first, second, (1.0, 0.25))
    reverse = structural_affinity_pair(second, first, (-1.0, -0.25))
    torch.testing.assert_close(forward, reverse)
    assert bool(((0 <= forward) & (forward <= 1)).all())


def test_t9_parallel_line_killer_scores_along_trace_not_cross_trace() -> None:
    horizontal = _field([0.0], [0.99], hyper=1.25, scale=0.55)
    along = structural_affinity_pair(horizontal, horizontal, (1.0, 0.0))
    across = structural_affinity_pair(horizontal, horizontal, (0.0, 2.0))
    assert float(along) > 0.8
    assert float(across) < 0.05
    assert float(along) > 20.0 * float(across)


def test_t10_crossing_two_modes_are_simultaneously_supported_without_competition() -> None:
    crossing = _field([0.0, math.pi / 2], [0.9, 0.85])
    horizontal = directed_step_support(crossing, (1.0, 0.0))
    vertical = directed_step_support(crossing, (0.0, 1.0))
    assert float(horizontal) > 0.75
    assert float(vertical) > 0.70
    changed = _field([0.0, math.pi / 2], [0.9, 0.05])
    torch.testing.assert_close(horizontal, directed_step_support(changed, (1.0, 0.0)))


def test_t11_curved_chain_does_not_require_equal_mode_indices() -> None:
    first = _field([0.15, 1.2], [0.95, 0.1], hyper=0.8, scale=1.0)
    second = _field([1.1, 0.30], [0.1, 0.95], hyper=0.8, scale=1.0)
    aligned = structural_affinity_pair(first, second, (1.0, 0.22))
    cross = structural_affinity_pair(first, second, (-0.22, 1.0))
    assert float(aligned) > float(cross)
    assert float(aligned) > 0.65


def test_t15_beta_zero_is_generic_affinity_identity() -> None:
    generic = torch.tensor([[-3.0, 0.0, 2.5]], dtype=torch.float32)
    prior = torch.tensor([[0.01, 0.5, 0.99]], dtype=torch.float32)
    combiner = GenericAffinityCombiner(initial_beta=0.0)
    combined = combiner(generic, prior, beta_override=0.0)
    torch.testing.assert_close(combined, generic, atol=1e-6, rtol=0)
    assert float(combiner.beta.detach()) >= 0.0
