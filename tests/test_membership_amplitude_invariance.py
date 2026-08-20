from __future__ import annotations

import pytest
import torch

from method_repair.audit import current_membership_gain, repaired_membership_gain


@pytest.mark.parametrize(
    ("membership", "expected_v2a", "expected_v2b"),
    [
        ([0.25, 0.25, 0.25, 0.25], 0.25, 0.125),
        ([1.0, 0.0, 0.0, 0.0], 1.0, 0.5),
        ([0.7, 0.1, 0.1, 0.1], 0.52, 0.26),
    ],
)
def test_frozen_v2_gain_depends_on_membership_entropy(
    membership: list[float], expected_v2a: float, expected_v2b: float
) -> None:
    mu = torch.tensor(membership)
    assert float(current_membership_gain(mu, variant="v2a")) == pytest.approx(expected_v2a)
    assert float(current_membership_gain(mu, variant="v2b")) == pytest.approx(expected_v2b)


def test_repaired_single_gate_sum_fusion_is_amplitude_invariant() -> None:
    memberships = torch.tensor(
        [
            [0.25, 1.0, 0.7],
            [0.25, 0.0, 0.1],
            [0.25, 0.0, 0.1],
            [0.25, 0.0, 0.1],
        ]
    )
    assert torch.allclose(repaired_membership_gain(memberships), torch.ones(3))


def test_invalid_membership_fails_closed() -> None:
    with pytest.raises(ValueError, match="positive mass"):
        repaired_membership_gain(torch.zeros(4))
    with pytest.raises(ValueError, match="finite and non-negative"):
        repaired_membership_gain(torch.tensor([1.0, float("nan")]))
