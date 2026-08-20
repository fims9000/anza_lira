import numpy as np
import pytest
import torch

from connectivity_repair.diffusion_oracle import (
    completion_gate_metrics,
    diffuse,
    diffusion_step,
    transition_from_anza_and_connectivity,
)
from models.azconv_affinity import LOCAL8_OFFSETS
from synthetic.affinity_targets import build_affinity_targets
from synthetic.crossing_trace_bench_v5 import generate_sample_v5
from synthetic.evaluation_corrected import evaluate_sample_corrected


def _uniform_raw(batch: int = 1, height: int = 9, width: int = 9) -> torch.Tensor:
    return torch.ones(batch, 2, 9, height * width)


def test_gt_connectivity_is_multi_directional_and_excludes_negative_gap() -> None:
    crossing = generate_sample_v5("validation", 300, image_size=64)
    negative = generate_sample_v5("validation", 128, image_size=64)
    crossing_target = build_affinity_targets(crossing, LOCAL8_OFFSETS)["affinity_positive"]
    negative_target = build_affinity_targets(negative, LOCAL8_OFFSETS)["affinity_positive"]
    assert crossing_target.sum(axis=0).max() >= 3
    assert not np.any(negative_target[:, np.asarray(negative["negative_gap_mask"], dtype=bool)])


def test_gt_connectivity_is_symmetric_under_reverse_edge_lookup() -> None:
    sample = generate_sample_v5("validation", 300, image_size=64)
    target = build_affinity_targets(sample, LOCAL8_OFFSETS)["affinity_positive"]
    reverse = {offset: LOCAL8_OFFSETS.index((-offset[0], -offset[1])) for offset in LOCAL8_OFFSETS}
    for channel, (dx, dy) in enumerate(LOCAL8_OFFSETS):
        opposite = target[reverse[(dx, dy)]]
        shifted = np.zeros_like(opposite)
        source_y = slice(max(0, dy), min(64, 64 + dy))
        source_x = slice(max(0, dx), min(64, 64 + dx))
        target_y = slice(max(0, -dy), min(64, 64 - dy))
        target_x = slice(max(0, -dx), min(64, 64 - dx))
        shifted[target_y, target_x] = opposite[source_y, source_x]
        assert np.array_equal(target[channel], shifted)


def test_transition_is_row_stochastic_and_finite() -> None:
    connectivity = torch.ones(2, 8, 9, 9)
    transition = transition_from_anza_and_connectivity(_uniform_raw(2), connectivity)
    assert torch.isfinite(transition).all()
    assert torch.allclose(transition.sum(dim=1), torch.ones(2, 9, 9), atol=1e-6)


def test_restart_diffusion_is_an_alpha_contraction() -> None:
    generator = torch.Generator().manual_seed(7)
    h0 = torch.rand(1, 1, 9, 9, generator=generator)
    first = torch.rand(1, 1, 9, 9, generator=generator)
    second = torch.rand(1, 1, 9, 9, generator=generator)
    transition = transition_from_anza_and_connectivity(_uniform_raw(), torch.ones(1, 8, 9, 9))
    alpha = 0.6
    left = diffusion_step(h0, first, transition, alpha=alpha)
    right = diffusion_step(h0, second, transition, alpha=alpha)
    assert torch.max(torch.abs(left - right)) <= alpha * torch.max(torch.abs(first - second)) + 1e-6


def test_t_step_support_grows_and_remains_finite() -> None:
    h0 = torch.zeros(1, 1, 9, 9)
    h0[0, 0, 4, 4] = 1.0
    transition = transition_from_anza_and_connectivity(_uniform_raw(), torch.ones(1, 8, 9, 9))
    one = diffuse(h0, transition, steps=1, alpha=0.8)
    four = diffuse(h0, transition, steps=4, alpha=0.8)
    assert one[0, 0, 4, 2] == 0
    assert four[0, 0, 4, 2] > 0
    assert torch.isfinite(four).all()


def test_transition_channel_uses_q_equals_p_plus_offset() -> None:
    h0 = torch.zeros(1, 1, 5, 5)
    state = torch.zeros_like(h0)
    state[0, 0, 2, 3] = 1.0
    transition = torch.zeros(1, 8, 5, 5)
    right_channel = LOCAL8_OFFSETS.index((1, 0))
    transition[:, right_channel] = 1.0
    result = diffusion_step(h0, state, transition, alpha=0.6)
    assert result[0, 0, 2, 2] == pytest.approx(0.6)


@pytest.mark.parametrize("index", [0, 128])
def test_completion_gate_endpoint_matches_corrected_evaluator(index: int) -> None:
    sample = generate_sample_v5("validation", index, image_size=64)
    prediction = np.asarray(sample["visible_fault_mask"], dtype=bool)
    compact = completion_gate_metrics(prediction[None].astype(np.float32), [sample], threshold=0.5)
    full = evaluate_sample_corrected(
        prediction, sample, predicted_completion_mask=prediction
    )["family_a"]
    assert compact["visible_dice"] == full["visible_dice"]
    assert compact["visible_cldice"] == full["visible_cldice"]
    assert compact["gap_recovery_rate"] == full["gap_recovery_rate"]
    assert compact["false_bridge_rate"] == full["false_bridge_rate"]
