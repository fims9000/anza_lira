from __future__ import annotations

import torch

from synthetic.crossing_trace_bench import generate_sample
from models.segmentation_v2 import build_comparable_model
from synthetic.structural_losses import branch_transition_logits, routing_supervision_loss


def _loss(sample: dict, logits: torch.Tensor) -> torch.Tensor:
    return routing_supervision_loss(
        logits,
        torch.as_tensor(sample["continuation_relation_matrix"]),
        torch.as_tensor(sample["continuation_eligible_matrix"]),
    )


def test_lineage_correct_x_routing_scores_better_than_wrong_straightest_pair() -> None:
    sample = generate_sample("validation", 130, case="nontrivial_pairing")
    target = torch.as_tensor(sample["continuation_relation_matrix"])
    eligible = torch.as_tensor(sample["continuation_eligible_matrix"])
    correct = torch.where(target, torch.tensor(6.0), torch.tensor(-6.0))
    wrong = torch.full_like(correct, -6.0)
    wrong[0, 2] = wrong[2, 0] = 6.0  # minimum-angle but wrong lineage
    wrong[1, 3] = wrong[3, 1] = 6.0
    assert _loss(sample, correct) < _loss(sample, wrong)
    assert torch.all(target <= eligible)


def test_y_branch_supports_one_to_many_continuation_targets() -> None:
    sample = generate_sample("validation", 131, case="y_junction")
    target = torch.as_tensor(sample["continuation_relation_matrix"])
    logits = torch.where(target, torch.tensor(5.0), torch.tensor(-5.0)).requires_grad_()
    loss = _loss(sample, logits)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert int(target[0].sum()) == 2


def test_routing_loss_is_invariant_to_consistent_branch_permutation() -> None:
    sample = generate_sample("validation", 132, case="curved_crossing")
    target = torch.as_tensor(sample["continuation_relation_matrix"])
    eligible = torch.as_tensor(sample["continuation_eligible_matrix"])
    logits = torch.arange(target.numel(), dtype=torch.float32).reshape_as(target)
    permutation = torch.tensor([2, 0, 3, 1])
    original = routing_supervision_loss(logits, target, eligible)
    permuted = routing_supervision_loss(
        logits[permutation][:, permutation],
        target[permutation][:, permutation],
        eligible[permutation][:, permutation],
    )
    assert torch.allclose(original, permuted)


def test_case_without_junction_has_zero_differentiable_route_loss() -> None:
    sample = generate_sample("validation", 133, case="single_straight")
    logits = torch.zeros(1, 1, requires_grad=True)
    loss = _loss(sample, logits)
    loss.backward()
    assert loss.item() == 0.0
    assert logits.grad is not None and logits.grad.item() == 0.0


def test_pixel_transport_aggregates_to_finite_branch_logits_with_gradients() -> None:
    sample = generate_sample("train", 134, image_size=16, case="nontrivial_pairing")
    model = build_comparable_model("anza_v2b", widths=(4, 8, 12, 16))
    diagnostics = model(
        torch.as_tensor(sample["image"]).unsqueeze(0),
        return_diagnostics=True,
    )
    first = diagnostics["transport_diagnostics"][0]
    logits = branch_transition_logits(
        first["transport"],
        torch.as_tensor(sample["branch_masks"]),
        variant=first["variant"],
    )
    loss = _loss(sample, logits)
    loss.backward()
    assert logits.shape == sample["continuation_relation_matrix"].shape
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
