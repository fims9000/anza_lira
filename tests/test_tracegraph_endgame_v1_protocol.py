from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from anza_tracegraph.endgame_v1.p0.corridor import branch_landing_corridor
from anza_tracegraph.endgame_v1.p0.dataset import STATUS_MISS, STATUS_NONE, STATUS_PRESENT, selected_training_ranks
from anza_tracegraph.endgame_v1.p0.legacy_loader import architecture_receipt, build_exact_p0
from anza_tracegraph.endgame_v1.p0.train import source_balanced_loss
from anza_tracegraph.endgame_v1.protocol import PROTOCOL
from anza_tracegraph.endgame_v1.split_data import SPLIT_SETTINGS, assert_seed_hygiene, generate_scene
from path_completion.pair_classifier import EndpointPairClassifier


class _Candidate:
    def __init__(self, score: float) -> None:
        self.geometric_score = score


def test_frozen_sbpp_and_fresh_split_contract() -> None:
    assert PROTOCOL["sbpp"]["tau_s"] == 0.20
    assert PROTOCOL["sbpp"]["candidate_k"] == 12
    assert_seed_hygiene()
    assert {row["size"] for row in SPLIT_SETTINGS.values()} == {19_200, 3_840}
    assert len({row["seed"] for row in SPLIT_SETTINGS.values()}) == 3
    with pytest.raises(ValueError):
        generate_scene("confirm", 0)


def test_exact_historical_p0_is_imported_not_reimplemented() -> None:
    model = build_exact_p0()
    receipt = architecture_receipt()
    assert isinstance(model, EndpointPairClassifier)
    assert receipt["architecture_reimplemented"] is False
    assert receipt["class_name"] == "path_completion.pair_classifier.EndpointPairClassifier"
    assert receipt["parameter_count"] == sum(parameter.numel() for parameter in model.parameters())


def test_candidate_miss_is_not_labeled_none_or_used_in_relation_loss() -> None:
    candidates = tuple(_Candidate(float(index)) for index in range(5))
    assert selected_training_ranks(STATUS_MISS, -1, candidates) == ()
    assert selected_training_ranks(STATUS_PRESENT, 2, candidates) == (2, 0, 1, 3)
    assert selected_training_ranks(STATUS_NONE, -1, candidates) == (0, 1, 2, 3)


def test_corridor_has_historical_semantics_and_visible_context() -> None:
    model_input = np.zeros((3, 96, 96), dtype=np.float32)
    model_input[0, 48, 8:90] = 1.0
    probability = np.zeros((96, 96), dtype=np.float32)
    probability[48, 8:35] = 1.0
    probability[48, 50:90] = 1.0
    crop = branch_landing_corridor(model_input, probability, (48.0, 34.0), (48.0, 55.0))
    assert crop.shape == (6, 32, 64)
    marker_profile = crop[4].max(axis=0)
    peak_columns = np.flatnonzero(marker_profile > 0.90)
    assert peak_columns.min() < 32 < peak_columns.max()
    assert crop[0, :, :8].max() > 0 and crop[0, :, -8:].max() > 0
    assert "truth" not in inspect.signature(branch_landing_corridor).parameters
    assert "target" not in inspect.signature(branch_landing_corridor).parameters


def test_source_balanced_loss_is_finite_for_variable_candidate_sets() -> None:
    logits = torch.tensor([[2.0, -1.0, -9.0], [-0.5, 0.2, -0.1]])
    labels = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    positive = torch.tensor([True, False])
    loss = source_balanced_loss(logits, labels, mask, positive)
    assert torch.isfinite(loss) and float(loss) > 0


def test_downstream_phases_are_locked() -> None:
    assert PROTOCOL["authorized_phases"] == ["E1", "E2", "E3"]
    for key in ("transformer", "path", "confirm_metrics", "cracks", "expert", "seeds_42_43"):
        assert PROTOCOL["locks"][key] is True
