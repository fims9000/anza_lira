from __future__ import annotations

import numpy as np
import torch

from synthetic.crossing_trace_bench import generate_sample
from synthetic.evaluation import (
    EVALUATION_PROTOCOL,
    continuation_probabilities,
    evaluation_protocol_hash,
    evaluate_frozen_test,
    minimum_angle_continuation_scores,
)


def test_continuation_probabilities_normalize_only_over_eligible_destinations() -> None:
    logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 0.0, -1.0], [0.5, 0.2, 0.0]])
    eligible = torch.tensor(
        [[False, True, True], [True, False, False], [True, False, False]]
    )
    probability = continuation_probabilities(logits, eligible)
    assert torch.allclose(probability.sum(dim=-1), torch.ones(3))
    assert torch.all(probability[~eligible] == 0)


def test_minimum_angle_readout_succeeds_on_trivial_x_but_fails_adversarial_lineage() -> None:
    trivial = generate_sample("validation", 150, case="x_junction")
    adversarial = generate_sample("validation", 150, case="nontrivial_pairing")
    trivial_scores = minimum_angle_continuation_scores(trivial)
    adversarial_scores = minimum_angle_continuation_scores(adversarial)
    assert np.array_equal(trivial_scores.astype(bool), trivial["continuation_relation_matrix"])
    assert not np.array_equal(
        adversarial_scores.astype(bool), adversarial["continuation_relation_matrix"]
    )


def test_evaluation_protocol_is_validation_only_and_frozen() -> None:
    assert EVALUATION_PROTOCOL["split"] == "validation"
    assert EVALUATION_PROTOCOL["indices"] == list(range(256))
    assert EVALUATION_PROTOCOL["test_stream"] == "FROZEN_UNOPENED"
    assert len(evaluation_protocol_hash()) == 16


def test_test_evaluation_requires_a_frozen_candidate(tmp_path) -> None:
    try:
        evaluate_frozen_test(tmp_path, device="cpu")
    except ValueError as error:
        assert "frozen" in str(error).lower()
    else:
        raise AssertionError("Test evaluation must fail closed without a freeze artifact")
