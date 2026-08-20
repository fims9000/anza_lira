"""Oracle checks for the frozen structural-completion endpoint.

These tests validate measurement semantics only.  They intentionally reuse the
legacy generator/evaluator and must pass before the independent v3 stream or
any context-repair training is opened.
"""

from __future__ import annotations

import numpy as np

from synthetic.crossing_trace_bench import generate_sample
from synthetic.structural_metrics import compute_structural_metrics


def _evaluate(sample: dict, completion: np.ndarray) -> dict[str, float | int]:
    return compute_structural_metrics(
        sample["visible_fault_mask"],
        sample,
        predicted_completion_mask=np.asarray(completion, dtype=bool),
    )


def test_visible_mask_does_not_recover_a_positive_gap_or_bridge_a_negative_gap() -> None:
    positive = generate_sample("validation", 901, case="fault_with_gap")
    negative = generate_sample("validation", 902, case="negative_gap")

    positive_metrics = _evaluate(positive, positive["visible_fault_mask"])
    negative_metrics = _evaluate(negative, negative["visible_fault_mask"])

    assert positive_metrics["gap_recovery_rate"] == 0.0
    assert negative_metrics["false_bridge_rate"] == 0.0


def test_positive_latent_completion_recovers_gap_without_false_bridge() -> None:
    positive = generate_sample("validation", 903, case="fault_with_gap")
    negative = generate_sample("validation", 904, case="negative_gap")

    positive_metrics = _evaluate(positive, positive["latent_fault_mask"])
    negative_metrics = _evaluate(negative, negative["latent_fault_mask"])

    assert positive_metrics["gap_recovery_rate"] == 1.0
    assert negative_metrics["false_bridge_rate"] == 0.0


def test_explicitly_closing_a_negative_gap_is_a_false_bridge() -> None:
    negative = generate_sample("validation", 905, case="negative_gap")
    closed = np.asarray(negative["latent_fault_mask"], dtype=bool) | np.asarray(
        negative["negative_gap_mask"], dtype=bool
    )

    metrics = _evaluate(negative, closed)

    assert metrics["false_bridge_rate"] == 1.0
    assert metrics["false_bridge_count"] == 1


def test_all_foreground_is_always_a_false_bridge_on_negative_gap() -> None:
    negative = generate_sample("validation", 906, case="negative_gap")
    all_foreground = np.ones_like(negative["visible_fault_mask"], dtype=bool)

    metrics = _evaluate(negative, all_foreground)

    assert metrics["false_bridge_rate"] == 1.0
    assert metrics["false_bridge_count"] == metrics["negative_gap_count"] == 1
