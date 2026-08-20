"""Public corrected evaluator facade kept separate from frozen legacy outputs."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from synthetic.structural_metrics_corrected import (
    compute_family_a_metrics,
    compute_route_metrics,
    geometry_only_minimum_angle_heuristic,
    unavailable_route_metrics,
)


CORRECTED_EVALUATOR_VERSION = "2.1"
ORIGINAL_TEST_RANGE = (0, 2000)
REPLACEMENT_TEST_RANGE = (2000, 4000)
PRIMARY_FALSE_BRIDGE_COVERAGE_THRESHOLD = 0.50
FALSE_BRIDGE_SENSITIVITY_THRESHOLDS = (0.25, 0.50, 0.75)


def evaluate_sample_corrected(
    predicted_visible_mask: np.ndarray,
    target: Mapping[str, Any],
    *,
    predicted_completion_mask: np.ndarray | None = None,
    predicted_instance_masks: np.ndarray | None = None,
    predicted_orientation: np.ndarray | None = None,
    routing_probabilities: np.ndarray | None = None,
    has_transport_output: bool = False,
    include_geometry_diagnostic: bool = False,
) -> dict[str, Any]:
    """Evaluate one sample without assigning geometry-derived routing to a model."""

    if has_transport_output != (routing_probabilities is not None):
        raise ValueError("Transport availability and routing probabilities must agree")
    family_a = compute_family_a_metrics(
        predicted_visible_mask,
        target,
        predicted_completion_mask=predicted_completion_mask,
        predicted_instance_masks=predicted_instance_masks,
        predicted_continuation_scores=None,
        predicted_orientation=predicted_orientation,
        bridge_coverage_threshold=PRIMARY_FALSE_BRIDGE_COVERAGE_THRESHOLD,
    )
    family_b = (
        compute_route_metrics(np.asarray(routing_probabilities), target)
        if has_transport_output
        else unavailable_route_metrics()
    )
    return {
        "corrected_evaluator_version": CORRECTED_EVALUATOR_VERSION,
        "family_a": family_a,
        "family_b": family_b,
        "family_c": geometry_only_minimum_angle_heuristic(target) if include_geometry_diagnostic else None,
    }
